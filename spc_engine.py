"""
SPC Engine
==========
Statistical Process Control primitives for batch pharmaceutical manufacturing.

Implements the tools a process/quality engineer actually uses:
  - I-MR (Individuals & Moving Range) control charts
  - Robust Phase I baseline establishment (iterative trim of out-of-control pts)
  - Western Electric rules 1-4 (shift/instability detection via sigma zones)
  - Nelson Rule 3 (sustained monotonic trend)
  - Rolling-slope rule (regression-based trend; catches sub-noise slow drift)
  - Cpk / Ppk process capability indices (within- vs overall-sigma)

Design note: control limits are set from a QUALIFICATION BASELINE (an in-control
reference period), then the full production series is monitored against those
fixed limits. Establishing limits from data that already contains special-cause
variation inflates the limits and masks the very events you need to catch.

References in spirit: ICH Q8/Q9/Q10 (QbD, quality risk mgmt), Montgomery's
"Statistical Quality Control", AIAG SPC manual constants.
"""

import numpy as np
import pandas as pd

# I-MR constants for a moving range of n=2 consecutive individuals
D2 = 1.128        # sigma_hat = MR_bar / d2
D4 = 3.267        # MR chart UCL factor  (D3 = 0 for n=2, so MR LCL = 0)


# --------------------------------------------------------------------------- #
# Sigma / baseline estimation
# --------------------------------------------------------------------------- #
def moving_range_bar(values):
    """Average moving range of consecutive individuals."""
    v = np.asarray(values, dtype=float)
    if v.size < 2:
        return np.nan
    return np.mean(np.abs(np.diff(v)))


def establish_baseline(values, trim=True, max_iter=10):
    """
    Phase I: estimate the in-control center line and within-process sigma.

    If trim=True, iteratively removes points beyond the current 3-sigma limits
    and recomputes until the kept set stabilizes (or `max_iter` is reached), so
    an isolated excursion sitting in the baseline window doesn't inflate the
    limits. On well-behaved data this converges in a single pass; the loop just
    guarantees a fixed point when one excursion unmasks another.

    Returns (center, sigma_within).
    """
    v = np.asarray(values, dtype=float)
    kept = v
    center = float(np.mean(kept))
    sigma = moving_range_bar(kept) / D2

    if trim:
        for _ in range(max_iter):
            if not (np.isfinite(sigma) and sigma > 0):
                break
            ucl, lcl = center + 3 * sigma, center - 3 * sigma
            keep = (kept <= ucl) & (kept >= lcl)
            if not (2 <= keep.sum() < kept.size):   # nothing (more) to trim
                break
            kept = kept[keep]
            center = float(np.mean(kept))
            sigma = moving_range_bar(kept) / D2
    return center, sigma


# --------------------------------------------------------------------------- #
# Control rules
# --------------------------------------------------------------------------- #
def western_electric(values, center, sigma):
    """
    Apply Western Electric rules 1-4. Returns dict {index: [rule strings]}.

    Zones (in sigma units from center):  A = 2-3, B = 1-2, C = 0-1.
      Rule 1: any point beyond 3-sigma.
      Rule 2: 2 of 3 consecutive points beyond 2-sigma, same side.
      Rule 3: 4 of 5 consecutive points beyond 1-sigma, same side.
      Rule 4: 8 consecutive points on one side of the center line.
    """
    v = np.asarray(values, dtype=float)
    n = v.size
    flags = {i: [] for i in range(n)}

    s1h, s2h, s3h = center + sigma, center + 2 * sigma, center + 3 * sigma
    s1l, s2l, s3l = center - sigma, center - 2 * sigma, center - 3 * sigma

    # Rule 1
    for i in range(n):
        if v[i] > s3h or v[i] < s3l:
            side = "high" if v[i] > s3h else "low"
            flags[i].append(f"WE1: beyond 3sigma ({side})")

    # Rule 2: 2 of 3 beyond 2-sigma same side
    for i in range(2, n):
        w = v[i - 2:i + 1]
        if np.sum(w > s2h) >= 2:
            flags[i].append("WE2: 2/3 beyond 2sigma (high)")
        if np.sum(w < s2l) >= 2:
            flags[i].append("WE2: 2/3 beyond 2sigma (low)")

    # Rule 3: 4 of 5 beyond 1-sigma same side
    for i in range(4, n):
        w = v[i - 4:i + 1]
        if np.sum(w > s1h) >= 4:
            flags[i].append("WE3: 4/5 beyond 1sigma (high)")
        if np.sum(w < s1l) >= 4:
            flags[i].append("WE3: 4/5 beyond 1sigma (low)")

    # Rule 4: 8 consecutive on one side of center
    for i in range(7, n):
        w = v[i - 7:i + 1]
        if np.all(w > center):
            flags[i].append("WE4: 8 consecutive above center")
        if np.all(w < center):
            flags[i].append("WE4: 8 consecutive below center")

    return flags


def nelson_rule3(values, k=6):
    """
    Nelson Rule 3: `k` points in a row, all steadily increasing or all steadily
    decreasing — a sustained monotonic trend. k=6 is the textbook default.

    Scope note: a strictly-monotonic rule only fires on trends that are clean
    relative to the noise. A slow drift whose per-batch step is small compared
    with the common-cause sigma (e.g. the tooling-wear drift planted in this
    project, ~1/20 sigma per batch) will NOT trip it — that kind of drift is
    caught by the runs-based rules (WE Rule 4 / Nelson Rule 2), which accumulate
    a one-sided signal instead of demanding monotonicity. This rule is here for
    genuine monotonic trends; it is deliberately not the tooling-wear detector.
    """
    v = np.asarray(values, dtype=float)
    n = v.size
    flags = {i: [] for i in range(n)}
    for i in range(k - 1, n):
        d = np.diff(v[i - k + 1:i + 1])
        if np.all(d > 0):
            flags[i].append(f"N3: {k} rising")
        elif np.all(d < 0):
            flags[i].append(f"N3: {k} falling")
    return flags


def rolling_slope_rule(values, window=24, t_thresh=3.0):
    """
    Regression-based trend rule. Over a trailing window of `window` points, fit
    an OLS line (value vs. index) and flag the point when the fitted slope is
    statistically significant: |t| > t_thresh with df = window - 2.

    Why this exists alongside Nelson Rule 3: a strictly-monotonic rule needs the
    trend to overpower the noise at every single step, so it cannot see a slow
    drift whose per-batch step is a small fraction of sigma. A regression slope
    instead *accumulates* that sub-noise signal across the window, so it CAN
    catch such a drift (here it independently catches the planted tooling wear,
    confirming WE Rule 4's runs-based catch). As a side benefit it also fires on
    a step change, which produces a large local slope in the straddling window.

    window=24 / t_thresh=3.0 were chosen so the rule catches the planted drift
    with no false alarms in the in-control baseline region; a shorter window
    starts tripping on common-cause wiggles.
    """
    v = np.asarray(values, dtype=float)
    n = v.size
    flags = {i: [] for i in range(n)}
    if n < window or window < 3:
        return flags
    x = np.arange(window, dtype=float)
    xc = x - x.mean()
    sxx = float((xc ** 2).sum())
    for i in range(window - 1, n):
        y = v[i - window + 1:i + 1]
        slope = float((xc * (y - y.mean())).sum() / sxx)
        resid = y - (y.mean() + slope * xc)
        s2 = float((resid ** 2).sum()) / (window - 2)
        if s2 <= 0:
            continue
        se = (s2 / sxx) ** 0.5
        t = slope / se if se > 0 else 0.0
        if abs(t) > t_thresh:
            direction = "rising" if slope > 0 else "falling"
            flags[i].append(f"SLOPE: {window}-pt trend {direction} (t={t:.1f})")
    return flags


# --------------------------------------------------------------------------- #
# Episode collapsing (per-batch flags -> out-of-control events)
# --------------------------------------------------------------------------- #
# Rule severity ranking, worst first. Used to score an episode's peak state.
_SEVERITY_ORDER = ["OOS", "WE1", "WE4", "WE2", "WE3", "N3", "SLOPE"]


def _peak_severity(rules_concat):
    """Human-readable worst state seen in a run of rule strings."""
    if "OOS" in rules_concat:
        return "out-of-spec"
    if "WE1" in rules_concat:
        return "beyond-3-sigma"
    return "out-of-trend"


def episodes(flag_df, reset=3, id_col="batch_id", date_col="date"):
    """
    Collapse per-batch flags into out-of-control EPISODES.

    A run of control-rule violations is not N independent events. Once a
    process shifts and is not re-centered, every subsequent point flags — so
    counting flagged *batches* massively overstates how many times the process
    actually signalled. An episode instead opens at the first flagged batch and
    stays open until `reset` consecutive in-control batches are seen, then
    closes. This yields the ONSET of each excursion (what an engineer acts on),
    not a wall of individual flags.

    Returns a DataFrame, one row per episode:
      onset_batch, onset_date, end_batch, span (batches onset->close),
      n_flagged (flagged batches within the episode), peak_severity,
      onset_rules (rules that first fired), rules (all codes seen).
    """
    rows = flag_df.reset_index(drop=True)
    n = len(rows)
    eps, cur, clean = [], None, 0

    def _close(ep):
        codes = sorted(
            {c for c in _SEVERITY_ORDER if c in ep["_all"]},
            key=_SEVERITY_ORDER.index,
        )
        eps.append({
            "onset_batch": ep["onset_batch"],
            "onset_date": ep["onset_date"],
            "end_batch": ep["end_batch"],
            "span": ep["end_idx"] - ep["onset_idx"] + 1,
            "n_flagged": ep["n_flagged"],
            "peak_severity": _peak_severity(ep["_all"]),
            "onset_rules": ep["onset_rules"],
            "rules": "; ".join(codes),
        })

    for i in range(n):
        flagged = bool(rows["flagged"].iloc[i])
        rule_str = str(rows["rules"].iloc[i])
        if flagged:
            if cur is None:
                cur = {"onset_idx": i, "onset_batch": rows[id_col].iloc[i],
                       "onset_date": rows[date_col].iloc[i],
                       "onset_rules": rule_str, "n_flagged": 0, "_all": ""}
            cur["n_flagged"] += 1
            cur["end_idx"] = i
            cur["end_batch"] = rows[id_col].iloc[i]
            cur["_all"] += " " + rule_str
            clean = 0
        elif cur is not None:
            clean += 1
            if clean >= reset:
                _close(cur)
                cur, clean = None, 0
    if cur is not None:
        _close(cur)

    return pd.DataFrame(eps, columns=[
        "onset_batch", "onset_date", "end_batch", "span", "n_flagged",
        "peak_severity", "onset_rules", "rules",
    ])


# --------------------------------------------------------------------------- #
# Capability
# --------------------------------------------------------------------------- #
def capability(values, lsl=None, usl=None, sigma_within=None):
    """
    Cp/Cpk (within-sigma, short-term potential) and Pp/Ppk (overall-sigma,
    long-term actual performance). Supports one- or two-sided specs.
    """
    v = np.asarray(values, dtype=float)
    mean = float(np.mean(v))
    sigma_overall = float(np.std(v, ddof=1))
    out = {"mean": round(mean, 3),
           "sigma_within": None if sigma_within is None else round(sigma_within, 4),
           "sigma_overall": round(sigma_overall, 4)}

    def _cpk(sig):
        if sig is None or sig <= 0:
            return None
        if lsl is not None and usl is not None:
            return min(usl - mean, mean - lsl) / (3 * sig)
        if lsl is not None:
            return (mean - lsl) / (3 * sig)
        if usl is not None:
            return (usl - mean) / (3 * sig)
        return None

    def _cp(sig):
        if sig is None or sig <= 0 or lsl is None or usl is None:
            return None
        return (usl - lsl) / (6 * sig)

    cp, cpk = _cp(sigma_within), _cpk(sigma_within)
    pp, ppk = _cp(sigma_overall), _cpk(sigma_overall)
    out["Cp"] = None if cp is None else round(cp, 3)
    out["Cpk"] = None if cpk is None else round(cpk, 3)
    out["Pp"] = None if pp is None else round(pp, 3)
    out["Ppk"] = None if ppk is None else round(ppk, 3)
    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_spc(df, column, baseline_n=80, lsl=None, usl=None, date_col="date",
            id_col="batch_id"):
    """
    Full SPC analysis of one column.

    baseline_n : number of leading rows used as the qualification baseline.
    Returns a dict with limits, capability snapshots, and a per-batch flag frame.
    """
    values = df[column].to_numpy(dtype=float)
    baseline = values[:baseline_n]

    center, sigma = establish_baseline(baseline, trim=True)
    ucl, lcl = center + 3 * sigma, center - 3 * sigma

    we = western_electric(values, center, sigma)
    tr = nelson_rule3(values, k=6)
    sl = rolling_slope_rule(values)

    rows = []
    for i in range(len(values)):
        rule_hits = we[i] + tr[i] + sl[i]
        oos = False
        if lsl is not None and values[i] < lsl:
            oos = True
        if usl is not None and values[i] > usl:
            oos = True
        if oos:
            rule_hits = ["OOS: outside spec"] + rule_hits
        rows.append({
            id_col: df[id_col].iloc[i],
            date_col: df[date_col].iloc[i],
            "value": values[i],
            "flagged": len(rule_hits) > 0,
            "rules": "; ".join(rule_hits),
        })
    flag_df = pd.DataFrame(rows)
    episode_df = episodes(flag_df, id_col=id_col, date_col=date_col)

    cap_qual = capability(baseline, lsl, usl, sigma_within=sigma)
    cap_all = capability(values, lsl, usl, sigma_within=sigma)

    return {
        "column": column,
        "center": center, "sigma": sigma, "ucl": ucl, "lcl": lcl,
        "lsl": lsl, "usl": usl, "baseline_n": baseline_n,
        "capability_qualification": cap_qual,
        "capability_full_production": cap_all,
        "flags": flag_df,
        "episodes": episode_df,
    }
