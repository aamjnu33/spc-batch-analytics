"""
SPC Engine
==========
Statistical Process Control primitives for batch pharmaceutical manufacturing.

Implements the tools a process/quality engineer actually uses:
  - I-MR (Individuals & Moving Range) control charts
  - Robust Phase I baseline establishment (iterative trim of out-of-control pts)
  - Western Electric rules 1-4 (shift/instability detection via sigma zones)
  - Trend rule (sustained drift, e.g. tooling wear)
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


def establish_baseline(values, trim=True):
    """
    Phase I: estimate the in-control center line and within-process sigma.

    If trim=True, performs one iterative pass that removes points beyond the
    initial 3-sigma limits and recomputes, so an isolated excursion sitting in
    the baseline window doesn't inflate the limits.

    Returns (center, sigma_within).
    """
    v = np.asarray(values, dtype=float)
    center = float(np.mean(v))
    sigma = moving_range_bar(v) / D2

    if trim and np.isfinite(sigma) and sigma > 0:
        ucl, lcl = center + 3 * sigma, center - 3 * sigma
        keep = (v <= ucl) & (v >= lcl)
        if 2 <= keep.sum() < v.size:          # something was trimmed
            v_kept = v[keep]
            center = float(np.mean(v_kept))
            sigma = moving_range_bar(v_kept) / D2
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


def trend_rule(values, run=7):
    """Sustained monotonic run of `run` points (catches gradual drift)."""
    v = np.asarray(values, dtype=float)
    n = v.size
    flags = {i: [] for i in range(n)}
    for i in range(run - 1, n):
        d = np.diff(v[i - run + 1:i + 1])
        if np.all(d > 0):
            flags[i].append(f"TREND: {run} rising")
        elif np.all(d < 0):
            flags[i].append(f"TREND: {run} falling")
    return flags


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
    tr = trend_rule(values, run=7)

    rows = []
    for i in range(len(values)):
        rule_hits = we[i] + tr[i]
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

    cap_qual = capability(baseline, lsl, usl, sigma_within=sigma)
    cap_all = capability(values, lsl, usl, sigma_within=sigma)

    return {
        "column": column,
        "center": center, "sigma": sigma, "ucl": ucl, "lcl": lcl,
        "lsl": lsl, "usl": usl, "baseline_n": baseline_n,
        "capability_qualification": cap_qual,
        "capability_full_production": cap_all,
        "flags": flag_df,
    }
