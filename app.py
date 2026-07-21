"""
Batch Process Analytics — SPC + ML dashboard
============================================
Interactive Quality-by-Design dashboard for solid-oral-dose (tablet) batch
manufacturing. Combines Statistical Process Control with an explainable ML
layer to detect when the process drifts and explain why.

Data is SYNTHETIC (generated with known ground-truth events) — no proprietary
or GMP-protected data is used. See generate_batch_data.py.

Run locally:   streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from spc_engine import run_spc, capability

# --------------------------------------------------------------------------- #
# Page config + design system
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Batch Process Analytics",
    page_icon="\u25C9",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Palette (status colours carry meaning)
INK = "#0E2A38"        # header band / deep teal-navy
TEAL = "#12868A"       # primary accent, in-control
GOOD = "#2E8B6F"       # capable / good
AMBER = "#D98A29"      # out-of-trend (WE rules, no spec breach)
RED = "#C0453B"        # out-of-spec / beyond 3 sigma
SURFACE = "#F4F7F9"
MUTED = "#5B6B78"
GRID = "#DDE5EA"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"], .stMarkdown, .stText {{
        font-family: 'IBM Plex Sans', sans-serif;
    }}
    .block-container {{ padding-top: 1.2rem; max-width: 1300px; }}

    .app-header {{
        background: linear-gradient(100deg, {INK} 0%, #123845 100%);
        border-radius: 12px; padding: 22px 26px; color: #EAF2F4;
        margin-bottom: 6px;
    }}
    .app-header h1 {{
        font-size: 1.55rem; font-weight: 700; margin: 0; letter-spacing: -0.01em;
        color: #FFFFFF;
    }}
    .app-header p {{ margin: 4px 0 0 0; color: #A9C2CB; font-size: 0.9rem; }}
    .app-meta {{
        font-family: 'IBM Plex Mono', monospace; font-size: 0.74rem;
        color: #8FB0BA; margin-top: 10px; letter-spacing: 0.02em;
    }}
    .synthetic-badge {{
        display: inline-block; background: rgba(217,138,41,0.15);
        color: {AMBER}; border: 1px solid rgba(217,138,41,0.4);
        padding: 2px 9px; border-radius: 20px; font-size: 0.68rem;
        font-family: 'IBM Plex Mono', monospace; letter-spacing: 0.03em;
    }}

    .kpi-card {{
        background: #FFFFFF; border: 1px solid {GRID}; border-radius: 10px;
        padding: 15px 18px; height: 100%;
    }}
    .kpi-label {{
        font-size: 0.72rem; color: {MUTED}; text-transform: uppercase;
        letter-spacing: 0.06em; font-weight: 600;
    }}
    .kpi-value {{
        font-family: 'IBM Plex Mono', monospace; font-size: 1.85rem;
        font-weight: 600; color: {INK}; line-height: 1.1; margin-top: 4px;
    }}
    .kpi-sub {{ font-size: 0.74rem; color: {MUTED}; margin-top: 2px; }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
    .stTabs [data-baseweb="tab"] {{
        font-weight: 600; font-size: 0.9rem; color: {MUTED};
        padding: 8px 16px;
    }}
    .stTabs [aria-selected="true"] {{ color: {INK}; }}

    .rule-pill {{
        font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem;
        padding: 2px 8px; border-radius: 4px; margin-right: 5px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Spec limits per CQA
# --------------------------------------------------------------------------- #
SPECS = {
    "dissolution_30min_pct": dict(label="Dissolution @ 30 min (%)", lsl=80, usl=None),
    "hardness_kp":           dict(label="Tablet hardness (kp)",     lsl=4,  usl=10),
    "assay_pct":             dict(label="Assay / potency (%)",       lsl=95, usl=105),
    "tablet_weight_mg":      dict(label="Tablet weight (mg)",        lsl=245, usl=255),
    "content_uniformity_rsd": dict(label="Content uniformity (%RSD)", lsl=None, usl=6),
}

CPP_FEATURES = [
    "granulation_water_pct", "massing_time_min", "inlet_air_temp_c",
    "blend_time_min", "compression_force_kn", "press_speed_rpm",
    "api_d90_um", "room_humidity_pct",
]
PRETTY = {
    "api_d90_um": "API particle size (d90)", "compression_force_kn": "Compression force",
    "room_humidity_pct": "Room humidity", "press_speed_rpm": "Press speed",
    "blend_time_min": "Blend time", "granulation_water_pct": "Granulation water",
    "massing_time_min": "Massing time", "inlet_air_temp_c": "Inlet air temp",
}


# --------------------------------------------------------------------------- #
# Data + compute (cached)
# --------------------------------------------------------------------------- #
@st.cache_data
def load_data():
    if not os.path.exists("batch_data.csv"):
        import generate_batch_data as g
        df, gt = g.generate()
        df.to_csv("batch_data.csv", index=False)
        gt.to_csv("ground_truth.csv", index=False)
    df = pd.read_csv("batch_data.csv", parse_dates=["date"])
    return df


@st.cache_data
def spc_all(_df):
    out = {}
    for col, spec in SPECS.items():
        out[col] = run_spc(_df, col, baseline_n=80, lsl=spec["lsl"], usl=spec["usl"])
    return out


@st.cache_resource
def train_model(_df):
    from sklearn.model_selection import train_test_split, cross_val_score, KFold
    from sklearn.metrics import r2_score, mean_squared_error
    from xgboost import XGBRegressor
    import shap

    X, y = _df[CPP_FEATURES], _df["dissolution_30min_pct"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.05,
                         subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=2)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    r2 = r2_score(yte, pred)
    rmse = mean_squared_error(yte, pred) ** 0.5
    cv = cross_val_score(model, X, y, cv=KFold(5, shuffle=True, random_state=42),
                         scoring="r2")
    # Time-ordered holdout: the honest forecasting view (train past, predict
    # future). Much lower than the random split because d90 barely varies within
    # one lot regime — reported so the explanatory vs predictive distinction is
    # explicit rather than hidden.
    cut = int(len(_df) * 0.8)
    model_t = XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.05,
                           subsample=0.9, colsample_bytree=0.9, random_state=42,
                           n_jobs=2)
    model_t.fit(X.iloc[:cut], y.iloc[:cut])
    r2_time = r2_score(y.iloc[cut:], model_t.predict(X.iloc[cut:]))
    sv = shap.TreeExplainer(model).shap_values(X)
    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=CPP_FEATURES)
    direction = {}
    for j, f in enumerate(CPP_FEATURES):
        c = np.corrcoef(X[f].to_numpy(), sv[:, j])[0, 1]
        direction[f] = "lower" if c < 0 else "higher"
    drivers = pd.DataFrame({
        "feature": CPP_FEATURES,
        "impact": mean_abs.values,
        "raises_dissolution_when": [direction[f] for f in CPP_FEATURES],
    }).sort_values("impact", ascending=False).reset_index(drop=True)
    return dict(r2=r2, rmse=rmse, cv_mean=cv.mean(), cv_std=cv.std(),
                r2_time=r2_time, drivers=drivers, yte=yte.to_numpy(), pred=pred)


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #
def severity(rules):
    if not rules:
        return "ok"
    if "OOS" in rules or "WE1" in rules:
        return "oos"
    return "oot"


def control_chart(result, df):
    r = result
    flags = r["flags"]
    x = df["batch_id"]
    y = flags["value"]
    center, ucl, lcl, sigma = r["center"], r["ucl"], r["lcl"], r["sigma"]

    sev = flags["rules"].apply(severity)
    colours = {"ok": TEAL, "oot": AMBER, "oos": RED}
    marker_colors = [colours[s] for s in sev]
    marker_sizes = [6 if s == "ok" else 10 for s in sev]

    hover = [
        f"<b>{bid}</b><br>{r['column']}: {val:.2f}"
        + (f"<br><span style='color:#ffb'>{rl}</span>" if rl else "")
        for bid, val, rl in zip(x, y, flags["rules"])
    ]

    fig = go.Figure()
    # sigma zones (subtle)
    for k, opacity in [(3, 0.03), (2, 0.05), (1, 0.07)]:
        fig.add_hrect(y0=center - k * sigma, y1=center + k * sigma,
                      line_width=0, fillcolor=TEAL, opacity=opacity, layer="below")
    # limit lines
    for yval, label, dash, col in [
        (center, "CL", "solid", MUTED),
        (ucl, "UCL", "dash", RED), (lcl, "LCL", "dash", RED),
    ]:
        fig.add_hline(y=yval, line_dash=dash, line_color=col, line_width=1.2,
                      annotation_text=f"{label} {yval:.2f}",
                      annotation_position="right",
                      annotation_font_size=10, annotation_font_color=col)
    # spec limits
    for s, lab in [(r["lsl"], "LSL"), (r["usl"], "USL")]:
        if s is not None:
            fig.add_hline(y=s, line_dash="dot", line_color="#7A2E2E", line_width=1,
                          annotation_text=f"{lab} {s}", annotation_position="left",
                          annotation_font_size=10, annotation_font_color="#7A2E2E")

    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines+markers", line=dict(color=INK, width=1),
        marker=dict(color=marker_colors, size=marker_sizes,
                    line=dict(color="white", width=0.8)),
        hovertext=hover, hoverinfo="text", name=r["column"],
    ))
    fig.add_vline(x=r["baseline_n"] - 0.5, line_dash="dot", line_color=MUTED,
                  line_width=1,
                  annotation_text="baseline \u2192 monitoring",
                  annotation_position="top left", annotation_font_size=9,
                  annotation_font_color=MUTED)

    fig.update_layout(
        height=380, margin=dict(l=10, r=60, t=20, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="IBM Plex Sans", color=INK, size=12),
        xaxis=dict(showgrid=False, showticklabels=False, title="Batch (time order)"),
        yaxis=dict(gridcolor=GRID, zeroline=False),
        showlegend=False,
    )
    return fig


def capability_plot(values, spec, cap):
    lsl, usl = spec["lsl"], spec["usl"]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=values, nbinsx=30, marker_color=TEAL,
                               opacity=0.75, name="batches"))
    for s, lab, col in [(lsl, "LSL", RED), (usl, "USL", RED),
                        (cap["mean"], "mean", INK)]:
        if s is not None:
            fig.add_vline(x=s, line_dash="dash" if lab != "mean" else "solid",
                          line_color=col, line_width=1.5,
                          annotation_text=lab, annotation_font_size=10,
                          annotation_font_color=col)
    fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="IBM Plex Sans", color=INK, size=12),
        xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID, title="batches"),
        showlegend=False, bargap=0.05,
    )
    return fig


def kpi(label, value, sub="", color=INK):
    return f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value" style="color:{color}">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>"""


# --------------------------------------------------------------------------- #
# App body
# --------------------------------------------------------------------------- #
df = load_data()
spc = spc_all(df)

n_batches = len(df)
product = df["product"].iloc[0]
line = df["line"].iloc[0]
date_range = f"{df['date'].min():%b %Y} \u2013 {df['date'].max():%b %Y}"

diss = spc["dissolution_30min_pct"]
n_flagged = int(diss["flags"]["flagged"].sum())
n_episodes = len(diss["episodes"])
n_oos = int(diss["flags"]["rules"].str.contains("OOS").sum())
ppk_full = diss["capability_full_production"]["Ppk"]
ppk_qual = diss["capability_qualification"]["Ppk"]

# Header
st.markdown(
    f"""
    <div class="app-header">
      <h1>Batch Process Analytics &nbsp;<span style="font-size:0.8rem;color:{TEAL}">SPC &times; ML</span></h1>
      <p>Quality-by-Design monitoring for solid-oral-dose manufacturing —
         detect process drift, explain root cause.</p>
      <div class="app-meta">{product} &nbsp;|&nbsp; {line} &nbsp;|&nbsp;
         {n_batches} batches &nbsp;|&nbsp; {date_range} &nbsp;
         <span class="synthetic-badge">SYNTHETIC DEMO DATA</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# KPI row
c1, c2, c3, c4 = st.columns(4)
c1.markdown(kpi("Batches monitored", f"{n_batches}", "one row per batch"),
            unsafe_allow_html=True)
c2.markdown(kpi("OOC episodes", f"{n_episodes}",
                f"{n_flagged} flagged batches", AMBER),
            unsafe_allow_html=True)
c3.markdown(kpi("OOS batches", f"{n_oos}", "below 80% spec", RED),
            unsafe_allow_html=True)
ppk_col = GOOD if ppk_full >= 1.33 else (AMBER if ppk_full >= 1.0 else RED)
c4.markdown(kpi("Ppk (production)", f"{ppk_full:.2f}",
                f"was {ppk_qual:.2f} at qualification", ppk_col),
            unsafe_allow_html=True)

st.write("")

tab1, tab2, tab3, tab4 = st.tabs(
    ["  Control charts  ", "  Capability  ", "  Process drivers  ", "  Batch explorer  "]
)

# ---- Tab 1: control charts ------------------------------------------------ #
with tab1:
    col = st.selectbox("Quality attribute", list(SPECS.keys()),
                       format_func=lambda c: SPECS[c]["label"])
    r = spc[col]
    st.plotly_chart(control_chart(r, df), width="stretch",
                    config={"displayModeBar": False})

    left, right = st.columns([1, 1])
    with left:
        st.markdown(
            f"<span class='rule-pill' style='background:{TEAL}22;color:{TEAL}'>&#9679; in control</span>"
            f"<span class='rule-pill' style='background:{AMBER}22;color:{AMBER}'>&#9679; out of trend (WE rule)</span>"
            f"<span class='rule-pill' style='background:{RED}22;color:{RED}'>&#9679; beyond 3&sigma; / OOS</span>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Control limits are set from the first 80 qualification batches "
            "(robust, outlier-trimmed), then held fixed to monitor production."
        )
    with right:
        eps = r["episodes"]
        flagged = r["flags"][r["flags"]["flagged"]][["batch_id", "value", "rules"]]
        st.caption(
            f"{len(eps)} out-of-control episode(s) — the process signalled "
            f"{len(eps)} time(s), collapsing {len(flagged)} flagged batches into "
            "onsets. Once a process shifts and is not re-centered, every "
            "subsequent batch flags, so episode onsets — not raw flag counts — "
            "are what an engineer acts on. Western Electric rules 1–4 + "
            "Nelson-3 + rolling-slope trend + spec checks."
        )

    eps = r["episodes"]
    if len(eps):
        show_eps = eps[[
            "onset_batch", "end_batch", "span", "n_flagged", "peak_severity",
            "onset_rules",
        ]].rename(columns={
            "onset_batch": "Onset", "end_batch": "Last flagged",
            "span": "Span (batches)", "n_flagged": "Flagged in episode",
            "peak_severity": "Peak severity", "onset_rules": "Rule(s) at onset",
        })
        with st.expander(f"View {len(eps)} episode onset(s)", expanded=True):
            st.dataframe(show_eps, width="stretch", hide_index=True)
    if len(flagged):
        with st.expander(f"View all {len(flagged)} flagged batches"):
            st.dataframe(flagged, width="stretch", hide_index=True)

# ---- Tab 2: capability ---------------------------------------------------- #
with tab2:
    col = st.selectbox("Quality attribute ", list(SPECS.keys()),
                       format_func=lambda c: SPECS[c]["label"], key="cap_sel")
    r = spc[col]
    spec = SPECS[col]
    cap_q = r["capability_qualification"]
    cap_a = r["capability_full_production"]

    cc = st.columns(4)
    def cap_card(colobj, label, val):
        if val is None:
            colobj.markdown(kpi(label, "\u2014"), unsafe_allow_html=True)
            return
        color = GOOD if val >= 1.33 else (AMBER if val >= 1.0 else RED)
        colobj.markdown(kpi(label, f"{val:.2f}", "", color), unsafe_allow_html=True)
    cap_card(cc[0], "Cpk (qual)", cap_q["Cpk"])
    cap_card(cc[1], "Ppk (qual)", cap_q["Ppk"])
    cap_card(cc[2], "Cpk (prod)", cap_a["Cpk"])
    cap_card(cc[3], "Ppk (prod)", cap_a["Ppk"])

    st.plotly_chart(capability_plot(df[col].to_numpy(), spec, cap_a),
                    width="stretch", config={"displayModeBar": False})
    st.caption(
        "Cpk uses within-batch (short-term) sigma; Ppk uses overall (long-term) "
        "sigma. A large Cpk\u2013Ppk gap signals a process that is capable "
        "short-term but unstable over time — the signature of undetected drift. "
        "Rule of thumb: \u22651.33 capable, 1.00\u20131.33 marginal, <1.00 not capable."
    )

# ---- Tab 3: process drivers ---------------------------------------------- #
with tab3:
    with st.spinner("Training XGBoost model and computing SHAP values\u2026"):
        m = train_model(df)

    k = st.columns(4)
    k[0].markdown(kpi("Explanatory R\u00b2", f"{m['r2']:.2f}", "random 80/20 split"),
                  unsafe_allow_html=True)
    k[1].markdown(kpi("5-fold CV R\u00b2", f"{m['cv_mean']:.2f}",
                      f"\u00b1{m['cv_std']:.2f}"), unsafe_allow_html=True)
    k[2].markdown(kpi("RMSE", f"{m['rmse']:.2f}%", "dissolution error"),
                  unsafe_allow_html=True)
    k[3].markdown(kpi("Forecast R\u00b2", f"{m['r2_time']:.2f}",
                      "time-ordered holdout", MUTED), unsafe_allow_html=True)

    st.markdown("##### What drives dissolution?")
    dr = m["drivers"].copy()
    dr["name"] = dr["feature"].map(PRETTY)
    top = dr.iloc[::-1]
    fig = go.Figure(go.Bar(
        x=top["impact"], y=top["name"], orientation="h",
        marker_color=[RED if f in ("api_d90_um", "room_humidity_pct",
                                    "compression_force_kn") else TEAL
                      for f in top["feature"]],
    ))
    fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="IBM Plex Sans", color=INK, size=12),
        xaxis=dict(title="mean |SHAP| — impact on dissolution (%)", gridcolor=GRID),
        yaxis=dict(gridcolor="white"),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    show = dr[["name", "impact", "raises_dissolution_when"]].rename(columns={
        "name": "Process parameter", "impact": "Impact (mean |SHAP|)",
        "raises_dissolution_when": "Dissolution improves when this is",
    })
    show["Impact (mean |SHAP|)"] = show["Impact (mean |SHAP|)"].round(3)
    st.dataframe(show, width="stretch", hide_index=True)
    st.caption(
        "The model was given only process parameters (no knowledge of the "
        "planted events) yet ranks API particle size, humidity, and compression "
        "force as the top drivers — matching the three root causes the SPC "
        "engine flagged. Tablet hardness is deliberately excluded (it's a CQA, "
        "not a knob), so the model surfaces **compression force** — the CPP that "
        "drives hardness — as the actionable upstream proxy. SPC finds *when*; "
        "ML explains *why*. This is an **explanatory** model for driver "
        "attribution: the random-split R² answers *which knobs matter*, while "
        "the lower time-ordered Forecast R² is the honest *predict-the-next-"
        "batch* view — within one API lot the top driver barely varies."
    )

# ---- Tab 4: batch explorer ----------------------------------------------- #
with tab4:
    cqa = st.selectbox("Overlay flags from", list(SPECS.keys()),
                       format_func=lambda c: SPECS[c]["label"], key="exp_sel")
    fl = spc[cqa]["flags"][["batch_id", "flagged", "rules"]]
    view = df.merge(fl, on="batch_id", how="left")
    only_flagged = st.checkbox("Show only flagged batches", value=False)
    if only_flagged:
        view = view[view["flagged"]]
    st.dataframe(view, width="stretch", hide_index=True, height=430)
    st.download_button("Download data (CSV)", view.to_csv(index=False),
                       "batch_data_flagged.csv", "text/csv")

st.write("")
st.caption(
    "Methodology: I-MR control charts, Western Electric rules, Cpk/Ppk capability, "
    "and XGBoost + SHAP driver analysis, framed by ICH Q8/Q9/Q10 (QbD). "
    "All data is synthetic and generated with known ground-truth events for "
    "validation. Built with Streamlit + Plotly."
)
