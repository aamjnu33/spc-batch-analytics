# Batch Process Analytics — SPC × ML for Pharmaceutical Manufacturing

**Live demo:** *(add your Streamlit Cloud URL here after deployment)*

A Quality-by-Design (QbD) analytics system for solid-oral-dose batch manufacturing
that answers the two questions every process team asks:

> **When did the process go out of control — and why?**

It combines a validated Statistical Process Control (SPC) engine (detects *when*)
with an explainable XGBoost model (explains *why*), wrapped in an interactive
Streamlit dashboard.

![Control chart dashboard](assets/dashboard_control_charts.png)

---

## Headline results

The system was validated against synthetic manufacturing data containing
**6 deliberately planted process events** with known ground truth:

| # | Planted event | Mechanism | Detected? | How |
|---|---|---|---|---|
| 1 | Tooling-wear drift (batch 90→) | Compression force ↑ → hardness ↑ | ✅ | WE Rule 4 (8 consecutive above CL) |
| 2 | API lot change (batch 140) | Coarser d90 → dissolution ↓ | ✅ | WE Rules 2/3 — flagged at the *first* shifted batch |
| 3–6 | 4 humidity excursions | RH spike → dissolution failure | ✅ 4/4 | WE Rule 1 (beyond 3σ); 2 also OOS |

**Detection rate: 6/6 (100%)** — see [`validate_spc.py`](validate_spc.py) for the reproducible check.

Because a process that shifts and is not re-centered flags *every* subsequent
batch, raw flag counts overstate how many times the process actually signalled.
The engine therefore collapses consecutive violations into **out-of-control
episodes** — each with an onset (what an engineer acts on), a span, and a peak
severity — so the dashboard reports a handful of episode onsets rather than a
wall of ~100 flagged batches. Detection is scored on whether each planted event
carries a signal; episodes are how that signal is summarised for action.

The capability analysis tells the process story through the **Ppk collapse and
the widening Cpk–Ppk gap**:

| Dissolution capability | Qualification | Full production |
|---|---|---|
| Cpk (within-σ) | 2.12 | 1.75 |
| **Ppk (overall-σ)** | **1.59** (capable) | **0.74** (not capable) |
| Cpk – Ppk gap | 0.53 | 1.01 |

Within-batch (short-term) variation stays tight, but the sustained mean shift
from the API lot change nearly doubles the *overall* σ — so Ppk collapses while
Cpk only dips. The **widening Cpk–Ppk gap is the textbook signature** of a
process that is capable short-term but unstable over time: exactly what the
planted drift and lot change created, and what the SPC engine caught. Note that
Cpk falls too (2.12 → 1.75), not just Ppk — because the mean drifts *toward* the
lower spec limit. It is a mean shift, not merely added variance, so the "Cpk
holds constant while Ppk collapses" idealisation doesn't quite apply here; the
honest signal is the gap.

The ML layer, given **only process parameters and no knowledge of the planted
events**, independently recovered all three root-cause mechanisms as its top
drivers, with physically correct directionality:

![SHAP driver analysis](assets/shap_summary.png)

| Rank | Driver | Direction | Maps to planted event |
|---|---|---|---|
| 1 | API particle size (d90) | coarser → lower dissolution | API lot change |
| 2 | Room humidity | higher → lower dissolution | Humidity excursions |
| 3 | Compression force | higher → lower dissolution | Tooling-wear drift |

Model performance is honest, not overfit: **test R² = 0.72, 5-fold CV R² = 0.718 ± 0.028**
(RMSE 2.6% dissolution).

---

## Why synthetic data?

Real batch manufacturing data is proprietary and GMP-protected — no responsible
scientist puts it on GitHub. Instead, this project simulates a wet-granulation →
compression tablet process where the CQAs are *genuine functions* of the CPPs
(dissolution truly depends on particle size, hardness, and humidity), plus
realistic noise, seasonality, and planted special-cause events.

This is a feature, not a compromise: **because the ground truth is known, the
detection system can be validated** — something rarely possible with real data.

## Architecture

```
generate_batch_data.py   Synthetic GMP batch data w/ planted events (ground truth)
        │
        ▼
spc_engine.py            I-MR charts · robust Phase-I baseline · Western Electric
        │                rules 1–4 · trend rule · Cpk/Ppk capability
        ├──▶ validate_spc.py    Proves 6/6 event detection vs ground truth
        ▼
ml_drivers.py            XGBoost CQA model · SHAP driver ranking + direction
        │
        ▼
app.py                   Streamlit dashboard (Plotly): control charts,
                         capability, drivers, batch explorer
```

**Methodology notes**

- Control limits are established from an 80-batch *qualification baseline*
  with iterative outlier trimming, then held fixed for production monitoring —
  computing limits from data that already contains special-cause variation
  inflates them and masks the very events you need to catch (Phase I / Phase II
  SPC discipline, per Montgomery).
- σ within is estimated as MR̄/d₂ (I-MR, n=2); capability reports both
  Cpk (within) and Ppk (overall) because the *gap* between them is diagnostic.
- The ML feature set includes only CPPs, raw-material and environmental
  inputs — no other CQAs — so importances map to actionable process knobs.
- Framing follows ICH Q8 (QbD), Q9 (quality risk management), and Q10
  (pharmaceutical quality system): identify CQAs, link them to CPPs, monitor,
  and improve.

## Run it

```bash
pip install -r requirements.txt
python generate_batch_data.py   # regenerate data (seeded, reproducible)
python validate_spc.py          # verify 6/6 event detection
python ml_drivers.py            # train model, produce SHAP report + figures
streamlit run app.py            # launch dashboard
```

## Repository contents

| File | Purpose |
|---|---|
| `generate_batch_data.py` | Seeded synthetic data generator (220 batches, 8 CPPs, 5 CQAs, 6 planted events) |
| `spc_engine.py` | Reusable SPC library (charts, rules, capability) |
| `validate_spc.py` | Ground-truth validation harness |
| `ml_drivers.py` | XGBoost + SHAP process-driver analysis |
| `app.py` | Streamlit dashboard |
| `batch_data.csv`, `ground_truth.csv` | Pre-generated data + event log |

## About

Built by a solid-state pharmaceutical scientist (13 y pharma R&D — crystallization,
polymorph science, co-crystal screening) moving deeper into CMC data science.
This project reflects how process monitoring is actually practiced in
development and manufacturing — the statistics, the vocabulary, and the
regulatory framing — implemented end-to-end in Python.

*All data is synthetic. No proprietary or GMP data was used.*
