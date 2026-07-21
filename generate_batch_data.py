"""
Synthetic GMP Batch Manufacturing Data Generator
=================================================

Simulates a solid-oral-dose (tablet) manufacturing process across ~2 years of
production. Designed as the data foundation for an SPC + ML portfolio project.

WHY SYNTHETIC: Real pharma batch data is proprietary and GMP-protected. This
generator produces statistically realistic data whose *ground truth* is known,
so the downstream SPC engine and ML models can be validated against events we
deliberately planted.

PROCESS MODEL (simplified wet-granulation -> compression -> coating line):
  CPPs (Critical Process Parameters, the inputs we control/measure):
    - granulation_water_pct      : binder solution added (%)
    - massing_time_min           : wet massing time (min)
    - inlet_air_temp_c           : fluid-bed dryer inlet temp (C)
    - blend_time_min             : final blend time (min)
    - compression_force_kn       : main compression force (kN)
    - press_speed_rpm            : tablet press turret speed (rpm)
    - api_d90_um                 : incoming API particle size, d90 (um)  [raw material]
    - room_humidity_pct          : ambient relative humidity (%)         [environment]

  CQAs (Critical Quality Attributes, the outputs / quality we care about):
    - tablet_weight_mg
    - hardness_kp                : breaking force (kilopond)
    - dissolution_30min_pct      : % API released at 30 min (key CQA)
    - assay_pct                  : potency, % of label claim
    - content_uniformity_rsd     : content uniformity, %RSD

EMBEDDED GROUND-TRUTH EVENTS (what your SPC engine should catch):
  1. GRADUAL DRIFT  : compression tooling wear -> hardness slowly climbs
                      from ~batch 90 onward (trend rule).
  2. STEP SHIFT     : new API raw-material lot at ~batch 140 with coarser d90
                      -> dissolution shifts down (Western Electric shift rules).
  3. OOT/OOS SPIKES : a handful of humidity excursions cause isolated
                      dissolution failures (out-of-limit points).
  4. SEASONALITY    : room humidity has an annual cycle that subtly nudges
                      dissolution (a real-world nuisance signal to handle).

Run:
    python generate_batch_data.py
Produces:
    batch_data.csv        (one row per batch)
    ground_truth.csv      (the planted events, for validating your SPC engine)
"""

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SEED = 42
N_BATCHES = 220
START_DATE = "2024-01-08"          # first Monday-ish; batches ~ every 3 days
PRODUCT = "Acetriptan 50 mg IR Tablet"   # fictional product
LINE = "Line-02"

# Event timing (batch index, 0-based)
DRIFT_START = 90        # tooling-wear drift begins
LOT_CHANGE = 140        # coarser API lot introduced
HUMIDITY_EXCURSIONS = [58, 112, 176, 201]   # isolated OOS-risk batches

rng = np.random.default_rng(SEED)


# --------------------------------------------------------------------------- #
# Helper
# --------------------------------------------------------------------------- #
def clip(x, lo, hi):
    return np.clip(x, lo, hi)


def generate():
    n = N_BATCHES
    idx = np.arange(n)

    dates = pd.date_range(start=START_DATE, periods=n, freq="3D")
    # day-of-year in radians for seasonal terms
    doy = dates.dayofyear.to_numpy()
    season = np.sin(2 * np.pi * doy / 365.25)

    # ----- Raw material / environment ------------------------------------- #
    # API particle size: stable, then a coarser lot from LOT_CHANGE onward.
    api_d90 = rng.normal(45, 2.0, n)
    api_d90[LOT_CHANGE:] += 9.0          # coarser lot -> slower dissolution
    api_d90 = clip(api_d90, 35, 70)

    # Room humidity: seasonal cycle + noise, with a few excursions.
    room_humidity = 45 + 8 * season + rng.normal(0, 3, n)
    for b in HUMIDITY_EXCURSIONS:
        room_humidity[b] += rng.uniform(18, 24)   # spike
    room_humidity = clip(room_humidity, 25, 85)

    # ----- CPPs (controlled, mostly in-spec with common-cause noise) ------ #
    granulation_water = rng.normal(32.0, 0.8, n)
    massing_time = rng.normal(6.0, 0.4, n)
    inlet_air_temp = rng.normal(60.0, 1.5, n)
    blend_time = rng.normal(15.0, 0.6, n)
    press_speed = rng.normal(45.0, 2.5, n)

    # Compression force: stable, then slow upward drift (tooling wear).
    compression_force = rng.normal(12.0, 0.5, n)
    drift = np.where(idx >= DRIFT_START, (idx - DRIFT_START) * 0.020, 0.0)
    compression_force = compression_force + drift
    compression_force = clip(compression_force, 9, 18)

    # ----- CQAs (functions of CPPs + noise) ------------------------------- #
    # Tablet weight: mostly independent, tight control.
    tablet_weight = rng.normal(250.0, 2.2, n)

    # Hardness: rises with compression force. Drift in force -> hardness trend.
    hardness = (
        6.0
        + 0.85 * (compression_force - 12.0)
        + 0.03 * (press_speed - 45.0)
        + rng.normal(0, 0.35, n)
    )
    hardness = clip(hardness, 3, 14)

    # Dissolution (KEY CQA): hurt by coarse API (d90), high hardness,
    # and high room humidity (hygroscopic slowdown). Seasonal nudge included.
    dissolution = (
        92.0
        - 0.55 * (api_d90 - 45.0)          # coarser API -> lower dissolution
        - 1.10 * (hardness - 6.0)          # harder tablets -> slower release
        - 0.18 * (room_humidity - 45.0)    # humidity effect
        + rng.normal(0, 1.6, n)
    )
    # Force the humidity-excursion batches to look like genuine failures.
    for b in HUMIDITY_EXCURSIONS:
        dissolution[b] -= rng.uniform(4, 7)
    dissolution = clip(dissolution, 55, 100)

    # Assay / potency: tight, slight blend-time dependence.
    assay = 100.0 + 0.4 * (blend_time - 15.0) + rng.normal(0, 1.1, n)
    assay = clip(assay, 90, 110)

    # Content uniformity (%RSD): worse with short blend or coarse API.
    cu_rsd = (
        2.5
        - 0.15 * (blend_time - 15.0)
        + 0.05 * (api_d90 - 45.0)
        + rng.normal(0, 0.4, n)
    )
    cu_rsd = clip(cu_rsd, 0.8, 8.0)

    # ----- Metadata ------------------------------------------------------- #
    api_lot = np.where(idx < LOT_CHANGE, "API-L0231", "API-L0248")
    shift = rng.choice(["A", "B", "C"], size=n)
    batch_id = [f"B24-{1000 + i}" for i in range(n)]

    df = pd.DataFrame(
        {
            "batch_id": batch_id,
            "date": dates,
            "product": PRODUCT,
            "line": LINE,
            "api_lot": api_lot,
            "shift": shift,
            # CPPs
            "granulation_water_pct": granulation_water.round(2),
            "massing_time_min": massing_time.round(2),
            "inlet_air_temp_c": inlet_air_temp.round(2),
            "blend_time_min": blend_time.round(2),
            "compression_force_kn": compression_force.round(2),
            "press_speed_rpm": press_speed.round(1),
            "api_d90_um": api_d90.round(1),
            "room_humidity_pct": room_humidity.round(1),
            # CQAs
            "tablet_weight_mg": tablet_weight.round(2),
            "hardness_kp": hardness.round(2),
            "dissolution_30min_pct": dissolution.round(2),
            "assay_pct": assay.round(2),
            "content_uniformity_rsd": cu_rsd.round(2),
        }
    )

    # ----- Ground-truth log (for validating the SPC engine) --------------- #
    events = []
    events.append(
        {"event": "tooling_wear_drift", "type": "trend",
         "start_batch": df.loc[DRIFT_START, "batch_id"],
         "affects": "compression_force_kn -> hardness_kp",
         "note": "Gradual upward drift from tooling wear."}
    )
    events.append(
        {"event": "api_lot_change", "type": "step_shift",
         "start_batch": df.loc[LOT_CHANGE, "batch_id"],
         "affects": "api_d90_um -> dissolution_30min_pct",
         "note": "Coarser API lot API-L0248 lowers dissolution."}
    )
    for b in HUMIDITY_EXCURSIONS:
        events.append(
            {"event": "humidity_excursion", "type": "oot_spike",
             "start_batch": df.loc[b, "batch_id"],
             "affects": "room_humidity_pct -> dissolution_30min_pct",
             "note": "Isolated humidity excursion -> dissolution failure risk."}
        )
    gt = pd.DataFrame(events)

    return df, gt


if __name__ == "__main__":
    df, gt = generate()
    df.to_csv("batch_data.csv", index=False)
    gt.to_csv("ground_truth.csv", index=False)

    print(f"Wrote batch_data.csv  ({len(df)} batches, {df.shape[1]} columns)")
    print(f"Wrote ground_truth.csv ({len(gt)} planted events)\n")
    print("Quick sanity check on key CQA (dissolution_30min_pct):")
    print(df["dissolution_30min_pct"].describe().round(2).to_string())
    print("\nPlanted events:")
    print(gt[["event", "type", "start_batch"]].to_string(index=False))
