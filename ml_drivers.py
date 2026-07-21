"""
Phase 3 - ML Process-Understanding Layer
========================================
Predict the key CQA (dissolution) from the Critical Process Parameters, then
explain the model to identify WHICH parameters drive dissolution failure.

This is the Quality-by-Design (QbD) payoff: SPC tells you *when* the process
went wrong; this tells you *why*. If the model is honest, its top drivers
should line up with the three planted event mechanisms:
    api_d90_um          <- API lot change (step shift)
    compression_force   <- tooling-wear drift (via tablet hardness)
    room_humidity_pct   <- humidity excursions

Outputs:
    ml_report.txt              text summary of metrics + ranked drivers
    fig_feature_importance.png XGBoost gain importance
    fig_shap_summary.png       SHAP beeswarm (magnitude + direction)
    fig_pred_vs_actual.png     model fit on held-out batches
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import shap

RANDOM_STATE = 42
TARGET = "dissolution_30min_pct"

# Controllable / measurable process inputs (CPPs + raw-material + environment).
# Deliberately excludes other CQAs so importances map to actionable knobs.
FEATURES = [
    "granulation_water_pct",
    "massing_time_min",
    "inlet_air_temp_c",
    "blend_time_min",
    "compression_force_kn",
    "press_speed_rpm",
    "api_d90_um",
    "room_humidity_pct",
]

PRETTY = {
    "api_d90_um": "API particle size (d90)",
    "compression_force_kn": "Compression force",
    "room_humidity_pct": "Room humidity",
    "press_speed_rpm": "Press speed",
    "blend_time_min": "Blend time",
    "granulation_water_pct": "Granulation water",
    "massing_time_min": "Massing time",
    "inlet_air_temp_c": "Inlet air temp",
}


def main():
    df = pd.read_csv("batch_data.csv")
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = XGBRegressor(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        n_jobs=2,
    )
    model.fit(X_train, y_train)

    # ---- Performance ----------------------------------------------------- #
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    mae = mean_absolute_error(y_test, pred)
    cv = cross_val_score(
        model, X, y, cv=KFold(5, shuffle=True, random_state=RANDOM_STATE),
        scoring="r2",
    )

    # ---- Honest forecasting check: time-ordered holdout ------------------ #
    # The random split above is the right choice for DRIVER ATTRIBUTION: we want
    # the model to see both API lots so SHAP can attribute the d90 effect. But a
    # shuffled split leaks temporal structure, so it is NOT a forecasting score.
    # Training on the earlier batches and testing on the most recent ones asks
    # the harder question -- can we predict the *next* batches? -- and scores far
    # lower, because within a single production regime the dominant driver
    # (API d90) barely varies. Reporting both keeps the model honest: it is an
    # explanatory model for process understanding, not a forecaster.
    cut = int(len(df) * 0.8)
    model_time = XGBRegressor(
        n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.9,
        colsample_bytree=0.9, random_state=RANDOM_STATE, n_jobs=2,
    )
    model_time.fit(X.iloc[:cut], y.iloc[:cut])
    r2_time = r2_score(y.iloc[cut:], model_time.predict(X.iloc[cut:]))

    # ---- Gain importance ------------------------------------------------- #
    gain = pd.Series(model.feature_importances_, index=FEATURES).sort_values(
        ascending=False
    )

    # ---- SHAP (magnitude + direction) ------------------------------------ #
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    mean_abs = pd.Series(np.abs(shap_values).mean(axis=0),
                         index=FEATURES).sort_values(ascending=False)
    # Direction: sign of correlation between feature value and its SHAP value.
    direction = {}
    for j, f in enumerate(FEATURES):
        corr = np.corrcoef(X[f].to_numpy(), shap_values[:, j])[0, 1]
        direction[f] = "higher -> LOWER dissolution" if corr < 0 else \
                        "higher -> HIGHER dissolution"

    # ---- Report ---------------------------------------------------------- #
    lines = []
    lines.append("XGBoost dissolution model -- performance")
    lines.append(f"  Test R^2 : {r2:.3f}")
    lines.append(f"  Test RMSE: {rmse:.3f} %")
    lines.append(f"  Test MAE : {mae:.3f} %")
    lines.append(f"  5-fold CV R^2: {cv.mean():.3f} +/- {cv.std():.3f}")
    lines.append(f"  Time-ordered R^2 (train first 80%, predict last 20%): {r2_time:.3f}")
    lines.append("    ^ the forecasting view is much lower because within one API-lot")
    lines.append("      regime the top driver (d90) is ~constant. This is an EXPLANATORY")
    lines.append("      model for driver attribution, not a forecaster.")
    lines.append("")
    lines.append("Ranked process drivers (by mean |SHAP|):")
    for f in mean_abs.index:
        lines.append(f"  {mean_abs[f]:6.3f}  {PRETTY.get(f, f):26s} "
                     f"{direction[f]}")
    report = "\n".join(lines)
    print(report)
    with open("ml_report.txt", "w") as fh:
        fh.write(report + "\n")

    # ---- Plots ----------------------------------------------------------- #
    plt.figure(figsize=(7, 4.2))
    top = mean_abs.sort_values()
    plt.barh([PRETTY.get(f, f) for f in top.index], top.values, color="#2b6cb0")
    plt.xlabel("mean |SHAP| (impact on dissolution, %)")
    plt.title("Drivers of tablet dissolution (XGBoost + SHAP)")
    plt.tight_layout()
    plt.savefig("fig_feature_importance.png", dpi=150)
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X,
                      feature_names=[PRETTY.get(f, f) for f in FEATURES],
                      show=False)
    plt.tight_layout()
    plt.savefig("fig_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.scatter(y_test, pred, alpha=0.7, color="#2b6cb0", edgecolor="white")
    lo, hi = y_test.min(), y_test.max()
    plt.plot([lo, hi], [lo, hi], "k--", lw=1)
    plt.xlabel("Actual dissolution (%)")
    plt.ylabel("Predicted dissolution (%)")
    plt.title(f"Held-out fit  (R2 = {r2:.2f})")
    plt.tight_layout()
    plt.savefig("fig_pred_vs_actual.png", dpi=150)
    plt.close()

    print("\nSaved: ml_report.txt, fig_feature_importance.png, "
          "fig_shap_summary.png, fig_pred_vs_actual.png")


if __name__ == "__main__":
    main()
