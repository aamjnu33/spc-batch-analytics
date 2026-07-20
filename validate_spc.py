"""
Validate the SPC engine against known ground truth.
====================================================
Runs the SPC engine on the synthetic batch data and confirms that every
deliberately planted process event is detected. This is the "does it actually
work" proof that turns a dashboard into evidence of competence.

Run:  python validate_spc.py   (after generate_batch_data.py)
"""

import pandas as pd
from spc_engine import run_spc

# Spec limits (fictional but realistic for an IR tablet)
SPECS = {
    "dissolution_30min_pct": {"lsl": 80, "usl": None},   # Q at 30 min, NLT 80%
    "hardness_kp":           {"lsl": 4,  "usl": 10},
}


def flagged_indices(result):
    """Row positions (0-based) of flagged batches, with their rules."""
    f = result["flags"]
    return {i: r for i, r in zip(f.index[f["flagged"]], f.loc[f["flagged"], "rules"])}


def idx_of(df, batch_id):
    return int(df.index[df["batch_id"] == batch_id][0])


def print_result(result):
    r = result
    print(f"\n=== {r['column']} ===")
    print(f"  Baseline (first {r['baseline_n']} batches): "
          f"CL={r['center']:.2f}  UCL={r['ucl']:.2f}  LCL={r['lcl']:.2f}  "
          f"sigma={r['sigma']:.3f}")
    cq = r["capability_qualification"]
    ca = r["capability_full_production"]
    print(f"  Capability @ qualification : Cpk={cq['Cpk']}  Ppk={cq['Ppk']}")
    print(f"  Capability @ full production: Cpk={ca['Cpk']}  Ppk={ca['Ppk']}  "
          f"(<- degradation is the story)")
    n_flagged = int(r["flags"]["flagged"].sum())
    print(f"  Batches flagged: {n_flagged} / {len(r['flags'])}")


def main():
    df = pd.read_csv("batch_data.csv")
    gt = pd.read_csv("ground_truth.csv")

    results = {}
    for col, spec in SPECS.items():
        results[col] = run_spc(df, col, baseline_n=80,
                               lsl=spec["lsl"], usl=spec["usl"])
        print_result(results[col])

    diss = results["dissolution_30min_pct"]
    hard = results["hardness_kp"]
    diss_hits = flagged_indices(diss)
    hard_hits = flagged_indices(hard)

    print("\n" + "=" * 66)
    print("GROUND-TRUTH DETECTION CHECK")
    print("=" * 66)

    all_pass = True

    def check(name, detected, detail):
        nonlocal all_pass
        status = "PASS" if detected else "FAIL"
        if not detected:
            all_pass = False
        print(f"  [{status}] {name}")
        print(f"         {detail}")

    # 1. Tooling-wear drift -> hardness rises from batch B24-1090
    start = idx_of(df, "B24-1090")
    window = [i for i in hard_hits if start <= i <= start + 45]
    check("tooling_wear_drift (hardness, from B24-1090)",
          bool(window),
          f"first hardness flag at batch {df['batch_id'].iloc[window[0]]} "
          f"-> {hard_hits[window[0]]}" if window else "no hardness flag in window")

    # 2. API lot change -> dissolution steps down from B24-1140
    start = idx_of(df, "B24-1140")
    window = [i for i in diss_hits if start <= i <= start + 45]
    check("api_lot_change (dissolution, from B24-1140)",
          bool(window),
          f"first dissolution flag at batch {df['batch_id'].iloc[window[0]]} "
          f"-> {diss_hits[window[0]]}" if window else "no dissolution flag in window")

    # 3. Humidity excursions -> isolated dissolution failures
    for bid in ["B24-1058", "B24-1112", "B24-1176", "B24-1201"]:
        i = idx_of(df, bid)
        hit = any(j in diss_hits for j in (i - 1, i, i + 1))
        j = next((j for j in (i, i - 1, i + 1) if j in diss_hits), None)
        check(f"humidity_excursion at {bid} (dissolution)",
              hit,
              f"flagged {df['batch_id'].iloc[j]} -> {diss_hits[j]}"
              if j is not None else "not flagged")

    print("\n" + "=" * 66)
    print(f"RESULT: {'ALL 6 EVENTS DETECTED' if all_pass else 'SOME EVENTS MISSED'}")
    print("=" * 66)


if __name__ == "__main__":
    main()
