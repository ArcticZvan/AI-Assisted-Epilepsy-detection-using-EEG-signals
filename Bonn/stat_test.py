"""
Statistical significance testing across 10-Fold CV results.

Performs paired t-tests between all model pairs within each task to determine
whether performance differences are statistically significant (p < 0.05).
"""
import json
import os
import itertools
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

TASKS = ["binary", "three", "five"]
MODELS = {
    "hybrid": "{task}_hybrid",
    "cnn": "{task}_cnn",
    "bilstm": "{task}_bilstm",
    "svm": "{task}_svm",
}


def load_fold_accuracies(task: str) -> dict[str, list[float]]:
    results = {}
    for model_name, dir_pattern in MODELS.items():
        dir_name = dir_pattern.format(task=task)
        path = os.path.join(OUTPUT_DIR, dir_name, "results.json")
        if not os.path.exists(path):
            print(f"  [WARN] Missing: {path}")
            continue
        with open(path) as f:
            data = json.load(f)
        results[model_name] = data["fold_accuracies"]
    return results


def run_paired_ttests(task: str, fold_accs: dict[str, list[float]]) -> list[dict]:
    pairs = list(itertools.combinations(sorted(fold_accs.keys()), 2))
    rows = []
    for m1, m2 in pairs:
        a1, a2 = fold_accs[m1], fold_accs[m2]
        t_stat, p_val = stats.ttest_rel(a1, a2)
        rows.append({
            "task": task,
            "model_a": m1,
            "model_b": m2,
            "mean_a": f"{sum(a1)/len(a1)*100:.2f}%",
            "mean_b": f"{sum(a2)/len(a2)*100:.2f}%",
            "t_statistic": round(t_stat, 4),
            "p_value": round(p_val, 6),
            "significant": "yes" if p_val < 0.05 else "no",
        })
    return rows


def print_table(all_rows: list[dict]) -> None:
    header = f"{'Task':<8} {'Model A':<10} {'Model B':<10} {'Mean A':>8} {'Mean B':>8} {'t':>8} {'p':>10} {'Sig?':>5}"
    print(header)
    print("-" * len(header))
    current_task = None
    for r in all_rows:
        if r["task"] != current_task:
            if current_task is not None:
                print()
            current_task = r["task"]
        sig = " *" if r["significant"] == "yes" else ""
        print(
            f"{r['task']:<8} {r['model_a']:<10} {r['model_b']:<10} "
            f"{r['mean_a']:>8} {r['mean_b']:>8} "
            f"{r['t_statistic']:>8.4f} {r['p_value']:>10.6f}{sig}"
        )


def main():
    all_rows = []
    for task in TASKS:
        print(f"\n[{task.upper()} classification]")
        fold_accs = load_fold_accuracies(task)
        if len(fold_accs) < 2:
            print("  Not enough models to compare.")
            continue
        rows = run_paired_ttests(task, fold_accs)
        all_rows.extend(rows)

    print("\n" + "=" * 80)
    print("PAIRED T-TEST RESULTS (p < 0.05 marked with *)")
    print("=" * 80 + "\n")
    print_table(all_rows)

    out_path = os.path.join(OUTPUT_DIR, "statistical_tests.json")
    with open(out_path, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
