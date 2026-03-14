import json
import argparse
import os


def load_results(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare(files):
    all_results = {}
    labels = []
    for fp in files:
        label = os.path.splitext(os.path.basename(fp))[0]
        labels.append(label)
        all_results[label] = load_results(fp)

    # Collect all problem names across files
    all_problems = sorted(
        set(p for r in all_results.values() for p in r)
    )

    # Header
    metric_cols = []
    for label in labels:
        metric_cols.extend([f"BLEU ({label})", f"chrF ({label})"])

    header = f"{'Problem':<25} {'Type':<12} {'Diff':<5} " + " ".join(f"{c:>22}" for c in metric_cols)
    print(header)
    print("-" * len(header))

    # Per-problem rows
    totals = {label: {"bleu_sum": 0, "chrf_sum": 0, "count": 0} for label in labels}

    for problem in all_problems:
        # Get type/difficulty from whichever file has the problem
        ptype, diff = "", ""
        for label in labels:
            if problem in all_results[label]:
                info = all_results[label][problem]
                ptype = ", ".join(info["type"]) if isinstance(info["type"], list) else info["type"]
                diff = str(info["difficulty"])
                break

        row = f"{problem:<25} {ptype:<12} {diff:<5} "
        for label in labels:
            if problem in all_results[label]:
                m = all_results[label][problem]["metrics"]
                bleu, chrf = m["BLEU"], m["chrF"]
                totals[label]["bleu_sum"] += bleu
                totals[label]["chrf_sum"] += chrf
                totals[label]["count"] += 1
                row += f"{bleu:>22.3f} {chrf:>22.3f} "
            else:
                row += f"{'N/A':>22} {'N/A':>22} "
        print(row)

    # Averages
    print("-" * len(header))
    avg_row = f"{'AVERAGE':<25} {'':<12} {'':<5} "
    for label in labels:
        t = totals[label]
        if t["count"] > 0:
            avg_row += f"{t['bleu_sum'] / t['count']:>22.3f} {t['chrf_sum'] / t['count']:>22.3f} "
        else:
            avg_row += f"{'N/A':>22} {'N/A':>22} "
    print(avg_row)

    # Delta table (if exactly 2 files)
    if len(labels) == 2:
        print(f"\n{'='*60}")
        print(f"DELTA: {labels[1]} vs {labels[0]} (positive = improvement)")
        print(f"{'='*60}")
        print(f"{'Problem':<25} {'dBLEU':>10} {'dchrF':>10}")
        print("-" * 47)

        delta_bleu_sum, delta_chrf_sum, delta_count = 0, 0, 0
        for problem in all_problems:
            if problem in all_results[labels[0]] and problem in all_results[labels[1]]:
                m0 = all_results[labels[0]][problem]["metrics"]
                m1 = all_results[labels[1]][problem]["metrics"]
                db = m1["BLEU"] - m0["BLEU"]
                dc = m1["chrF"] - m0["chrF"]
                delta_bleu_sum += db
                delta_chrf_sum += dc
                delta_count += 1
                print(f"{problem:<25} {db:>+10.3f} {dc:>+10.3f}")

        print("-" * 47)
        if delta_count > 0:
            print(f"{'AVG DELTA':<25} {delta_bleu_sum / delta_count:>+10.3f} {delta_chrf_sum / delta_count:>+10.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two or more result JSON files")
    parser.add_argument("files", nargs="+", help="Paths to result JSON files")
    args = parser.parse_args()

    if len(args.files) < 2:
        print("Please provide at least 2 result files to compare.")
        raise SystemExit(1)

    compare(args.files)
