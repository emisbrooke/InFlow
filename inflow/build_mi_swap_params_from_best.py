import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Build MI swap Slurm params from best lambda/threshold table.")
    p.add_argument("--best-csv", type=Path, default=Path("outputs/model_plots/summary/best_lambda_by_group.csv"))
    p.add_argument("--out-file", type=Path, default=Path("examples/mi_swap_params.txt"))
    p.add_argument("--age", type=int, default=3, help="Use rows for this age (typically 3).")
    p.add_argument("--gene-type", default="TF", help="Use rows for this gene type (typically TF).")
    p.add_argument("--ages-csv", default="3,24", help="Swap ages CSV for run_mi_swaps, e.g. 3,24 or 3,18,24")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--n-tf-samples", type=int, default=30000)
    p.add_argument("--int-burn", type=int, default=200000)
    p.add_argument("--int-save", type=int, default=500)
    p.add_argument("--mc-batch-size", type=int, default=500)
    p.add_argument("--tissues", nargs="*", default=None, help="Optional tissue allowlist.")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.best_csv.exists():
        raise FileNotFoundError(f"Missing best CSV: {args.best_csv}")

    df = pd.read_csv(args.best_csv)
    required = {"tissue", "age_m", "gene_type", "lambda", "theta_threshold"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {args.best_csv}: {sorted(missing)}")

    sub = df[(df["age_m"] == args.age) & (df["gene_type"].astype(str).str.upper() == args.gene_type.upper())].copy()
    if args.tissues:
        keep = {t.lower() for t in args.tissues}
        sub = sub[sub["tissue"].astype(str).str.lower().isin(keep)]

    if sub.empty:
        raise RuntimeError("No matching rows found in best CSV for requested filters.")

    sub = sub.sort_values("tissue")
    args.out_file.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# tissue lambda theta_threshold iters n_tf_samples int_burn int_save mc_batch_size ages_csv",
        "# Auto-generated from best_lambda_by_group.csv",
        "",
    ]
    for _, r in sub.iterrows():
        lines.append(
            f"{r['tissue']} {float(r['lambda']):g} {float(r['theta_threshold']):g} "
            f"{args.iters} {args.n_tf_samples} {args.int_burn} {args.int_save} {args.mc_batch_size} {args.ages_csv}"
        )

    args.out_file.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(sub)} rows to {args.out_file}")


if __name__ == "__main__":
    main()
