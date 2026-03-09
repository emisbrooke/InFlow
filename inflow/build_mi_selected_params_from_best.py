import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Build mi_selected_params.txt from best lambda table.")
    p.add_argument("--best-csv", type=Path, default=Path("outputs/model_plots/summary/best_lambda_by_group.csv"))
    p.add_argument("--out-file", type=Path, default=Path("examples/mi_selected_params.txt"))
    p.add_argument("--age", type=int, default=3, help="Selection age used to read best-csv.")
    p.add_argument("--source-ages-csv", default="3,24", help="Source TF ages to run, CSV format.")
    p.add_argument("--gene-type", default="TF")
    p.add_argument("--tissues", nargs="*", default=None)
    p.add_argument("--target-ages-csv", default="3,24")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--n-tf-samples", type=int, default=30000)
    p.add_argument("--int-burn", type=int, default=60000)
    p.add_argument("--int-save", type=int, default=500)
    p.add_argument("--mc-batch-size", type=int, default=1200)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    if not args.best_csv.exists():
        raise FileNotFoundError(f"Missing best CSV: {args.best_csv}")

    df = pd.read_csv(args.best_csv)
    required = {"tissue", "age_m", "gene_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {args.best_csv}: {sorted(missing)}")

    sub = df[
        (df["age_m"] == args.age)
        & (df["gene_type"].astype(str).str.upper() == args.gene_type.upper())
    ].copy()
    if args.tissues:
        keep = {t.lower() for t in args.tissues}
        sub = sub[sub["tissue"].astype(str).str.lower().isin(keep)]
    if sub.empty:
        raise RuntimeError("No matching rows found for requested filters.")

    sub = sub.drop_duplicates(subset=["tissue"]).sort_values("tissue")
    args.out_file.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# tissue select_age source_ages_csv gene_type target_ages_csv iters n_tf_samples int_burn int_save mc_batch_size [seed]",
        "# Auto-generated from best_lambda_by_group.csv",
        "",
    ]
    for _, r in sub.iterrows():
        lines.append(
            f"{r['tissue']} {args.age} {args.source_ages_csv} {args.gene_type.upper()} {args.target_ages_csv} "
            f"{args.iters} {args.n_tf_samples} {args.int_burn} {args.int_save} {args.mc_batch_size} {args.seed}"
        )

    args.out_file.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(sub)} rows to {args.out_file}")


if __name__ == "__main__":
    main()
