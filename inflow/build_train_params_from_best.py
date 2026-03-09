import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Build Slurm train params from best lambda table.")
    p.add_argument("--best-csv", type=Path, default=Path("outputs/model_plots/summary/best_lambda_by_group.csv"))
    p.add_argument("--out-file", type=Path, default=Path("examples/train_params_selected.txt"))
    p.add_argument("--select-age", type=int, default=3)
    p.add_argument("--select-gene-type", default="TF")
    p.add_argument("--tissues", nargs="*", default=None)
    p.add_argument("--train-ages", nargs="+", type=int, default=[3, 24])
    p.add_argument("--train-gene-types", nargs="+", default=["TF", "TG"])
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--n-restarts", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    if not args.best_csv.exists():
        raise FileNotFoundError(f"Missing best CSV: {args.best_csv}")

    df = pd.read_csv(args.best_csv)
    required = {"tissue", "age_m", "gene_type", "lambda"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {args.best_csv}: {sorted(missing)}")

    sub = df[
        (df["age_m"] == args.select_age)
        & (df["gene_type"].astype(str).str.upper() == args.select_gene_type.upper())
    ].copy()
    if args.tissues:
        keep = {t.lower() for t in args.tissues}
        sub = sub[sub["tissue"].astype(str).str.lower().isin(keep)]
    if sub.empty:
        raise RuntimeError("No matching rows found in best CSV for requested filters.")

    sort_cols = [c for c in ["corr_pearson_upper", "mean_pearson"] if c in sub.columns]
    if sort_cols:
        sub = sub.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    sub = sub.drop_duplicates(subset=["tissue"], keep="first").sort_values("tissue")

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# tissue age gene_type steps n_restarts lambda1 [lambda2 ...]",
        "# Auto-generated from best_lambda_by_group.csv (lambda selected from 3m TF by default).",
        "",
    ]

    for _, r in sub.iterrows():
        tissue = str(r["tissue"])
        lam = float(r["lambda"])
        for age in args.train_ages:
            for gt in args.train_gene_types:
                lines.append(
                    f"{tissue} {int(age)} {str(gt).upper()} {args.steps} {args.n_restarts} {lam:g}"
                )

    args.out_file.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines) - 3} rows to {args.out_file}")


if __name__ == "__main__":
    main()
