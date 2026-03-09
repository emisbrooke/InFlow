import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Train selected models using best lambda values (typically chosen from 3m TF)."
    )
    p.add_argument("--best-csv", type=Path, default=Path("outputs/model_plots/summary/best_lambda_by_group.csv"))
    p.add_argument("--select-age", type=int, default=3, help="Age used when selecting best lambda rows.")
    p.add_argument("--select-gene-type", default="TF", help="Gene type used when selecting best lambda rows.")
    p.add_argument("--tissues", nargs="*", default=None, help="Optional tissue allowlist.")
    p.add_argument("--train-ages", nargs="+", type=int, default=[3, 24], help="Ages to train for each selected tissue.")
    p.add_argument("--train-gene-types", nargs="+", default=["TF", "TG"], help="Gene types to train for each selected tissue.")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--n-restarts", type=int, default=10)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/models"))
    p.add_argument("--cpu-only", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print commands only.")
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

    # If duplicates exist per tissue, keep best by corr then mean if available.
    sort_cols = [c for c in ["corr_pearson_upper", "mean_pearson"] if c in sub.columns]
    if sort_cols:
        sub = sub.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    sub = sub.drop_duplicates(subset=["tissue"], keep="first").sort_values("tissue")

    train_script = Path(__file__).parent / "train_sweep.py"

    for _, r in sub.iterrows():
        tissue = str(r["tissue"])
        lam = float(r["lambda"])
        cmd = [
            sys.executable,
            str(train_script),
            "--data-dir",
            str(args.data_dir),
            "--out-dir",
            str(args.out_dir),
            "--tissues",
            tissue,
            "--ages",
            *[str(a) for a in args.train_ages],
            "--gene-types",
            *[gt.upper() for gt in args.train_gene_types],
            "--lambdas",
            f"{lam:g}",
            "--steps",
            str(args.steps),
            "--n-restarts",
            str(args.n_restarts),
        ]
        if args.cpu_only:
            cmd.append("--cpu-only")
        if args.overwrite:
            cmd.append("--overwrite")
        if not args.verbose:
            cmd.append("--quiet")

        print("RUN:", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
