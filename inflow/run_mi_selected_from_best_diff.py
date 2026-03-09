import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Run MI swaps using selected lambda/theta-threshold from best table."
    )
    p.add_argument("--best-csv", type=Path, default=Path("outputs/model_plots/summary/best_lambda_by_group.csv"))
    p.add_argument("--age", type=int, default=3, help="Select rows for this age in best-csv (selection age).")
    p.add_argument("--source-ages", nargs="+", type=int, default=None, help="Source TF ages to run. Default: selection age only.")
    p.add_argument("--gene-type", default="TF", help="Select rows for this gene type.")
    p.add_argument("--tissues", nargs="*", default=None, help="Optional tissue allowlist.")
    p.add_argument("--target-ages", nargs="+", type=int, default=[3, 24], help="Target TG ages to run against source age.")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--n-tf-samples", type=int, default=30000)
    p.add_argument("--int-burn", type=int, default=200000)
    p.add_argument("--int-save", type=int, default=500)
    p.add_argument("--mc-batch-size", type=int, default=500)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--models-dir", type=Path, default=Path("outputs/models"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/mi_swaps"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print commands only.")
    p.add_argument(
        "--use-age-specific-hparams",
        action="store_true",
        help="Use age-specific lambda/theta-threshold from best-csv for each source/target age.",
    )
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

    selection = df[
        (df["age_m"] == args.age)
        & (df["gene_type"].astype(str).str.upper() == args.gene_type.upper())
    ].copy()
    if args.tissues:
        keep = {t.lower() for t in args.tissues}
        selection = selection[selection["tissue"].astype(str).str.lower().isin(keep)]
    if selection.empty:
        raise RuntimeError("No matching rows found in best CSV for requested filters.")

    # If duplicates exist per tissue, keep best by corr then mean.
    sort_cols = ["corr_pearson_upper", "mean_pearson"]
    sort_cols = [c for c in sort_cols if c in selection.columns]
    run_script = Path(__file__).parent / "run_mi_swaps.py"
    source_ages = sorted(set(args.source_ages)) if args.source_ages is not None else [args.age]
    ages_union = sorted(set(source_ages + args.target_ages))
    tissues = sorted(selection["tissue"].astype(str).unique())

    for tissue in tissues:
        if args.use_age_specific_hparams:
            rows = df[
                (df["tissue"].astype(str) == tissue)
                & (df["age_m"].isin(ages_union))
                & (df["gene_type"].astype(str).str.upper() == args.gene_type.upper())
            ].copy()
            if rows.empty:
                raise RuntimeError(f"No rows found for tissue={tissue} and ages={ages_union}")
            if sort_cols:
                rows = rows.sort_values(sort_cols, ascending=[False] * len(sort_cols))
            rows = rows.drop_duplicates(subset=["age_m"], keep="first")
            missing_ages = sorted(set(ages_union) - set(rows["age_m"].astype(int).tolist()))
            if missing_ages:
                raise RuntimeError(
                    f"Missing best-csv rows for tissue={tissue}, gene_type={args.gene_type}, ages={missing_ages}"
                )
            lam_map = {int(r["age_m"]): float(r["lambda"]) for _, r in rows.iterrows()}
            th_map = {int(r["age_m"]): float(r["theta_threshold"]) for _, r in rows.iterrows()}
            lam_map_s = ",".join(f"{age}:{lam_map[age]:g}" for age in sorted(lam_map))
            th_map_s = ",".join(f"{age}:{th_map[age]:g}" for age in sorted(th_map))
        else:
            rows = selection[selection["tissue"].astype(str) == tissue].copy()
            if rows.empty:
                raise RuntimeError(f"No selection row for tissue={tissue}")
            if sort_cols:
                rows = rows.sort_values(sort_cols, ascending=[False] * len(sort_cols))
            r = rows.iloc[0]
            lam = float(r["lambda"])
            th = float(r["theta_threshold"])

        cmd = [
            sys.executable,
            str(run_script),
            "--tissue",
            tissue,
            "--ages",
            *[str(x) for x in ages_union],
            "--source-ages",
            *[str(x) for x in source_ages],
            "--target-ages",
            *[str(x) for x in args.target_ages],
            "--iters",
            str(args.iters),
            "--n-tf-samples",
            str(args.n_tf_samples),
            "--int-burn",
            str(args.int_burn),
            "--int-save",
            str(args.int_save),
            "--mc-batch-size",
            str(args.mc_batch_size),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--data-dir",
            str(args.data_dir),
            "--models-dir",
            str(args.models_dir),
            "--out-dir",
            str(args.out_dir),
        ]
        if args.use_age_specific_hparams:
            cmd.extend(["--lambda-by-age", lam_map_s, "--theta-threshold-by-age", th_map_s])
        else:
            cmd.extend(["--lambda", f"{lam:g}", "--theta-threshold", f"{th:g}"])
        if args.overwrite:
            cmd.append("--overwrite")
        if args.verbose:
            cmd.append("--verbose")

        print("RUN:", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
