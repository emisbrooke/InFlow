import argparse
import numpy as np
import torch
import get_st_funcs_generalized as gs
from pathlib import Path
import matplotlib.pyplot as plt
from importlib import reload
reload(gs)

# This file is intended to be adpoted into a shell script so you can add parameters to it. It has a single function

def calc_all_structs(t, theta_tf3, theta_tf24, theta_tg3, theta_tg24, factor=0.01, names_tf=None, out_dir=Path(".")):
    '''
    This is a main function that calculates and saves a bunch of structure attributes and makes a preliminary plot for some. You can comment out what you 
    do or do not want to include in calculations. If you want to save to a different file path, you need to change in this file.
    * Note that all loops and feedback counts only make sense on TF-TF because TF-TG doesn't include feedback *

    Inputs:
        t: the name of the tissue (used for loading)
        theta_tf3: the theta matrix for TF-TF GRN in young age (3m)
        theta_tf24: the theta matrix for TF-TF GRN in old age (24m)
        theta_tg3: the theta matrix for TF-TG GRN in young age (3m)
        theta_tg24: the theta matrix for TF-TG GRN in old age (24m)
        factor: The threshold under which connections in theta are crossed out, default=0.01
    outputs:
        None
    '''
  
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if names_tf is None:
        names_tf = np.array([f"TF_{i}" for i in range(theta_tf3.shape[0])], dtype=object)
    theta_tf3 = theta_tf3.copy()
    theta_tf24 = theta_tf24.copy()
    theta_tg3 = theta_tg3.copy()
    theta_tg24 = theta_tg24.copy()

    nTF, _ = theta_tf3.shape
    
    #### Cut off theta to take away weak connections
    theta_tf3[np.abs(theta_tf3) < factor] = 0; theta_tf24[np.abs(theta_tf24) < factor] = 0
    theta_tg3[np.abs(theta_tg3) < factor] = 0; theta_tg24[np.abs(theta_tg24) < factor] = 0
    
    # Calculate initial sparsities
    sp3_tf = np.where(theta_tf3 != 0)[0].shape[0]/theta_tf3.flatten().shape[0]
    sp24_tf = np.where(theta_tf24 != 0)[0].shape[0]/theta_tf24.flatten().shape[0]
    sp3_tg = np.where(theta_tg3 != 0)[0].shape[0]/theta_tg3.flatten().shape[0]
    sp24_tg = np.where(theta_tg24 != 0)[0].shape[0]/theta_tg24.flatten().shape[0]
    
    if sp3_tf < sp24_tf:
        print('Lowest sparsity for tf is 3m....')
        theta_tf_sparse = theta_tf3; theta_tf_dense = theta_tf24
        age_tf_sparse = 3; age_tf_dense = 24
        sp_sparse_tf = sp3_tf
    else:
        print('Lowest sparsity for tf is 24m....')
        theta_tf_sparse = theta_tf24; theta_tf_dense = theta_tf3   
        age_tf_sparse = 24; age_tf_dense = 3
        sp_sparse_tf = sp24_tf
    
    if sp3_tg < sp24_tg:
        print('and Lowest sparsity for tg is 3m....')
        theta_tg_sparse = theta_tg3; theta_tg_dense = theta_tg24
        age_tg_sparse = 3; age_tg_dense = 24 
        sp_sparse_tg = sp3_tg
    else:
        print('and Lowest sparsity for tg is 24m....')
        theta_tg_sparse = theta_tg24; theta_tg_dense = theta_tg3 
        age_tg_sparse = 24; age_tg_dense = 3 
        sp_sparse_tg = sp24_tg
         
    print(f'current sparsity for 3m is {sp3_tf} for TF and {sp3_tg} for TG')
    print(f'current sparsity for 24m is {sp24_tf} for TF and {sp24_tg} for TG')
    
    # Correct for sparsity 
    theta_tf_sparsified, out_deg_tf_sparsified, out_deg_tf_sparsified_eff, n0_diff_tf, sp_new_tf = gs.correct_sparsity(theta_tf_dense, theta_tf_sparse, nTF, niters=50)
    theta_tg_sparsified, out_deg_tg_sparsified, out_deg_tg_sparsified_eff, n0_diff_tg, sp_new_tg = gs.correct_sparsity(theta_tg_dense, theta_tg_sparse, nTF, niters=50)
    
    # save the matrix of out degrees (niters x nTG/TF)
    np.save(out_dir / f'out_deg_{t}_{age_tf_dense}m_tf_corrected_sp', out_deg_tf_sparsified)
    np.save(out_dir / f'out_deg_{t}_{age_tg_dense}m_tg_corrected_sp', out_deg_tg_sparsified)
    np.save(out_dir / f'out_deg_{t}_{age_tf_dense}m_tf_corrected_sp_sp_age_{age_tf_sparse}m', out_deg_tf_sparsified)
    np.save(out_dir / f'out_deg_{t}_{age_tg_dense}m_tg_corrected_sp_sp_age_{age_tg_sparse}m', out_deg_tg_sparsified)
    
    theta_tf_sparsified_bin = theta_tf_sparsified.copy(); theta_tf_sparsified_bin[np.abs(theta_tf_sparsified_bin) > 0] = 1
    theta_tg_sparsified_bin = theta_tg_sparsified.copy(); theta_tg_sparsified_bin[np.abs(theta_tg_sparsified_bin) > 0] = 1

    # Binarize the one that is already sparse
    theta_tf_sparse_bin = theta_tf_sparse.copy(); theta_tf_sparse_bin[np.abs(theta_tf_sparse_bin) > 0] = 1
    theta_tg_sparse_bin = theta_tg_sparse.copy(); theta_tg_sparse_bin[np.abs(theta_tg_sparse_bin) > 0] = 1
    
    print(f'remaining sparsity for sparse TF is {sp_sparse_tf} and {sp_new_tf}')
    #print(f'remaining sparsity for sparse TG is {sp_sparse_tg} and {sp_new_tg}')


    ################### IFL AND IFFL ######################

    # Calc and save IFL
    motif_df, counts_df, graphs_by_age = gs.analyze_IFL(theta_tf_sparsified, theta_tf_sparse, nTF, niters=20, names_tf = names_tf, age_sparse=age_tf_sparse, age_dense=age_tf_dense)
    motif_df.to_csv(out_dir / f'IFL_df_tf_{t}.csv', index=False)
    counts_df.to_csv(out_dir / f'IFL_counts_df_tf_{t}.csv', index=False)

    # Calc and save IFFL
    motif_df, counts_df, graphs_by_age = gs.analyze_IFFL(theta_tf_sparsified, theta_tf_sparse, nTF, niters=30, names_tf = names_tf, age_sparse=age_tf_sparse, age_dense=age_tf_dense)
    motif_df.to_csv(out_dir / f'IFFL_df_tf_{t}.csv', index=False)
    counts_df.to_csv(out_dir / f'IFFL_counts_df_tf_{t}.csv', index=False)

    ############## Look at 3 node feedback stuff ##################
    motif_df, counts_df, graphs_by_age = gs.analyze_3_node_feedback(theta_tf_sparsified, theta_tf_sparse, nTF, niters=50, names_tf = names_tf, age_sparse=age_tf_sparse, age_dense=age_tf_dense)
    motif_df.to_csv(out_dir / f'3_node_feedback_df_tf_{t}.csv', index=False)
    counts_df.to_csv(out_dir / f'3_node_feedback_counts_df_tf_{t}.csv', index=False)

    ############## Look at 2 node feedback stuff ##################
    motif_df, counts_df, graphs_by_age = gs.analyze_struct_2nodes(theta_tf_sparsified, theta_tf_sparse, nTF, niters=50, names_tf = names_tf, age_sparse=age_tf_sparse, age_dense=age_tf_dense)
    motif_df.to_csv(out_dir / f'2_node_feedback_df_{t}.csv', index=False)
    counts_df.to_csv(out_dir / f'2_node_feedback_counts_df_{t}.csv', index=False)
    
    ############ Now plot out degrees ###########

    null_out_deg_tf = gs.get_null(theta_tf3, theta_tf24, nTF, niters=100)
    null_out_deg_tg = gs.get_null(theta_tg3, theta_tg24, nTF, niters=100)
    np.save(out_dir / f'out_deg_{t}_null_tf', null_out_deg_tf)
    np.save(out_dir / f'out_deg_{t}_null_tg', null_out_deg_tg)
    
    out_deg_tf_sparse = theta_tf_sparse_bin.sum(axis=1); out_deg_tg_sparse = theta_tg_sparse_bin.sum(axis=1)
    out_deg_tf_sparse_eff = np.abs(theta_tf_sparse).sum(axis=1); out_deg_tg_sparse_eff = np.abs(theta_tg_sparse).sum(axis=1)
    np.save(out_dir / f'out_deg_{t}_{age_tf_sparse}m_tf_corrected_sp_dense_age_{age_tf_dense}m', out_deg_tf_sparse)
    np.save(out_dir / f'out_deg_{t}_{age_tg_sparse}m_tg_corrected_sp_dense_age_{age_tg_dense}m', out_deg_tg_sparse)

    # Plot real out degrees
    gs.plot_out_deg(out_deg_tf_sparsified, out_deg_tf_sparse, null_out_deg_tf, t, 'tf', dense_age=age_tf_dense, sparse_age=age_tf_sparse)
    gs.plot_out_deg(out_deg_tg_sparsified, out_deg_tg_sparse, null_out_deg_tg, t, 'tg', dense_age=age_tg_dense, sparse_age=age_tg_sparse)
    
    # Plot effective out degrees
    gs.plot_out_deg_eff(out_deg_tf_sparsified_eff, out_deg_tf_sparse_eff, t, 'tf', dense_age=age_tf_dense, sparse_age=age_tf_sparse)
    gs.plot_out_deg_eff(out_deg_tg_sparsified_eff, out_deg_tg_sparse_eff, t, 'tg', dense_age=age_tg_dense, sparse_age=age_tg_sparse)
    gs.plot_2nodes(motif_df, counts_df, graphs_by_age, t)


def parse_age_value_map(raw: str | None) -> dict[int, float]:
    if raw is None:
        return {}
    out: dict[int, float] = {}
    for part in str(raw).split(","):
        p = part.strip()
        if not p:
            continue
        if ":" not in p:
            raise ValueError(f"Invalid age map entry '{p}'. Expected format age:value.")
        age_s, val_s = p.split(":", 1)
        out[int(age_s.strip())] = float(val_s.strip())
    return out


def resolve_value_by_age(ages: list[int], default: float | None, overrides: dict[int, float], name: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for age in ages:
        if age in overrides:
            out[age] = float(overrides[age])
        elif default is not None:
            out[age] = float(default)
        else:
            raise ValueError(f"No {name} specified for age {age}. Provide --{name} or --{name}-by-age.")
    return out


def data_path(data_dir: Path, tissue: str, age: int, gene_type: str) -> Path:
    preferred = data_dir / f"{tissue}_{gene_type.lower()}_data_binary_{age}m_droplet_union.npy"
    if preferred.exists():
        return preferred
    legacy = data_dir / f"data_bin_filt_{tissue}_{age}m_{gene_type.upper()}.npy"
    return legacy


def lambda_tag(lambda_value: float) -> str:
    return f"{lambda_value:g}".replace("-", "m").replace(".", "p")


def model_path(models_dir: Path, tissue: str, age: int, gene_type: str, lam: float) -> Path:
    return models_dir / f"model_{tissue}_{age}m_{gene_type.upper()}_lam{lambda_tag(lam)}.pt"


def load_theta(models_dir: Path, tissue: str, age: int, gene_type: str, lam: float, threshold: float) -> np.ndarray:
    p = model_path(models_dir, tissue, age, gene_type, lam)
    if not p.exists():
        raise FileNotFoundError(f"Missing model file: {p}")
    payload = torch.load(p, map_location="cpu")
    theta = payload["theta"]
    if isinstance(theta, torch.Tensor):
        theta = theta.detach().cpu().numpy()
    theta = theta.copy()
    if threshold > 0:
        theta[np.abs(theta) < threshold] = 0
    return theta


def load_names_tf(data_dir: Path, tissue: str, age: int, n_tf: int) -> np.ndarray:
    candidates = [
        data_dir / f"final_tf_names_{tissue}.npy",
        Path("data") / f"final_tf_names_{tissue}.npy",
    ]
    for p in candidates:
        if p.exists():
            names = np.load(p, allow_pickle=True)
            if len(names) == n_tf:
                return names
    return np.array([f"TF_{i}" for i in range(n_tf)], dtype=object)


def parse_args():
    p = argparse.ArgumentParser(description="Compute structure metrics/plots from trained TF/TG models.")
    p.add_argument("--tissue", required=True)
    p.add_argument("--ages", nargs=2, type=int, default=[3, 24], help="Two ages to compare, e.g. 3 24.")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--models-dir", type=Path, default=Path("outputs/models/final"))
    p.add_argument("--out-dir", type=Path, default=Path("."))
    p.add_argument("--lambda", dest="lam", type=float, default=None)
    p.add_argument("--lambda-by-age", default=None, help="Comma map, e.g. 3:0.7,24:0.4")
    p.add_argument("--theta-threshold", type=float, default=0.0)
    p.add_argument("--theta-threshold-by-age", default=None, help="Comma map, e.g. 3:0.02,24:0.05")
    p.add_argument("--dry-run-check", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    ages = sorted(set(args.ages))
    if len(ages) != 2:
        raise ValueError("--ages must contain exactly two distinct ages.")

    lam_by_age = resolve_value_by_age(ages, args.lam, parse_age_value_map(args.lambda_by_age), "lambda")
    th_by_age = resolve_value_by_age(ages, args.theta_threshold, parse_age_value_map(args.theta_threshold_by_age), "theta-threshold")

    age_a, age_b = ages
    theta_tf_a = load_theta(args.models_dir, args.tissue, age_a, "TF", lam_by_age[age_a], th_by_age[age_a])
    theta_tf_b = load_theta(args.models_dir, args.tissue, age_b, "TF", lam_by_age[age_b], th_by_age[age_b])
    theta_tg_a = load_theta(args.models_dir, args.tissue, age_a, "TG", lam_by_age[age_a], th_by_age[age_a])
    theta_tg_b = load_theta(args.models_dir, args.tissue, age_b, "TG", lam_by_age[age_b], th_by_age[age_b])

    tf_data_p = data_path(args.data_dir, args.tissue, age_a, "TF")
    tf_data = np.load(tf_data_p)
    names_tf = load_names_tf(args.data_dir, args.tissue, age_a, theta_tf_a.shape[0])

    if args.dry_run_check:
        print("=== DRY RUN CHECK ===")
        print(f"tissue={args.tissue} ages={ages}")
        print(f"lambda_by_age={lam_by_age}")
        print(f"theta_threshold_by_age={th_by_age}")
        print(f"tf_data: {tf_data_p} shape={tf_data.shape}")
        print(f"theta_tf[{age_a}] shape={theta_tf_a.shape}")
        print(f"theta_tf[{age_b}] shape={theta_tf_b.shape}")
        print(f"theta_tg[{age_a}] shape={theta_tg_a.shape}")
        print(f"theta_tg[{age_b}] shape={theta_tg_b.shape}")
        return

    calc_all_structs(
        t=args.tissue,
        theta_tf3=theta_tf_a,
        theta_tf24=theta_tf_b,
        theta_tg3=theta_tg_a,
        theta_tg24=theta_tg_b,
        factor=0.0,
        names_tf=names_tf,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
