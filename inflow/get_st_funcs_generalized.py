
import argparse
import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy import sparse


def lambda_tag(lambda_value: float) -> str:
    return f"{lambda_value:g}".replace("-", "m").replace(".", "p")


def data_path(data_dir: Path, tissue: str, age: int, gene_type: str) -> Path:
    preferred = data_dir / f"{tissue}_{gene_type.lower()}_data_binary_{age}m_droplet_union.npy"
    if preferred.exists():
        return preferred
    legacy = data_dir / f"data_bin_filt_{tissue}_{age}m_{gene_type.upper()}.npy"
    return legacy


def model_path(models_dir: Path, tissue: str, age: int, gene_type: str, lam: float) -> Path:
    return models_dir / f"model_{tissue}_{age}m_{gene_type.upper()}_lam{lambda_tag(lam)}.pt"


def load_pt_param(tissue: str, models_dir: str | Path, age: int, lam: float, gene_type: str = "TG"):
    p = model_path(Path(models_dir), tissue, age, gene_type, lam)
    if not p.exists():
        raise FileNotFoundError(f"Missing model file: {p}")
    payload = torch.load(p, map_location="cpu")
    theta = payload["theta"]
    m = payload["m"]
    if isinstance(theta, torch.Tensor):
        theta = theta.detach().cpu().numpy()
    if isinstance(m, torch.Tensor):
        m = m.detach().cpu().numpy()
    return theta, m


def load_h5_param(t, directory_path, age, lam, n_iters = 10):
    '''
    This function takes a filepath, age, and lambda value and returns the model parameters corresponding
    to that model
    '''
    L_max = float('-inf') # Find the model run with highest likelihood
    best_n = 30 # store which that is 
    #print(f'{age}m, lam = {lam}')

    for i in range(n_iters):
        if not os.path.exists(f'{directory_path}/model_dict_tg_age_{age}m_iter{i}_lambda{lam}_no_ho.h5'):
            print(f'There is no thing for iter {i}')
            continue
        with h5py.File(f'{directory_path}/model_dict_tg_age_{age}m_iter{i}_lambda{lam}_no_ho.h5', 'r') as h5f:
            Larr = h5f['Larr'][:]
            if Larr[-1]>L_max:
                L_max = Larr[-1]
                best_n = i

    #print(f'best n is {best_n}')
    if os.path.exists(f'{directory_path}/theta_t{t}_lam{lam}_iter{best_n}_{age}m.npz'):
        theta = sparse.load_npz(f'{directory_path}/theta_t{t}_lam{lam}_iter{best_n}_{age}m.npz').toarray()
        with h5py.File(f'{directory_path}/model_dict_tg_age_{age}m_iter{best_n}_lambda{lam}_no_ho.h5', 'r') as h5f:
            m = h5f['m'][:]
            pi = h5f['pi'][:]
    
    else:
        print('loading...')
        with h5py.File(f'{directory_path}/model_dict_tg_age_{age}m_iter{best_n}_lambda{lam}_no_ho.h5', 'r') as h5f:
            pi = h5f['pi'][:]
            theta = h5f['theta'][:]
            m = h5f['m'][:]

    return pi, theta, m

def correct_sparsity(theta_dense, theta_sparse, nTF, niters=100):
    '''
    Code to sparsify one matrix to match the other
    Input:
        theta_dense: a matrix with a lower sparsity then
        theta_sparse: the matrix with higher sparsity
        nTF: # of TFs (or genes) in the matrix
        niters: # of iterations of sparsifying to avg to get out degree, default=100
    Output:
        theta_sp: the new sparsified theta 
        out_deg_sp: the out degrees of the niters iterations of sparsifying 
        out_deg_sp_eff: the effective out degree 
        n0_diff: number of connection lost 
        sp_new: sparsity after removing connections *should be the same as that of theta_sparse
    '''
    rng = np.random.default_rng()
    sp_dense = np.where(theta_dense != 0)[0].shape[0]/theta_dense.flatten().shape[0]
    sp_sparse = np.where(theta_sparse != 0)[0].shape[0]/theta_sparse.flatten().shape[0]

    n0_diff = np.where(theta_dense != 0)[0].shape[0] - theta_dense.flatten().shape[0] * sp_sparse
    print(n0_diff)
    n0_diff = int(n0_diff)
    connect_dense = np.where(theta_dense.flatten() != 0)[0]
    print(n0_diff)

    out_deg_sp = np.zeros((niters, nTF))
    out_deg_sp_eff = np.zeros((niters, nTF))

    for i in range(niters):

        perm = rng.permutation(connect_dense)
        set0 = perm[:n0_diff]
        indices0 = np.unravel_index(set0, theta_dense.shape)

        # set connections to 0
        theta_sp = theta_dense.copy()
        theta_sp[indices0] = 0

        # Do the same to binary
        theta_sp_bin = theta_sp.copy()
        theta_sp_bin[theta_sp_bin!=0] = 1

        # check new sparsities
        sp_new = np.where(theta_sp != 0)[0].shape[0]/theta_sp.flatten().shape[0]
        sp_old = np.where(theta_sparse != 0)[0].shape[0]/theta_sparse.flatten().shape[0]

        if not np.isclose(sp_new, sp_old, atol=1e-12):
            print(f'YIKES on TFs {sp_new}, {sp_old}')
            break

        out_deg_sp[i] = np.abs(theta_sp_bin).sum(axis=1)
        out_deg_sp_eff[i] = np.abs(theta_sp).sum(axis=1)

    return theta_sp, out_deg_sp, out_deg_sp_eff, n0_diff, sp_new


def sparsify_once_to_match(theta_dense, theta_sparse, rng=None):
    """
    Draw a single sparsified version of `theta_dense` whose nonzero fraction matches `theta_sparse`.
    """
    if rng is None:
        rng = np.random.default_rng()

    sp_sparse = np.count_nonzero(theta_sparse) / theta_sparse.size
    n_keep = int(round(theta_dense.size * sp_sparse))
    flat_dense = theta_dense.reshape(-1)
    nonzero_idx = np.flatnonzero(flat_dense)

    if n_keep >= nonzero_idx.size:
        return theta_dense.copy()

    n_drop = nonzero_idx.size - n_keep
    drop_idx = rng.choice(nonzero_idx, size=n_drop, replace=False)
    theta_sp = theta_dense.copy().reshape(-1)
    theta_sp[drop_idx] = 0
    return theta_sp.reshape(theta_dense.shape)

def keep_top_pct(theta, pct=0.1):
    """Keep only the top `pct` fraction of non-zero entries by |value|, zero the rest.

    Parameters
    ----------
    theta : array-like
        The matrix to filter.
    pct : float
        Fraction of non-zero entries to keep (e.g. 0.1 = top 10%).

    Returns
    -------
    theta_filtered : np.ndarray
        A copy with only the strongest entries retained.
    """
    theta = np.asarray(theta).copy()
    nonzero_vals = np.abs(theta[theta != 0])
    if nonzero_vals.size == 0:
        return theta
    cutoff = np.percentile(nonzero_vals, (1 - pct) * 100)
    theta[np.abs(theta) < cutoff] = 0
    return theta


def get_null(theta_dense, theta_sparse, nTF, niters=100):
    '''
    code to calculate null model for literal out degree. This is only based on the sparsity of the most
    sparse of the ages. For effective null model, you have to consider each age seperately.

    returns:
        out_deg_null: an array of out degrees for the niters of trying null models
    '''
    rng = np.random.default_rng()
    out_deg_null = np.zeros((niters*2, nTF))

    for i in range(0, niters*2, 2):
        null_dense = np.zeros(theta_dense.flatten().shape[0])
        n_connect = np.where(theta_dense != 0)[0].shape[0]
        null_dense[:n_connect] = 1

        null_sparse = np.zeros(theta_sparse.flatten().shape[0])
        n_connect = np.where(theta_sparse != 0)[0].shape[0]
        null_sparse[:n_connect] = 1

        null_dense =rng.permutation(null_dense).reshape(theta_dense.shape)
        null_sparse =rng.permutation(null_sparse).reshape(theta_sparse.shape)

        out_deg_null[i] = null_dense.sum(axis=1)
        out_deg_null[i+1] = null_sparse.sum(axis=1)

    return out_deg_null


#############################################################
#*******FUNCTIIONS FOR FINDING DIFFERENT MOTIFS*********
#############################################################

def _build_graph(theta, names_tf):
    """Build a NetworkX DiGraph from a theta matrix, casting to float64 for scipy.sparse compatibility."""
    theta = np.asarray(theta, dtype=np.float64)
    M = sparse.csr_matrix(theta)
    coo = M.tocoo()
    src = [names_tf[i] for i in coo.row]
    tgt = [names_tf[j] for j in coo.col]
    wts = coo.data
    df_edges = pd.DataFrame({'source': src, 'target': tgt, 'weight': wts})
    return nx.from_pandas_edgelist(df_edges, 'source', 'target', edge_attr='weight', create_using=nx.DiGraph())

def find_two_cycles(G):
    """
    Return each 2-cycle once, as (u,v) with u < v and edges u->v and v->u.
    """
    cycles = set()
    for A, B in G.edges():
        if B in G.predecessors(A):
            cycles.add((A, B))
    return list(cycles)


def find_two_cycles_no_parent(G):
    """
    Return each 2-cycle once, as (u,v) with u < v and edges u->v and v->u.
    """
    cycles = set()
    count = 0
    count1 = 0
    for A, B in G.edges(): # A->B 
        if G.has_edge(B, A): # and B->A
            count1+=1
            if (G.in_degree(A) == 1) and (G.in_degree(B) == 1): # B and A have only one regulator (each other)
                print(f'found 1: {A}, {B}')
                cycles.add((A, B))
            for C in G.nodes():
                if C not in (A, B):
                    if (C in G.predecessors(A)) or (C in G.predecessors(B)):
                        count+=1 # This just counts the number of feedback loops that do have a parent
                        break
    print(f'there are {count1} normal cycles and then {count} that don\'t count')
    return list(cycles)

    
def find_IFFL(G):
    """
    Return (A,B,C) triples where edges A→B, A→C, and C→B all exist (incoherent feedforward loops)
    """
    motifs = []
    for C, B in G.edges(): # If C->B
        if B not in G.predecessors(C): # But B does not -> C
            for A in G.predecessors(B): # And A->B
                if (G.has_edge(A, B)) and (B not in G.predecessors(A)): # A->B but not B->A
                    if (G.has_edge(A, C)) and (C not in G.predecessors(A)): # A->C but not C->A
                        motifs.append((A, B, C))
    return list({tuple(m) for m in motifs}) #motifs

def find_IFL(G):
    """
    Return (A,B,C) triples where edges A→B, B→C, and C→B all exist but not C->A or B->A (incoherent feedback loop)
    """
    motifs = []
    for B, C in G.edges():
        if G.has_edge(C, B): # if B->C, and C->B
            for A in G.predecessors(B): # And A->B   
                if A not in (B, C): # But A is not B, C
                    if (B not in G.predecessors(A)) and (C not in G.predecessors(A)) and (A not in G.predecessors(C)): # there is no B->A or C->A
                        motifs.append((A, B, C))
    return list({tuple(m) for m in motifs}) 




##############################################################################################################
##############################################################################################################
'''
GET MOTIFS AND GIVE A DB
'''
##############################################################################################################
##############################################################################################################

def analyze_3_node_feedback(
    theta_dense,
    theta_sparse,
    nG,
    niters,
    names_tf,
    age_sparse=24,
    age_dense=3,
    max_records_per_age_pattern=5,
):
    '''
    Inputs:
        theta_dense: the theta of the denser age
        theta_sparse: the theta for the less dense age
        nG: number of genes (TG or TF)
        names_tf: names of the TFs to make db of loops
        age_sparse: actual age of less dense age, default=24
        age_dense: actual age of more dense age, default=3
        niters: number of iterations to repeat sparsifying more dense matrix, default = 100

    Returns:
        motif_df: a df with the motifs and names of tfs/tgs in them
        counts_df: a df with counts for each type of loop
        graphs_by_age: the graph corresponding the theta for each age
    '''
    from collections import defaultdict
    rng = np.random.default_rng()
    counts_acc = defaultdict(float)
    all_records = []
    record_acc = defaultdict(int)

    G_sparse = _build_graph(theta_sparse, names_tf)

    for i in range(niters):
        print(f'******* On iter {i} ********')
        theta_sp = sparsify_once_to_match(theta_dense, theta_sparse, rng=rng)

        G_dense = _build_graph(theta_sp, names_tf)
        graphs_by_age = {age_dense: G_dense, age_sparse: G_sparse}

        for age, G in graphs_by_age.items():
            # This path currently reuses the IFL motif definition for the 3-node feedback summary.
            motifs = find_IFL(G)
            for A, B, C in motifs:
                sAB = '+' if G[A][B]['weight'] > 0 else '-'
                sBC = '+' if G[B][C]['weight'] > 0 else '-'
                sCB = '+' if G[C][B]['weight'] > 0 else '-'
                pattern = sAB + sBC + sCB
                counts_acc[(age, pattern)] += 1
                if record_acc[(age, pattern)] < max_records_per_age_pattern:
                    all_records.append({
                        'age':     age,
                        'A':       A,
                        'B':       B,
                        'C':       C,
                        'pattern': pattern
                    })
                    record_acc[(age, pattern)] += 1

    motif_df = pd.DataFrame(all_records)
    counts_rows = [{'age': k[0], 'pattern': k[1], 'count': v / niters} for k, v in counts_acc.items()]
    counts_df = pd.DataFrame(counts_rows) if counts_rows else pd.DataFrame(columns=['age', 'pattern', 'count'])

    return motif_df, counts_df, graphs_by_age

def analyze_IFFL(
    theta_dense,
    theta_sparse,
    nG,
    niters,
    names_tf,
    age_sparse=24,
    age_dense=3,
    return_iter_counts=False,
    max_records_per_age_pattern=5,
):

    '''
    Inputs:
        theta_dense: the theta of the denser age
        theta_sparse: the theta for the less dense age
        nG: number of genes (TG or TF)
        names_tf: names of the TFs to make db of loops
        age_sparse: actual age of less dense age, default=24
        age_dense: actual age of more dense age, default=3
        niters: number of iterations to repeat sparsifying more dense matrix, default=100
    Returns:
        motif_df: a df with the IFFL motifs and names of tfs/tgs in them
        counts_df: a df with counts for each type of IFFL loop
        graphs_by_age: the graph corresponding the theta for each age
    '''
    from collections import defaultdict
    rng = np.random.default_rng()
    counts_acc = defaultdict(float)
    all_records = []
    iter_count_records = []
    record_acc = defaultdict(int)

    G_sparse = _build_graph(theta_sparse, names_tf)

    for i in range(niters):
        print(f'******* On iter {i} ********')
        theta_sp = sparsify_once_to_match(theta_dense, theta_sparse, rng=rng)

        G_dense = _build_graph(theta_sp, names_tf)
        graphs_by_age = {age_dense: G_dense, age_sparse: G_sparse}

        iter_counts_acc = defaultdict(float)
        for age, G in graphs_by_age.items():
            motifs = find_IFFL(G)
            for A, B, C in motifs:
                sAB = '+' if G[A][B]['weight'] > 0 else '-'
                sAC = '+' if G[A][C]['weight'] > 0 else '-'
                sCB = '+' if G[C][B]['weight'] > 0 else '-'
                pattern = sAB + sAC + sCB
                counts_acc[(age, pattern)] += 1
                iter_counts_acc[(age, pattern)] += 1
                if record_acc[(age, pattern)] < max_records_per_age_pattern:
                    all_records.append({
                        'age':     age,
                        'A':       A,
                        'B':       B,
                        'C':       C,
                        'pattern': pattern
                    })
                    record_acc[(age, pattern)] += 1

        if return_iter_counts:
            for (age, pattern), count in iter_counts_acc.items():
                iter_count_records.append({
                    "iter": i,
                    "age": age,
                    "pattern": pattern,
                    "count": count,
                })

    motif_df = pd.DataFrame(all_records)
    counts_rows = [{'age': k[0], 'pattern': k[1], 'count': v / niters} for k, v in counts_acc.items()]
    counts_df = pd.DataFrame(counts_rows) if counts_rows else pd.DataFrame(columns=['age', 'pattern', 'count'])

    if return_iter_counts:
        iter_counts_df = (
            pd.DataFrame(iter_count_records)
            if iter_count_records
            else pd.DataFrame(columns=["iter", "age", "pattern", "count"])
        )
        return motif_df, counts_df, graphs_by_age, iter_counts_df
    return motif_df, counts_df, graphs_by_age



def analyze_IFL(
    theta_dense,
    theta_sparse,
    nG,
    niters,
    names_tf,
    age_sparse=24,
    age_dense=3,
    return_iter_counts=False,
    max_records_per_age_pattern=5,
):
    '''
    Inputs:
        theta_dense: the theta of the denser age
        theta_sparse: the theta for the less dense age
        nG: number of genes (TG or TF)
        names_tf: names of the TFs to make db of loops
        age_sparse: actual age of less dense age, default=24
        age_dense: actual age of more dense age, default=3
        niters: number of iterations to repeat sparsifying more dense matrix, default = 100

    Returns:
        motif_df: a df with the IFL motifs and names of tfs/tgs in them
        counts_df: a df with counts for each type of IFL loop
        graphs_by_age: the graph corresponding the theta for each age
    '''
    from collections import defaultdict
    rng = np.random.default_rng()
    counts_acc = defaultdict(float)
    all_records = []
    iter_count_records = []
    record_acc = defaultdict(int)

    G_sparse = _build_graph(theta_sparse, names_tf)

    print(f'Graph has been built')
    print(f'Begin extraction')
    for i in range(niters):
        print(f'On iter {i}')
        theta_sp = sparsify_once_to_match(theta_dense, theta_sparse, rng=rng)

        G_dense = _build_graph(theta_sp, names_tf)
        graphs_by_age = {age_dense: G_dense, age_sparse: G_sparse}

        iter_counts_acc = defaultdict(float)
        for age, G in graphs_by_age.items():
            motifs = find_IFL(G)
            for A, B, C in motifs:
                w_ab = G[A][B]['weight']
                w_cb = G[C][B]['weight']
                w_bc = G[B][C]['weight']
                sAB = '+' if w_ab > 0 else '-'
                sBC = '+' if w_bc > 0 else '-'
                sCB = '+' if w_cb > 0 else '-'
                pattern = sAB + sBC + sCB

                counts_acc[(age, pattern)] += 1
                iter_counts_acc[(age, pattern)] += 1
                if record_acc[(age, pattern)] < max_records_per_age_pattern:
                    all_records.append({
                        'age':     age,
                        'A':       A,
                        'B':       B,
                        'C':       C,
                        'pattern': pattern
                    })
                    record_acc[(age, pattern)] += 1

        if return_iter_counts:
            for (age, pattern), count in iter_counts_acc.items():
                iter_count_records.append({
                    "iter": i,
                    "age": age,
                    "pattern": pattern,
                    "count": count,
                })

    motif_df = pd.DataFrame(all_records)
    counts_rows = [{'age': k[0], 'pattern': k[1], 'count': v / niters} for k, v in counts_acc.items()]
    counts_df = pd.DataFrame(counts_rows) if counts_rows else pd.DataFrame(columns=['age', 'pattern', 'count'])

    if return_iter_counts:
        iter_counts_df = (
            pd.DataFrame(iter_count_records)
            if iter_count_records
            else pd.DataFrame(columns=["iter", "age", "pattern", "count"])
        )
        return motif_df, counts_df, graphs_by_age, iter_counts_df
    return motif_df, counts_df, graphs_by_age

##############################################################################################################
##############################################################################################################
'''
PLOTTING
'''
##############################################################################################################
##############################################################################################################
def plot_out_deg(
    out_deg_dense,
    out_deg_sparse,
    out_deg_null,
    t,
    gene_type,
    dense_age=3,
    sparse_age=24,
    dense_sparsity=None,
    sparse_sparsity=None,
):
    '''
    plot the out degree given 2 ages of out degrees and a null model. The rest is just for file writing
    '''
    niters=out_deg_dense.shape[0]
    # Plot
    fig1 = plt.figure(figsize=(7, 6)) # to plot TGs

    for i in range(niters):
        sorted_data_dense = np.sort(out_deg_dense[i])
        ccdf_dense = 1.0 - np.arange(1, len(sorted_data_dense)+1) / (len(sorted_data_dense))

        #sorted_data_null = np.sort(out_deg_null[i])
        #ccdf_null = 1.0 - np.arange(1, len(sorted_data_null)+1) / (2*len(sorted_data_null))

        # Plot each iteration as very light
        plt.figure(fig1)
        plt.plot(sorted_data_dense, ccdf_dense, markersize=4, c = 'grey', alpha = 0.4)
        #plt.plot(sorted_data_null, ccdf_null, markersize=2,c='lightgrey', alpha =.4)
    
    niters=out_deg_null.shape[0]

    for i in range(niters):
        #sorted_data_dense = np.sort(out_deg_dense[i])
        #ccdf_dense = 1.0 - np.arange(1, len(sorted_data_dense)+1) / (2*len(sorted_data_dense))

        sorted_data_null = np.sort(out_deg_null[i])
        ccdf_null = 1.0 - np.arange(1, len(sorted_data_null)+1) / (len(sorted_data_null))

        # Plot each iteration as very light
        plt.figure(fig1)
        #lt.plot(sorted_data_dense, ccdf_dense, markersize=4, c = 'grey', alpha = 0.4)
        plt.plot(sorted_data_null, ccdf_null, markersize=2,c='lightgrey', alpha =.4)

    # Plot the mean for ones with iterations and the full thing otherwise
    sorted_data_dense = np.sort(out_deg_dense.mean(axis=0))
    sorted_data_sparse = np.sort(out_deg_sparse)
    sorted_data_null = np.sort(out_deg_null.mean(axis=0))

    ccdf_dense = 1.0 - np.arange(1, len(sorted_data_dense)+1) / (len(sorted_data_dense))
    ccdf_sparse = 1.0 - np.arange(1, len(sorted_data_sparse)+1) / (len(sorted_data_sparse))
    ccdf_null = 1.0 - np.arange(1, len(sorted_data_null)+1) / (len(sorted_data_null))

    color_3m = '#76c7c0'
    color_24m = '#f9b194'

    if dense_age == 3:
        color_dense = color_3m
        color_sparse = color_24m

    else:
        color_dense = color_24m
        color_sparse = color_3m

    plt.figure(fig1)
    plt.plot(sorted_data_dense, ccdf_dense, label=f'{dense_age}', linewidth=2.3, c = color_dense)
    plt.plot(sorted_data_sparse, ccdf_sparse, label=f'{sparse_age}', linewidth=2.3, c = color_sparse)
    plt.plot(sorted_data_null, ccdf_null, label='null', linewidth=2.3,c='black', alpha =.6)

    if (dense_sparsity is not None) and (sparse_sparsity is not None):
        plt.gca().text(
            0.98,
            0.98,
            f"{dense_age}m sp={dense_sparsity:.3f}\n{sparse_age}m sp={sparse_sparsity:.3f}",
            transform=plt.gca().transAxes,
            ha='right',
            va='top',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='lightgrey'),
        )

    #plt.xlabel('Out degree', fontsize = 14)
    #plt.ylabel('P(out degree > x)', fontsize = 14)
    #plt.title(f'Out Degree Distribution {gene_type}', fontsize = 16)
    plt.legend(fontsize = 12)
    plt.grid(False)
    plt.tight_layout()

    os.makedirs("paper_figs_final", exist_ok=True)
    plt.savefig(f'paper_figs_final/out_deg_{t}_comparison_{gene_type}.pdf', dpi=200)


def plot_out_deg_eff(
    out_deg_dense,
    out_deg_sparse,
    t,
    gene_type,
    dense_age=3,
    sparse_age=24,
    dense_sparsity=None,
    sparse_sparsity=None,
):

    '''
    plot the effective out degree (based on theta weight) given 2 ages of out degrees and a null model. 
    The rest is just for file writing
    '''


    niters = out_deg_dense.shape[0]
    print(out_deg_dense.shape)
    fig1 = plt.figure(figsize=(6, 4)) # to plot TGs


    for i in range(niters):
        sorted_data_dense = np.sort(out_deg_dense[i])
        ccdf_dense = 1.0 - np.arange(1, len(sorted_data_dense)+1) / len(sorted_data_dense)

        # Plot each iteration as very light
        plt.figure(fig1)
        plt.plot(sorted_data_dense, ccdf_dense, markersize=4, c = 'grey', alpha = 0.4)

    # Plot the mean for ones with iterations and the full thing otherwise
    sorted_data_dense = np.sort(out_deg_dense.mean(axis=0))
    sorted_data_sparse = np.sort(out_deg_sparse)

    ccdf_dense = 1.0 - np.arange(1, len(sorted_data_dense)+1) / len(sorted_data_dense)
    ccdf_sparse = 1.0 - np.arange(1, len(sorted_data_sparse)+1) / len(sorted_data_sparse)

    plt.figure(fig1)
    plt.plot(sorted_data_dense, ccdf_dense, label=f'{dense_age}', linewidth=2.3)
    plt.plot(sorted_data_sparse, ccdf_sparse, label=f'{sparse_age}', linewidth=2.3)

    if (dense_sparsity is not None) and (sparse_sparsity is not None):
        plt.gca().text(
            0.98,
            0.98,
            f"{dense_age}m sp={dense_sparsity:.3f}\n{sparse_age}m sp={sparse_sparsity:.3f}",
            transform=plt.gca().transAxes,
            ha='right',
            va='top',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='lightgrey'),
        )

    plt.xlabel('Out degree', fontsize = 14)
    plt.ylabel('P(out degree > x)', fontsize = 14)
    plt.title(f'Out Degree Distribution {gene_type}', fontsize = 16)
    plt.legend(fontsize = 12)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f'out_deg_{t}_comparison_{gene_type}_eff')


def plot_out_deg_top_pct(out_deg_a, out_deg_b, t, gene_type, age_a, age_b, pct=0.1, effective=False):
    """Plot CCDF of out degrees for top-pct filtered thetas, comparing two ages.

    Parameters
    ----------
    out_deg_a, out_deg_b : 1-D arrays
        Out degree vectors for each age (one per TF/gene).
    t : str
        Tissue name (for file naming).
    gene_type : str
        'tf' or 'tg'.
    age_a, age_b : int
        Ages in months.
    pct : float
        The percentile used for filtering (for labeling only).
    effective : bool
        If True, labels indicate effective (weighted) out degree.
    """
    color_3m = '#76c7c0'
    color_24m = '#f9b194'

    color_a = color_3m if age_a <= age_b else color_24m
    color_b = color_24m if age_a <= age_b else color_3m

    fig = plt.figure(figsize=(7, 6))

    sorted_a = np.sort(out_deg_a)
    sorted_b = np.sort(out_deg_b)
    ccdf_a = 1.0 - np.arange(1, len(sorted_a) + 1) / len(sorted_a)
    ccdf_b = 1.0 - np.arange(1, len(sorted_b) + 1) / len(sorted_b)

    plt.plot(sorted_a, ccdf_a, label=f'{age_a}m', linewidth=2.3, c=color_a)
    plt.plot(sorted_b, ccdf_b, label=f'{age_b}m', linewidth=2.3, c=color_b)

    deg_label = 'Effective out degree' if effective else 'Out degree'
    pct_label = f'top {int(pct * 100)}%'
    plt.xlabel(deg_label, fontsize=14)
    plt.ylabel(f'P({deg_label.lower()} > x)', fontsize=14)
    plt.title(f'{deg_label} Distribution ({pct_label}) {gene_type.upper()}', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(False)
    plt.tight_layout()

    eff_tag = '_eff' if effective else ''
    os.makedirs("paper_figs_final", exist_ok=True)
    plt.savefig(f'paper_figs_final/out_deg_{t}_{gene_type}_top{int(pct*100)}pct{eff_tag}.pdf', dpi=200)
    plt.close(fig)


def plot_3_node_feedback(motif_df, counts_df, graphs_by_age, tissue):

    bar_df = counts_df.pivot(index='age', columns='pattern', values='count').fillna(0)

    #### Stacked bar plot of types of motifs
    plt.figure(figsize=(6,4))
    bar_df.plot(kind='bar', stacked=True, edgecolor='black', legend=False)
    plt.xlabel("Age (months)")
    plt.ylabel("Number of A→B↔C motifs")
    plt.title("Composition of input–feedback motifs by sign-pattern TF-TF Interactions")
    plt.xticks(rotation=0)
    plt.legend(title="pattern", bbox_to_anchor=(1.02,1))
    plt.tight_layout()
    plt.show()
    plt.savefig(f'stacked_bar_plot_3_node_feeback_{tissue}')

    ########### heatmap with number of each motif
    plt.figure(figsize=(4,6))
    sns.heatmap(bar_df.T, annot=True, fmt='g', cbar_kws={'label':'Count'})
    plt.xlabel("Age (months)")
    plt.ylabel("Sign-pattern")
    plt.title("Pattern frequencies across ages")
    plt.tight_layout()
    plt.show()
    plt.savefig(f'heatmap_3_node_feeback_{tissue}')


    patterns = sorted(bar_df.columns)
    ages     = sorted(bar_df.index)

    ############ example connections
    fig, axes = plt.subplots(len(patterns), len(ages),
                            figsize=(3*len(ages), 2*len(patterns)),
                            squeeze=False)

    for i, pat in enumerate(patterns):
        for j, age in enumerate(ages):
            subset = motif_df[(motif_df.age==age)&(motif_df.pattern==pat)]
            if subset.empty:
                axes[i][j].axis('off')
                continue
            ex = subset.iloc[0]   # pick first example
            A,B,C = ex['A'], ex['B'], ex['C']
            print(A, B, C)
            H = graphs_by_age[age].subgraph([A,B,C]).copy()
            pos = {A:(0,1), B:(0,0), C:(1,0)}
            edge_colors = ['green' if H[u][v]['weight']<0 else 'red' if H[u][v]['weight']>0 else 'white'
                        for u,v in H.edges()]
            nx.draw(H, pos, ax=axes[i][j],
                    with_labels=True, edge_color=edge_colors,
                    arrowsize=12)
            axes[i][j].set_title(f"{pat} @ {age}m")
            axes[i][j].axis('off')

    plt.tight_layout()
    plt.show()
    plt.savefig(f'example_connections_3_node_feeback_{tissue}')

def plot_IFFL(motif_df, counts_df, graphs_by_age, tissue):

    bar_df = counts_df.pivot(index='age', columns='pattern', values='count').fillna(0)

    #### Stacked bar plot of types of motifs
    plt.figure(figsize=(6,4))
    bar_df.plot(kind='bar', stacked=True, edgecolor='black', legend=False)
    plt.xlabel("Age (months)")
    plt.ylabel("Number of A→B↔C motifs")
    plt.title("Composition of input–feedback motifs by sign-pattern TF-TF Interactions")
    plt.xticks(rotation=0)
    plt.legend(title="pattern", bbox_to_anchor=(1.02,1))
    plt.tight_layout()
    plt.show()
    plt.savefig(f'stacked_bar_plot_IFFL_{tissue}')

    ########### heatmap with number of each motif
    plt.figure(figsize=(4,6))
    sns.heatmap(bar_df.T, annot=True, fmt='g', cbar_kws={'label':'Count'})
    plt.xlabel("Age (months)")
    plt.ylabel("Sign-pattern")
    plt.title("Pattern frequencies across ages")
    plt.tight_layout()
    plt.show()
    plt.savefig(f'heatmap_IFFL_{tissue}')


    patterns = sorted(bar_df.columns)
    ages     = sorted(bar_df.index)

    ############ example connections
    fig, axes = plt.subplots(len(patterns), len(ages),
                            figsize=(3*len(ages), 2*len(patterns)),
                            squeeze=False)

    for i, pat in enumerate(patterns):
        for j, age in enumerate(ages):
            subset = motif_df[(motif_df.age==age)&(motif_df.pattern==pat)]
            if subset.empty:
                axes[i][j].axis('off')
                continue
            ex = subset.iloc[0]   # pick first example
            A,B,C = ex['A'], ex['B'], ex['C']
            print(A, B, C)
            H = graphs_by_age[age].subgraph([A,B,C]).copy()
            pos = {A:(0,1), B:(0,0), C:(1,0)}
            edge_colors = ['green' if H[u][v]['weight']<0 else 'red' if H[u][v]['weight']>0 else 'white'
                        for u,v in H.edges()]
            nx.draw(H, pos, ax=axes[i][j],
                    with_labels=True, edge_color=edge_colors,
                    arrowsize=12)
            axes[i][j].set_title(f"{pat} @ {age}m")
            axes[i][j].axis('off')

    plt.tight_layout()
    plt.show()
    plt.savefig(f'example_connections_IFFL_{tissue}')

def plot_2nodes(motif_df, counts_df, graphs_by_age, tissue):
    # pivot for plotting
    bar_df = counts_df.pivot(index='age', columns='pattern', values='count').fillna(0)
    # rows = ages, cols = sign-patterns

    plt.figure(figsize=(6,4))
    bar_df.plot(kind='bar', stacked=True, edgecolor='black', legend=False)
    plt.xlabel("Age (months)")
    plt.ylabel("Number of A<→>B motifs")
    plt.title("Composition of input–feedback motifs by sign-pattern TF-TF Interactions")
    plt.xticks(rotation=0)
    plt.legend(title="pattern", bbox_to_anchor=(1.02,1))
    plt.tight_layout()
    plt.show()
    plt.savefig(f'stacked_bar_plot_2_node_feeback_{tissue}')


    plt.figure(figsize=(4,6))
    sns.heatmap(bar_df.T, annot=True, fmt='g', cbar_kws={'label':'Count'})
    plt.xlabel("Age (months)")
    plt.ylabel("Sign-pattern")
    plt.title("Pattern frequencies across ages")
    plt.tight_layout()
    plt.show()
    plt.savefig(f'heatmap_2_node_feeback_{tissue}')

    patterns = sorted(bar_df.columns)
    ages     = sorted(bar_df.index)

    fig, axes = plt.subplots(len(patterns), len(ages),
                            figsize=(3*len(ages), 2*len(patterns)),
                            squeeze=False)

    for i, pat in enumerate(patterns):
        for j, age in enumerate(ages):
            subset = motif_df[(motif_df.age==age)&(motif_df.pattern==pat)]
            if subset.empty:
                axes[i][j].axis('off')
                continue
            ex = subset.iloc[0]   # pick first example
            A,B = ex['A'], ex['B']
            H = graphs_by_age[age].subgraph([A,B]).copy()
            pos = {A:(0,1), B:(0,0)}
            edge_colors = ['green' if H[u][v]['weight']>0 else 'red'
                        for u,v in H.edges()]
            nx.draw(H, pos, ax=axes[i][j],
                    with_labels=True, edge_color=edge_colors,
                    arrowsize=12)
            axes[i][j].set_title(f"{pat} @ {age}m")
            axes[i][j].axis('off')

    plt.tight_layout()
    plt.show()
    plt.savefig(f'example_connections_2_node_feeback_{tissue}')


def analyze_struct_2nodes(
    theta_dense,
    theta_sparse,
    nG,
    niters,
    names_tf,
    age_sparse = 24,
    age_dense = 3,
    no_parents=True,
    max_records_per_age_pattern=5,
):
    from collections import defaultdict
    rng = np.random.default_rng()
    counts_acc = defaultdict(float)
    all_records = []
    record_acc = defaultdict(int)
    find_func = find_two_cycles if no_parents else find_two_cycles_no_parent

    G_sparse = _build_graph(theta_sparse, names_tf)

    for i in range(niters):
        theta_sp = sparsify_once_to_match(theta_dense, theta_sparse, rng=rng)

        G_dense = _build_graph(theta_sp, names_tf)
        graphs_by_age = {age_dense: G_dense, age_sparse: G_sparse}

        for age, G in graphs_by_age.items():
            motifs = find_func(G)
            for A, B in motifs:
                w_ab = G[A][B]['weight']
                w_ba = G[B][A]['weight']

                sAB = '+' if w_ab > 0 else '-'
                sBA = '+' if w_ba > 0 else '-'
                pattern = sAB + sBA

                counts_acc[(age, pattern)] += 1
                if record_acc[(age, pattern)] < max_records_per_age_pattern:
                    all_records.append({
                        'age':     age,
                        'A':       A,
                        'B':       B,
                        'pattern': pattern
                    })
                    record_acc[(age, pattern)] += 1

    motif_df = pd.DataFrame(all_records)
    counts_rows = [{'age': k[0], 'pattern': k[1], 'count': v / niters} for k, v in counts_acc.items()]
    counts_df = pd.DataFrame(counts_rows) if counts_rows else pd.DataFrame(columns=['age', 'pattern', 'count'])

    return motif_df, counts_df, graphs_by_age
