
import argparse
from importlib import reload
import numpy as np
import torch
import h5py
from importlib import reload
import pandas as pd
from scipy import sparse
import seaborn as sns
import networkx as nx
import os
import matplotlib.pyplot as plt


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
    theta_dense: a matrix with a lower sparsity then
    theta_sparse: the matrix with higher sparsity
    nTF: # of TFs (or genes) in the matrix
    niters: # of iterations of sparsifying to avg to get out defree

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

        if (sp_new != sp_old):
            print(f'YIKES on TFs {sp_new}, {sp_old}')
            break

        out_deg_sp[i] = np.abs(theta_sp_bin).sum(axis=1)
        out_deg_sp_eff[i] = np.abs(theta_sp).sum(axis=1)

    return theta_sp, out_deg_sp, out_deg_sp_eff, n0_diff, sp_new

def get_null(theta_dense, theta_sparse, nTF, niters=100):
    '''
    code to calculate null model for literal out degree. This is only based on the sparsity of the most
    sparse of the ages. For effective null model, you have to consider each age seperately.
    '''
    rng = np.random.default_rng()
    out_deg_null = np.zeros((niters*2, nTF))

    for i in range(0, (niters*2)-1, 2):
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

def plot_out_deg(out_deg_dense, out_deg_sparse, out_deg_null, t, gene_type, dense_age = 3, sparse_age = 24):
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

    #plt.xlabel('Out degree', fontsize = 14)
    #plt.ylabel('P(out degree > x)', fontsize = 14)
    #plt.title(f'Out Degree Distribution {gene_type}', fontsize = 16)
    plt.legend(fontsize = 12)
    plt.grid(False)
    plt.tight_layout()

    plt.savefig(f'paper_figs_final/out_deg_{t}_comparison_{gene_type}.pdf', dpi=200)


def plot_out_deg_eff(out_deg_dense, out_deg_sparse, t, gene_type,dense_age = 3, sparse_age = 24):
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

    plt.xlabel('Out degree', fontsize = 14)
    plt.ylabel('P(out degree > x)', fontsize = 14)
    plt.title(f'Out Degree Distribution {gene_type}', fontsize = 16)
    plt.legend(fontsize = 12)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f'out_deg_{t}_comparison_{gene_type}_eff')


#############################################################
#*******FUNCTIIONS FOR FINDING DIFFERENT MOTIFS*********
#############################################################

def find_input_feedback(G):
    """
    Return (A,B,C) triples where edges A→B, B→C, and C→B all exist.
    """
    motifs = []
    for B, C in G.edges():
        if G.has_edge(C, B):            # there's B<->C
            for A in G.predecessors(B):
                if A not in (B, C):
                    if ((B not in G.predecessors(A)) and (C not in G.predecessors(A)) and (A not in G.predecessors(C))): # there is no B->A or C->A
                        motifs.append((A, B, C))
    return motifs


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
    for A, B in G.edges():
        if G.has_edge(B, A):
            count1+=1
            if (G.in_degree(A) == 1) and (G.in_degree(B) == 1):
                print(f'found 1: {A}, {B}')
                cycles.add((A, B))
            for C in G.nodes():
                if C not in (A, B):
                    if (C in G.predecessors(A)) or (C in G.predecessors(B)):
                        count+=1
                        break
    print(f'there are {count1} normal cycles and then {count} that don\'t count')
    return list(cycles)

    
def find_IFFL(G):
    """
    Return (A,B,C) triples where edges A→B, B→C, and C→B all exist.
    """
    motifs = []
    for C, B in G.edges():
        if B not in G.predecessors(C):
            for A in G.predecessors(B):
                if (G.has_edge(A, B)) and (B not in G.predecessors(A)): 
                    if (G.has_edge(A, C)) and (C not in G.predecessors(A)): 
                        motifs.append((A, B, C))
    return list({tuple(m) for m in motifs}) #motifs

def find_IFL(G):
    """
    Return (A,B,C) triples where edges A→B, B→C, and C→B all exist but not C->A or B->A
    """
    motifs = []
    for B, C in G.edges():
        if G.has_edge(C, B):
            for A in G.predecessors(B):            
                if A not in (B, C):
                    if (B not in G.predecessors(A)) and (C not in G.predecessors(A)) and (A not in G.predecessors(C)): # there is no B->A or C->A
                        motifs.append((A, B, C))
    return list({tuple(m) for m in motifs}) 

def analyze_3_node_feedback(theta_dense, theta_sparse, nG, niters, names_tf, age_sparse = 24, age_dense = 3):
    theta_sp, _, _, _, _ = correct_sparsity(theta_dense, theta_sparse, nG, niters=100)

    # Convert to CSR if not already
    M_dense = theta_sp
    M_sparse = theta_sparse
    counts_df = pd.DataFrame(columns=['age', 'pattern', 'count'])

    for i in range(niters):
        print(f'******* On iter {i} ********')
        theta_sp, _, _, _, _ = correct_sparsity(theta_dense, theta_sparse, nG, niters=100)

        # Convert to CSR if not already
        M3 = theta_sp
        M24 = theta_sparse
        matrices = {
            3:  sparse.csr_matrix(M3) if not sparse.isspmatrix_csr(M3)  else M3,
            24: sparse.csr_matrix(M24) if not sparse.isspmatrix_csr(M24) else M24
        }

        graphs_by_age = {}
        for age, M in matrices.items():
            # get non-zero coords
            coo = M.tocoo()
            # map index→gene
            src = [names_tf[i] for i in coo.row]
            tgt = [names_tf[j] for j in coo.col]
            wts = coo.data
            # build DiGraph
            df_edges = pd.DataFrame({'source':src,'target':tgt,'weight':wts})
            G = nx.from_pandas_edgelist(df_edges, 'source','target', edge_attr='weight',
                                        create_using=nx.DiGraph())
            graphs_by_age[age] = G

        records = []
        for age, G in graphs_by_age.items():
            for A, B, C in find_input_feedback(G):
                sAB = '+' if G[A][B]['weight'] > 0 else '-'
                sBC = '+' if G[B][C]['weight'] > 0 else '-'
                sCB = '+' if G[C][B]['weight'] > 0 else '-'
                pattern = sAB + sBC + sCB
                records.append({
                    'age':     age,
                    'A':       A,
                    'B':       B,
                    'C':       C,
                    'pattern': pattern
                })

        motif_df = pd.DataFrame(records)

        counts_df_current = (
            motif_df
            .groupby(['age','pattern'])
            .size()
            .reset_index(name='count')
        )

        for _, row in counts_df_current.iterrows():
            age = row['age']
            pattern = row['pattern']
            count = row['count']
            mask = (counts_df['age'] == age) & (counts_df['pattern'] == pattern)

            if mask.any():
                counts_df.loc[mask, 'count'] += count
            else:
                new_row = {'age': age, 'pattern': pattern, 'count': count}
                counts_df = pd.concat([counts_df, pd.DataFrame([new_row])], ignore_index=True)
        

    for _, row in counts_df.iterrows():
        age = row['age']
        pattern = row['pattern']
        count = row['count']
        counts_df.loc[(counts_df['age'] == age) & (counts_df['pattern'] == pattern), 'count'] /= niters+1

    return motif_df, counts_df, graphs_by_age

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

def analyze_struct_2nodes(theta_dense, theta_sparse, nG, niters, names_tf, age_sparse = 24, age_dense = 3, no_parents=True):

    theta_sp, _, _, _, _ = correct_sparsity(theta_dense, theta_sparse, nG, niters=niters)

    # Convert to CSR if not already
    M3 = theta_sp # dense one
    M24 = theta_sparse
    counts_df = pd.DataFrame(columns=['age', 'pattern', 'count'])
    
    for i in range(niters):

        theta_sp, _, _, _, _ = correct_sparsity(theta_dense, theta_sparse, nG, niters=niters)
        # Convert to CSR if not already
        M3 = theta_sp
        matrices = {
            3:  sparse.csr_matrix(M3) if not sparse.isspmatrix_csr(M3)  else M3,
            24: sparse.csr_matrix(M24) if not sparse.isspmatrix_csr(M24) else M24
        }

        graphs_by_age = {}
        for age, M in matrices.items():
            # get non-zero coords
            coo = M.tocoo()
            # map index→gene
            src = [names_tf[i] for i in coo.row]
            tgt = [names_tf[j] for j in coo.col]
            wts = coo.data
            # build DiGraph
            df_edges = pd.DataFrame({'source':src,'target':tgt,'weight':wts})
            G = nx.from_pandas_edgelist(df_edges, 'source','target', edge_attr='weight',
                                        create_using=nx.DiGraph())
            graphs_by_age[age] = G


        records = []
        if no_parents == True:
            for age, G in graphs_by_age.items():
                for A, B in find_two_cycles_no_parent(G):
                    # pull weights just once
                    w_ab = G[A][B]['weight']
                    w_ba = G[B][A]['weight']
                    
                    sAB = '+' if w_ab > 0 else '-'
                    sBA = '+' if w_ba > 0 else '-'
                    pattern = sAB + sBA
        
                    records.append({
                        'age':     age,
                        'A':       A,
                        'B':       B,
                        'pattern': pattern
                    })
        else:
            for age, G in graphs_by_age.items():
                for A, B in find_two_cycles_no_parent(G):
                    # pull weights just once
                    w_ab = G[A][B]['weight']
                    w_ba = G[B][A]['weight']
                    
                    sAB = '+' if w_ab > 0 else '-'
                    sBA = '+' if w_ba > 0 else '-'
                    pattern = sAB + sBA
        
                    records.append({
                        'age':     age,
                        'A':       A,
                        'B':       B,
                        'pattern': pattern
                    })


        motif_df = pd.DataFrame(records)

        counts_df_current = (
            motif_df
            .groupby(['age','pattern'])
            .size()
            .reset_index(name='count')
        )
        for _, row in counts_df_current.iterrows():
            age = row['age']
            pattern = row['pattern']
            count = row['count']
            mask = (counts_df['age'] == age) & (counts_df['pattern'] == pattern)

            if mask.any():
                counts_df.loc[mask, 'count'] += count
            else:
                new_row = {'age': age, 'pattern': pattern, 'count': count}
                counts_df = pd.concat([counts_df, pd.DataFrame([new_row])], ignore_index=True)

    for _, row in counts_df.iterrows():
        age = row['age']
        pattern = row['pattern']
        count = row['count']
        counts_df.loc[(counts_df['age'] == age) & (counts_df['pattern'] == pattern), 'count'] /= niters+1

    return motif_df, counts_df, graphs_by_age

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


####################
    ###############
#######################


def analyze_IFFL(theta_dense, theta_sparse, nG, niters, names_tf, age_sparse = 24, age_dense = 3):

    theta_sp, _, _, _, _ = correct_sparsity(theta_dense, theta_sparse, nG, niters=100)

    # Convert to CSR if not already
    M_dense = theta_sp
    M_sparse = theta_sparse
    counts_df = pd.DataFrame(columns=['age', 'pattern', 'count'])

    for i in range(niters):
        print(f'******* On iter {i} ********')
        theta_sp, _, _, _, _ = correct_sparsity(theta_dense, theta_sparse, nG, niters=100)

        # Convert to CSR if not already
        M3 = theta_sp
        M24 = theta_sparse
        matrices = {
            3:  sparse.csr_matrix(M3) if not sparse.isspmatrix_csr(M3)  else M3,
            24: sparse.csr_matrix(M24) if not sparse.isspmatrix_csr(M24) else M24
        }

        graphs_by_age = {}
        for age, M in matrices.items():
            # get non-zero coords
            coo = M.tocoo()
            # map index→gene
            src = [names_tf[i] for i in coo.row]
            tgt = [names_tf[j] for j in coo.col]
            wts = coo.data
            # build DiGraph
            df_edges = pd.DataFrame({'source':src,'target':tgt,'weight':wts})
            G = nx.from_pandas_edgelist(df_edges, 'source','target', edge_attr='weight',
                                        create_using=nx.DiGraph())
            graphs_by_age[age] = G

        records = []
        for age, G in graphs_by_age.items():
            for A, B, C in find_IFFL(G):
                sAB = '+' if G[A][B]['weight'] > 0 else '-'
                sAC = '+' if G[A][C]['weight'] > 0 else '-'
                sCB = '+' if G[C][B]['weight'] > 0 else '-'
                pattern = sAB + sAC + sCB
                records.append({
                    'age':     age,
                    'A':       A,
                    'B':       B,
                    'C':       C,
                    'pattern': pattern
                })

        motif_df = pd.DataFrame(records)

        counts_df_current = (
            motif_df
            .groupby(['age','pattern'])
            .size()
            .reset_index(name='count')
        )

        for _, row in counts_df_current.iterrows():
            age = row['age']
            pattern = row['pattern']
            count = row['count']
            mask = (counts_df['age'] == age) & (counts_df['pattern'] == pattern)

            if mask.any():
                counts_df.loc[mask, 'count'] += count
            else:
                new_row = {'age': age, 'pattern': pattern, 'count': count}
                counts_df = pd.concat([counts_df, pd.DataFrame([new_row])], ignore_index=True)
        
    for _, row in counts_df.iterrows():
        age = row['age']
        pattern = row['pattern']
        count = row['count']
        counts_df.loc[(counts_df['age'] == age) & (counts_df['pattern'] == pattern), 'count'] /= niters+1

    return motif_df, counts_df, graphs_by_age

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

def analyze_IFL(theta_dense, theta_sparse, nG, niters, names_tf, age_sparse = 24, age_dense = 3):

    theta_sp, _, _, _, _ = correct_sparsity(theta_dense, theta_sparse, nG, niters=niters)
    counts_df = pd.DataFrame(columns=['age', 'pattern', 'count'])
    # Convert to CSR if not already
    M3 = theta_sp
    M24 = theta_sparse
    print(f'Graph has been built')
    print(f'Begin extraction')
    for i in range(niters):
        print(f'On iter {i}')
        theta_sp, _, _, _, _ = correct_sparsity(theta_dense, theta_sparse, nG, niters=niters)
        # Convert to CSR if not already
        M3 = theta_sp
        matrices = {
            3:  sparse.csr_matrix(M3) if not sparse.isspmatrix_csr(M3)  else M3,
            24: sparse.csr_matrix(M24) if not sparse.isspmatrix_csr(M24) else M24
        }

        graphs_by_age = {}
        for age, M in matrices.items():
            # get non-zero coords
            coo = M.tocoo()
            # map index→gene
            src = [names_tf[i] for i in coo.row]
            tgt = [names_tf[j] for j in coo.col]
            wts = coo.data
            # build DiGraph
            df_edges = pd.DataFrame({'source':src,'target':tgt,'weight':wts})
            G = nx.from_pandas_edgelist(df_edges, 'source','target', edge_attr='weight',
                                        create_using=nx.DiGraph())
            graphs_by_age[age] = G


        records = []
        for age, G in graphs_by_age.items():
            for A, B, C in find_IFL(G):
                # pull weights just once
                w_ab = G[A][B]['weight']
                w_cb = G[C][B]['weight']
                w_bc = G[B][C]['weight']
                #print(f'we got {G[A][B], G[C][B], G[B][C]}')
                sAB = '+' if w_ab > 0 else '-'
                sBC = '+' if w_bc > 0 else '-'
                sCB = '+' if w_cb > 0 else '-'
                pattern = sAB + sBC + sCB

                records.append({
                    'age':     age,
                    'A':       A,
                    'B':       B,
                    'C':       C,
                    'pattern': pattern
                })

        motif_df = pd.DataFrame(records)

        counts_df_current = (
            motif_df
            .groupby(['age','pattern'])
            .size()
            .reset_index(name='count')
        )
        for _, row in counts_df_current.iterrows():
            age = row['age']
            pattern = row['pattern']
            count = row['count']
            mask = (counts_df['age'] == age) & (counts_df['pattern'] == pattern)

            if mask.any():
                counts_df.loc[mask, 'count'] += count
            else:
                new_row = {'age': age, 'pattern': pattern, 'count': count}
                counts_df = pd.concat([counts_df, pd.DataFrame([new_row])], ignore_index=True)

    for _, row in counts_df.iterrows():
        age = row['age']
        pattern = row['pattern']
        count = row['count']
        counts_df.loc[(counts_df['age'] == age) & (counts_df['pattern'] == pattern), 'count'] /= niters+1                                      

    return motif_df, counts_df, graphs_by_age   