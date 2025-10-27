import argparse
from importlib import reload
import numpy as np
import torch
import get_st_funcs_generalized as gs
import h5py
import os
import functools
import random
import matplotlib.pyplot as plt
reload(gs)

# This file is intended to be adpoted into a shell script so you can add parameters to it. It has a single function

def calc_all_structs(t, theta_tf3, theta_tf24, theta_tg3, theta_tg24, factor=0.01):
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
  
    names_tf = np.load(f'data/final_tf_names_{t}.npy', allow_pickle=True)

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
    np.save(f'out_deg_{t}_{age_tf_dense}m_tf_corrected_sp', out_deg_tf_sparsified)
    np.save(f'out_deg_{t}_{age_tg_dense}m_tg_corrected_sp', out_deg_tg_sparsified)
    
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
    motif_df.to_csv(f'IFL_df_tf_{t}.csv', index=False)
    counts_df.to_csv(f'IFL_counts_df_tf_{t}.csv', index=False)

    # Calc and save IFFL
    motif_df, counts_df, graphs_by_age = gs.analyze_IFFL(theta_tf_sparsified, theta_tf_sparse, nTF, niters=30, names_tf = names_tf, age_sparse=age_tf_sparse, age_dense=age_tf_dense)
    motif_df.to_csv(f'IFFL_df_tf_{t}.csv', index=False)
    counts_df.to_csv(f'IFFL_counts_df_tf_{t}.csv', index=False)

    ############## Look at 3 node feedback stuff ##################
    motif_df, counts_df, graphs_by_age = gs.analyze_3_node_feedback(theta_tf_sparsified, theta_tf_sparse, nTF, niters=50, names_tf = names_tf, age_sparse=age_tf_sparse, age_dense=age_tf_dense)
    motif_df.to_csv(f'3_node_feedback_df_tf_{t}.csv', index=False)
    counts_df.to_csv(f'3_node_feedback_counts_df_tf_{t}.csv', index=False)

    ############## Look at 2 node feedback stuff ##################
    motif_df, counts_df, graphs_by_age = gs.analyze_struct_2nodes(theta_tf_sparsified, theta_tf_sparse, nTF, niters=50, names_tf = names_tf, age_sparse=age_tf_sparse, age_dense=age_tf_dense)
    motif_df.to_csv(f'2_node_feedback_df_{t}.csv', index=False)
    counts_df.to_csv(f'2_node_feedback_counts_df_{t}.csv', index=False)
    
    ############ Now plot out degrees ###########

    null_out_deg_tf = gs.get_null(theta_tf3, theta_tf24, nTF, niters=100)
    null_out_deg_tg = gs.get_null(theta_tg3, theta_tg24, nTF, niters=100)
    
    out_deg_tf_sparse = theta_tf_sparse_bin.sum(axis=1); out_deg_tg_sparse = theta_tg_sparse_bin.sum(axis=1)
    out_deg_tf_sparse_eff = np.abs(theta_tf_sparse).sum(axis=1); out_deg_tg_sparse_eff = np.abs(theta_tg_sparse).sum(axis=1)

    # Plot real out degrees
    gs.plot_out_deg(out_deg_tf_sparsified, out_deg_tf_sparse, null_out_deg_tf, t, 'tf', dense_age = 3, sparse_age = 24)
    gs.plot_out_deg(out_deg_tg_sparsified, out_deg_tg_sparse, null_out_deg_tg, t, 'tg', dense_age = 3, sparse_age = 24)
    
    # Plot effective out degrees
    gs.plot_out_deg_eff(out_deg_tf_sparsified_eff, out_deg_tf_sparse_eff, t, 'tf')
    gs.plot_out_deg_eff(out_deg_tg_sparsified_eff, out_deg_tg_sparse_eff, t, 'tg')
    gs.plot_2nodes(motif_df, counts_df, graphs_by_age, t)
    
