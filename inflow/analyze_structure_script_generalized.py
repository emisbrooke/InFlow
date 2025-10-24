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

def main(t, lam3, lam24):
    factor = 0.01 # Cutoff for analysis
    gene_data3 = np.load(f'data/{t}_tg_data_binary_3m_filt.npy') # Actual TG data
    tf_data3 = np.load(f'data/{t}_tf_data_binary_3m_filt.npy') # Actual TF data
    age = 24
    gene_data24 = np.load(f'data/{t}_tg_data_binary_{age}m_filt.npy')
    tf_data24 = np.load(f'data/{t}_tf_data_binary_{age}m_filt.npy')
    
    names_tf = np.load(f'data/final_tf_names_{t}.npy', allow_pickle=True)
    #if os.path.exists(f'IFL_df_{t}.csv'):
    #    print(f'already ran for {t}')
    #    return
    nTF, nC3 = tf_data3.shape
    nTF, nC24 = tf_data24.shape
    nG, _ = gene_data3.shape
    
    ################ Load parameters for 24m ###################
    directory_path = f'../project/Aging/output_tf_{t}/{age}m/'
    pi_tf24, theta_tf24, m_tf24 = gs.load_h5_param(t, directory_path, age, lam24) 
    
    #### TG model parameters
    directory_path = f'../project/Aging/output_tg_{t}/{age}m/'
    pi_tg24, theta_tg24, m_tg24 = gs.load_h5_param(t, directory_path, age, lam24) 
    
    ################ Load parameters for 3m ###################
    print('Loading model paramters....')
    age = 3
    
    #### TG model parameters
    directory_path = f'../project/Aging/output_tg_{t}/{age}m/'
    pi_tg3, theta_tg3, m_tg3 = gs.load_h5_param(t, directory_path, age, lam3) 
    
    #### TF model parameters
    directory_path = f'../project/Aging/output_tf_{t}/{age}m/'
    pi_tf3, theta_tf3, m_tf3 = gs.load_h5_param(t, directory_path, age, lam3) 
    
    #### Cut off theta to take away weak connections
    theta_tf3[np.abs(theta_tf3) < factor] = 0; theta_tf24[np.abs(theta_tf24) < factor] = 0
    theta_tg3[np.abs(theta_tg3) < factor] = 0; theta_tg24[np.abs(theta_tg24) < factor] = 0
    
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
    
    theta_tf_sparse_bin = theta_tf_sparse.copy(); theta_tf_sparse_bin[np.abs(theta_tf_sparse_bin) > 0] = 1
    theta_tg_sparse_bin = theta_tg_sparse.copy(); theta_tg_sparse_bin[np.abs(theta_tg_sparse_bin) > 0] = 1
    
    # Correct for sparsity 
    theta_tf_sp, out_deg_tf_dense, out_deg_tf_dense_eff, n0_diff_tf, sp_new_tf = gs.correct_sparsity(theta_tf_dense, theta_tf_sparse, nTF, niters=50)
    #theta_tg_sp, out_deg_tg_dense, out_deg_tg_dense_eff, n0_diff_tg, sp_new_tg = gs.correct_sparsity(theta_tg_dense, theta_tg_sparse, nTF, niters=50)
    theta_tf_sp_bin = theta_tf_sp.copy(); theta_tf_sp_bin[np.abs(theta_tf_sp_bin) > 0] = 1
    #theta_tg_sp_bin = theta_tg_sp.copy(); theta_tg_sp_bin[np.abs(theta_tg_sp_bin) > 0] = 1
    
    print(f'remaining sparsity for sparse TF is {sp_sparse_tf} and {sp_new_tf}')
    #print(f'remaining sparsity for sparse TG is {sp_sparse_tg} and {sp_new_tg}')
    ################### IFFL ######################
    #50 iters
    motif_df, counts_df, graphs_by_age = gs.analyze_IFL(theta_tf_sp, theta_tf_sparse, nTF, niters=20, names_tf = names_tf, age_sparse=age_tf_sparse, age_dense=age_tf_dense)
    # now plot them
    motif_df.to_csv(f'IFL_df_{t}.csv', index=False)
    counts_df.to_csv(f'IFL_counts_df_{t}.csv', index=False)

    motif_df, counts_df, graphs_by_age = gs.analyze_IFFL(theta_tf_sp, theta_tf_sparse, nTF, niters=30, names_tf = names_tf, age_sparse=age_tf_sparse, age_dense=age_tf_dense)
    # now plot them
    motif_df.to_csv(f'IFFL_df_{t}.csv', index=False)
    counts_df.to_csv(f'IFFL_counts_df_{t}.csv', index=False)
    
    #gs.plot_IFL(motif_df, counts_df, graphs_by_age, t)

    '''
    np.save(f'out_deg_{t}_{age_tf_dense}m_tf_corrected_sp', out_deg_tf_dense)
    np.save(f'out_deg_{t}_{age_tg_dense}m_tg_corrected_sp', out_deg_tg_dense)
    
    null_out_deg_tf = gs.get_null(theta_tf3, theta_tf24, nTF, niters=100)
    null_out_deg_tg = gs.get_null(theta_tg3, theta_tg24, nTF, niters=100)
    
    out_deg_tf_sparse = theta_tf_sparse_bin.sum(axis=1); out_deg_tg_sparse = theta_tg_sparse_bin.sum(axis=1)
    out_deg_tf_sparse_eff = np.abs(theta_tf_sparse).sum(axis=1); out_deg_tg_sparse_eff = np.abs(theta_tg_sparse).sum(axis=1)
    
    
    ############ Now plot out degrees ###########
    # TF real
    gs.plot_out_deg(out_deg_tf_dense, out_deg_tf_sparse, null_out_deg_tf, t, 'tf', dense_age = 3, sparse_age = 24)
    # TG real
    gs.plot_out_deg(out_deg_tg_dense, out_deg_tg_sparse, null_out_deg_tg, t, 'tg', dense_age = 3, sparse_age = 24)
    
    # TF effective
    gs.plot_out_deg_eff(out_deg_tf_dense_eff, out_deg_tf_sparse_eff, t, 'tf')
    # TG effectie
    gs.plot_out_deg_eff(out_deg_tg_dense_eff, out_deg_tg_sparse_eff, t, 'tg')
    
    ############## Look at 3 node feedback stuff ##################
    motif_df, counts_df, graphs_by_age = gs.analyze_3_node_feedback(theta_tf_sp, theta_tf_sparse, nTF, niters=50, names_tf = names_tf, age_sparse=age_tf_sparse, age_dense=age_tf_dense)

    motif_df.to_csv(f'3_node_feedback_df_{t}.csv', index=False)
    counts_df.to_csv(f'3_node_feedback_counts_df_{t}.csv', index=False)

    
    # now plot them
    gs.plot_3_node_feedback(motif_df, counts_df, graphs_by_age, t)
    
    ################### IFFL ######################
    motif_df, counts_df, graphs_by_age = gs.analyze_IFFL(theta_tf_sp, theta_tf_sparse, nTF, niters=50, names_tf = names_tf, age_sparse=age_tf_sparse, age_dense=age_tf_dense)
    # now plot them
    motif_df.to_csv(f'IFFL_df_{t}.csv', index=False)
    counts_df.to_csv(f'IFFL_counts_df_{t}.csv', index=False)
    
    gs.plot_IFFL(motif_df, counts_df, graphs_by_age, t)
    
    ############## Look at 2 node feedback stuff ##################
    
    motif_df, counts_df, graphs_by_age = gs.analyze_struct_2nodes(theta_tf_sp, theta_tf_sparse, nTF, niters=50, names_tf = names_tf, age_sparse=age_tf_sparse, age_dense=age_tf_dense)

    motif_df.to_csv(f'2_node_feedback_df_{t}.csv', index=False)
    counts_df.to_csv(f'2_node_feedback_counts_df_{t}.csv', index=False)
    
    gs.plot_2nodes(motif_df, counts_df, graphs_by_age, t)
    '''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run analysis with tissue and age inputs.")
    parser.add_argument('--t', type=str, required=True, help='Name of the tissue')
    parser.add_argument('--lam3', type=str, required=True, help='Best lambda 3m')
    parser.add_argument('--lam24', type=str, required=True, help='Best lambda 24m')

    args = parser.parse_args()
    
    main(args.t, args.lam3, args.lam24)