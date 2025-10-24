import knock_in_final as ki
import argparse
from importlib import reload
import numpy as np
import torch
import model_tf_jump as model
#import model_tg_no_sparsity as model
import sys
#import model_tg_1age as model
import h5py
import os
import functools
from importlib import reload
import random
import matplotlib.pyplot as plt
reload(ki)

def main(t, age, lam):
    #ki = knock_in(theta, tf_samples, m, tf_idx, int_burn=10000, nSamples=1000, int_save=100)
    print(f'On tissue {t}')

    factor = 0.01
    gene_data3 = np.load(f'data/{t}_tg_data_binary_3m_filt.npy')
    tf_data3 = np.load(f'data/{t}_tf_data_binary_3m_filt.npy')
    tf_data = np.load(f'data/{t}_tf_data_binary_{age}m_filt.npy')
    tf_data_mean = tf_data.mean(axis=1)

    gene_data24 = np.load(f'data/{t}_tg_data_binary_24m_filt.npy')
    tf_data24 = np.load(f'data/{t}_tf_data_binary_24m_filt.npy')
    
    nTF, nC24 = tf_data24.shape
    
    best_n_tf = 30
    L_max_tf= float('-inf')
    best_ntf = 11

    directory_path = f'../project/Aging/output_tg_{t}/{age}m/'
    directory_path_tf = f'../project/Aging/output_tf_{t}/{age}m/'
    
    print(lam)
    best_n_tf = 30
    L_max_tf= float('-inf')
    best_ntf = 11
    for i in range(10):
        if not os.path.exists(f'{directory_path_tf}/model_dict_tg_age_{age}m_iter{i}_lambda{lam}_no_ho.h5'):
            print(f'There is no thing for iter {i}')
            
            continue
    
        with h5py.File(f'{directory_path_tf}/model_dict_tg_age_{age}m_iter{i}_lambda{lam}_no_ho.h5', 'r') as h5f:
            Larr = h5f['Larr'][:]
            if Larr[-1]>L_max_tf:
                L_max_tf = Larr[-1]
                best_ntf = i
    with h5py.File(f'{directory_path_tf}/model_dict_tg_age_{age}m_iter{best_ntf}_lambda{lam}_no_ho.h5', 'r') as h5f:      
        pi_tf = h5f['pi'][:]
        theta_tf = h5f['theta'][:]
        m_tf = h5f['m'][:]
    
    theta_tf[np.abs(theta_tf) < factor] = 0
    
    sp_tf = np.where(theta_tf != 0)[0].shape[0]/theta_tf.flatten().shape[0]

    L1_arr = np.zeros(nTF)
    count=0
    niters=20
    print(f'age = {age}') 
    tf_samples = tf_data #np.load(f'{directory_path}/samples_24m_200000_lam_{lam}_factor_{factor}_ho_set{n}.npy')    
    for i in range(niters):
        print(f'on iter {i}')
        
        if os.path.exists(f'../project/Aging/stab_samples/samples_both_iter_{i}_{t}_fact_{factor}_no_ho_stab_{age}m'):
            print(f'already ran ./project/Aging/stab_samples/samples_both_iter_{i}_{t}_fact_{factor}_no_ho_stab_{age}m')
            continue
    
        plt.figure(i)
        plt.scatter(tf_data3.mean(axis=1), tf_data.mean(axis=1), s=6, label = f'Data', c='black')
        
        poss_idxs = np.arange(0,nTF)
        np.random.shuffle(poss_idxs)
        #print(poss_idxs)
        num_idxs = int(nTF*.1)
        tf_idxs = poss_idxs[:num_idxs]
        mask_low = tf_data_mean <= 0.5
        mask_high = tf_data_mean > 0.5

        print(f'min is {tf_data_mean.min()} and {tf_data_mean[tf_idxs].min()}')
        print(f'max is {tf_data_mean.max()} and {tf_data_mean[tf_idxs].max()}')
        print(len(tf_idxs))

        tf_idxs_ki = tf_idxs[np.where(mask_low[tf_idxs])]
        print(len(tf_idxs_ki))
        tf_idxs_ko = tf_idxs[np.where(mask_high[tf_idxs])]
        print(len(tf_idxs_ko))


        samples = ki.knock_both_many(theta_tf, tf_samples, m_tf, tf_idxs_ki=tf_idxs_ki, tf_idxs_ko=tf_idxs_ko, int_burn=40000, nSamples=2000, int_save=100)
        samp_mean = samples.cpu().numpy().mean(axis=1)
        l1 = np.sum(np.abs(samp_mean-tf_data_mean))
        L1_arr[count] = l1
        plt.scatter(tf_data3.mean(axis=1), samp_mean, s=6, label = f'KI')
        #np.save(f'ki_samples/samples_ki_tf_{i}_{t}_fact_{factor}_ho_set{n}',ki_samples.cpu().numpy())
        np.save(f'../project/Aging/stab_samples/samples_both_iter_{i}_{t}_fact_{factor}_no_ho_stab_{age}m',samples.cpu().numpy())
        count += 1
        plt.title('Changed mean activity')
        plt.xlabel('3m')
        plt.ylabel(f'{age}m')
        plt.legend()
        plt.savefig(f'stab_figs/rand_iter_{i}_{t}_mean_act_no_ho_stab_{age}m')
        plt.close()

    np.save(f'../project/Aging/stab_samples/{t}_L1_tfs_no_ho_stab_{age}m_both', L1_arr)
    print('made it')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run analysis with tissue and age inputs.")
    parser.add_argument('--t', type=str, required=True, help='Name of the tissue')
    parser.add_argument('--age', type=int, required=True, help='age, eg 3')
    parser.add_argument('--lam', type=int, required=True, help='lambda for age, e.g. 3')
    
    args = parser.parse_args()
    
    main(args.t, args.age, args.lam)