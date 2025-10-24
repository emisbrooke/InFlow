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
#ki = knock_in(theta, tf_samples, m, tf_idx, int_burn=10000, nSamples=1000, int_save=100)

def main(t, lam, start, stop):
    age = 24
    factor = 0.01

    cMuscle24 = np.load('data/cell_type_idxs_Limb_Muscle_24m_skeletal muscle satellite cell.npy')
    cMuscle3 = np.load('data/cell_type_idxs_Limb_Muscle_3m_skeletal muscle satellite cell.npy')
    print(f'Running for {t}, lam = {lam}')


    age=3
    gene_data3 = np.load(f'data/{t}_tg_data_binary_{age}m_filt.npy')[:, cMuscle3[:,0]]
    tf_data3 = np.load(f'data/{t}_tf_data_binary_{age}m_filt.npy')[:, cMuscle3[:,0]]

    age = 24
    gene_data24 = np.load(f'data/{t}_tg_data_binary_{age}m_filt.npy')[:, cMuscle24[:,0]]
    tf_data24 = np.load(f'data/{t}_tf_data_binary_{age}m_filt.npy')[:, cMuscle24[:,0]]

    best_n_tf = 30
    L_max_tf= float('-inf')
    best_ntf = 11
    nHoSets = 8

    directory_path = f'../project/Aging/output_tf_{t}/{age}m/ho/'
    for n in range(1,nHoSets):
        print(f'--------------------ON HO set {n}----------------------------')
        for i in range(10):
            with h5py.File(f'{directory_path}/model_dict_tf_age_{age}m_iter{i}_lambda{lam}_with_ho_10p_set{n}.h5', 'r') as h5f:
                pi_tf = h5f['pi'][:]
                Larr = h5f['Larr'][:]
                if Larr[-1]>L_max_tf:
                    L_max_tf = Larr[-1]
                    best_ntf = i
                        
        with h5py.File(f'../project/Aging/output_tf_{t}/{age}m/ho/model_dict_tf_age_{age}m_iter{best_ntf}_lambda{lam}_with_ho_10p_set{n}.h5', 'r') as h5f:
            pi_tf24 = h5f['pi'][:]
            theta_tf24 = h5f['theta'][:]
            m_tf24 = h5f['m'][:]


        theta_tf24[np.abs(theta_tf24) < factor] = 0
        nTF = tf_data24.shape[0]
        
        if start > nTF:
            print('Start should be less than the number of TFs')
            return
        
        if stop > nTF:
            stop = nTF

        sp24_tf = np.where(theta_tf24 != 0)[0].shape[0]/theta_tf24.flatten().shape[0]
    
        tf_data3_mean = tf_data3.mean(axis=1)
        L1_arr = np.zeros(stop-start)
        count=0
        tf_samples = tf_data24 #np.load(f'{directory_path}/samples_24m_200000_lam_{lam}_factor_{factor}_ho_set{n}.npy')    

        for i in range(start,stop):
            print(f'on tf {i}')
            
            if os.path.exists(f'ki_samples/samples_ki_tf_{i}_24m_Limb_Muscle_fact_{factor}_ho_set{n}_skeletal_satellite_cell.npy'):
                print(f'already ran ki_samples/samples_ki_tf_{i}_24m_Limb_Muscle_fact_{factor}_ho_set{n}.npy')
                continue
    
            plt.figure(i)
            plt.scatter(tf_data3.mean(axis=1), tf_data24.mean(axis=1), s=6, label = f'24m', c='black')
            tf_idx = i #random.randint(0,tf_samples.shape[0])
            ki_samples = ki.knock_in(theta_tf24, tf_samples, m_tf24, tf_idx, int_burn=80000, nSamples=2000, int_save=100)
            ki_samp_mean = ki_samples.mean(axis=1)
            l1 = np.sum(np.abs(ki_samp_mean.cpu().numpy()-tf_data3_mean))
            L1_arr[count] = l1
            plt.scatter(tf_data3.mean(axis=1), ki_samples.cpu().numpy().mean(axis=1), s=6, label = f'top {i} TF')
            #np.save(f'ki_samples/samples_ki_tf_{i}_{t}_fact_{factor}_ho_set{n}',ki_samples.cpu().numpy())
            np.save(f'ki_samples/samples_ki_tf_{i}_{t}_fact_{factor}_ho_set{n}_skeletal_satellite_cell',ki_samples.cpu().numpy())
            
            plt.title('KI mean activity')
            plt.xlabel('3m')
            plt.ylabel('KI 24m')
            count+=1
            plt.legend()
            plt.savefig(f'ki_figs/tf_{i}_{t}_mean_act_ho_set{n}_sk_satt_cell')
            plt.close()
        np.save(f'ki_samples/{t}_L1_tfs_ho_set{n}_skeletal_satellite_cell_{start}_{stop}', L1_arr)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run analysis with tissue and age inputs.")
    parser.add_argument('--start', type=int, required=True, help='start of tf')
    parser.add_argument('--stop', type=int, required=True, help='stop of tf')
    
    args = parser.parse_args()
    
    main('Limb_Muscle', 3, int(args.start), int(args.stop))