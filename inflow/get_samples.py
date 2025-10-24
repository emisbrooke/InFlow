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
print = functools.partial(print, flush=True)
@torch.jit.script
def update_samples(theta, tf_samples, m, row_idxs) -> torch.Tensor:
    #print(theta.is_floating_point(), tf_samples.is_floating_point(), m.is_floating_point())
    why = theta[0,0].item()
    #print(why, theta.shape)
    exp = torch.mm(theta.transpose(0, 1), tf_samples.to(torch.float)) + m.unsqueeze(1)
    pi = torch.sigmoid(-exp)
    #print(pi.is_floating_point())
    chain_arr = torch.arange(len(row_idxs), device=tf_samples.device)
    chosen_pi = pi[row_idxs, chain_arr]
    #print(chosen_pi.is_floating_point())
    
    new_spins = torch.bernoulli(chosen_pi)
    #print(tf_samples[row_idxs, chain_arr].is_floating_point(), new_spins.is_floating_point())
    new_spins = new_spins.to(tf_samples.dtype)
    tf_samples[row_idxs, chain_arr] = new_spins
    #print('shape', tf_samples.shape)
    # Use in-place update to avoid cloning
    #print(new_spins.shape, row_idxs.shape, tf_samples.shape)
    #tf_samples.index_copy_(0, torch.stack((row_idxs, chain_arr), dim=1), new_spins)
    return tf_samples
def generate_samples_mc(theta, tf_samples, m, int_burn=10000, nSamples=1000, int_save=1000, add=False):
    nTF, nChain = tf_samples.shape
    #print('nTF', nTF)
    if nSamples < nChain:
        nSamples = nChain
    stored_samples = []
    if add == True:  
        int_burn = int_save
        #print('We are adding, iters is', iters)
    iters = (nSamples // nChain) * int_save + (int_burn - int_save)
    #print('iters is ', iters, int_burn, int_save, nSamples)

    #print('device is', tf_samples.device)

    # Pre-generate all row indices for updating
    row_idx_all = torch.randint(nTF, (iters+1, nChain), device=tf_samples.device)
    #print('shape is', tf_samples.shape, row_idx_all)
    count = 0 
    for i in range(0, iters+1, 1000):  # Update in batches of 100
        row_idx_batch = row_idx_all[i:i+1000]  # Select batch of indices
        print(f'on batch {i}')
        for row_idxs in row_idx_batch:
            tf_samples = update_samples(theta, tf_samples, m, row_idxs)
            if ((count >= int_burn) and (count % int_save == 0)) or (count==int_burn):
                stored_samples.append(tf_samples.clone().cpu())
            count +=1
    
    samples = torch.hstack(stored_samples)
    samples = samples.to(tf_samples.device)

    del tf_samples
    torch.cuda.empty_cache()

    # samples shape: [nTF, total_samples]
    #print('Shared sample bank shape is', samples.shape)
    return samples

def main(t, age, lam):
    print(f"Tissue: {t}, Age: {age}")
    
    # your a#for gene_d, tf_d, a in zip(gene_data, tf_data, ages):
    iters = 8
    factor = .01
    print(f'running for {age}m age')

    gene_data = np.load(f'data/{t}_tg_data_binary_{age}m_filt.npy')
    tf_data = np.load(f'data/{t}_tf_data_binary_{age}m_filt.npy')

    tf_data = tf_data.astype(np.float64)
    gene_data = gene_data.astype(np.float64)

    nTF, nC = tf_data.shape
    nG, nC = gene_data.shape
    print('shape is', nTF, nC, nG, nC)

    directory_path = f'../project/Aging/output_tf_{t}/{age}m/ho/'

    
    os.makedirs(directory_path, exist_ok=True)
    nHoSets = 8
    for n in range(nHoSets):

        if os.path.exists(f'{directory_path}/samples_{age}m_40000_lam_{lam}_factor_{factor}_set{n}.npy'):
            print(f'already ran {directory_path}/samples_{age}m_40000_lam_{lam}_factor_{factor}_set{n}.npy')
            continue

        print('----------------------------------------------------------------------')
        print(f'N = {n}')
        print('----------------------------------------------------------------------')

        nTF, nC = tf_data.shape
        L_max_tf = float('-inf')
        best_ntf = 13
        i=1

        if not os.path.exists(f'{directory_path}/model_dict_tf_age_{age}m_iter7_lambda{lam}_with_ho_10p_set{n}.h5'):
            print(f'dont have {directory_path}/model_dict_tf_age_{age}m_iter7_lambda{lam}_with_ho_10p_set{n}.h5')
            continue
            
        with h5py.File(f'{directory_path}/model_dict_tf_age_{age}m_iter{i}_lambda{lam}_with_ho_10p_set{n}.h5', 'r') as h5f:
            theta = h5f['theta'][:]
            theta_where = np.where(np.abs(theta).flatten()>=factor)[0]
            theta_flat_new = theta.copy().flatten()
            
            #theta_flat_new[theta_flat_new<1e-3] = 0
            theta[np.abs(theta)<factor] = 0 #theta_flat_new.reshape(theta.shape)
            check = np.where(np.abs(theta).flatten()!=0)[0]
            
        for i in range(iters):
            with h5py.File(f'{directory_path}/model_dict_tf_age_{age}m_iter{i}_lambda{lam}_with_ho_10p_set{n}.h5', 'r') as h5f:
                Larr_tf = h5f['Larr'][:]
                if Larr_tf[-1] > L_max_tf:
                    print(f'Iteration {i}: Larr_tf[-1]={Larr_tf[-1]} > {L_max_tf}')
                    L_max_tf = Larr_tf[-1]
                    best_ntf = i
                pi = h5f['pi'][:]
                theta = h5f['theta'][:]
                m = h5f['m'][:]
                theta_where = np.where(np.abs(theta).flatten()>=factor)[0]
                theta_flat_new = theta.copy().flatten()
                
                #theta_flat_new[theta_flat_new<1e-3] = 0
                theta[np.abs(theta)<factor] = 0 #theta_flat_new.reshape(theta.shape)
                check2 = np.where(np.abs(theta).flatten()!=0)[0]
                check = np.intersect1d(check, check2)
        np.save(f'{directory_path}/intersection_theta_lambda{lam}_{age}m_factor_{factor}_set{n}', check)   


        print('sparse shape is', check.shape)
        with h5py.File(f'{directory_path}/model_dict_tf_age_{age}m_iter{best_ntf}_lambda{lam}_with_ho_10p_set{n}.h5', 'r') as h5f:
            pi = h5f['pi'][:]
            theta = h5f['theta'][:]
            m = h5f['m'][:]
        pi = torch.tensor(pi)#.to('cuda')
        theta = torch.tensor(theta)#.to('cuda')
        theta_argsort = torch.sort(theta.flatten())[1]
        theta_flat = theta.clone().flatten()
        theta_flat[torch.abs(theta.flatten())<factor] = 0
        theta[torch.abs(theta)<factor] = 0 # = theta_flat.reshape(theta.shape)

        theta = theta.to(torch.float).to('cuda')
        m = torch.tensor(m).to('cuda').to(torch.float)
        tf_data_cuda = torch.tensor(tf_data.astype(float)).to('cuda')
        print(type(theta[0,0].item()), type(tf_data_cuda[0,0].item()), type(m[0].item()))
        print(theta.shape)


        samples_tf_3m_40000 = generate_samples_mc(theta, tf_data_cuda, m, int_burn=40000, nSamples=nC*5, int_save=1000, add=False)
        samples_tf_3m_80000 = generate_samples_mc(theta, tf_data_cuda, m, int_burn=80000, nSamples=nC*5, int_save=1000, add=False)
        samples_tf_3m_200000 = generate_samples_mc(theta, tf_data_cuda, m, int_burn=200000, nSamples=nC*5, int_save=1000, add=False)
        #samples_tf_3m_400000 = generate_samples_mc(theta, tf_data_cuda, m, int_burn=400000, nSamples=nC*5, int_save=1000, add=False)

        np.save(f'{directory_path}/samples_{age}m_40000_lam_{lam}_factor_{factor}_set{n}', samples_tf_3m_40000.cpu().numpy())
        np.save(f'{directory_path}/samples_{age}m_80000_lam_{lam}_factor_{factor}_set{n}', samples_tf_3m_80000.cpu().numpy())
        np.save(f'{directory_path}/samples_{age}m_200000_lam_{lam}_factor_{factor}_set{n}', samples_tf_3m_200000.cpu().numpy())
        #np.save(f'{directory_path}/samples_{age}m_400000_lam_{lam}_factor_{factor}', samples_tf_3m_400000.cpu().numpy())



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run analysis with tissue and age inputs.")
    parser.add_argument('--tissue', type=str, required=True, help='Name of the tissue')
    parser.add_argument('--age', type=str, required=True, help='Age group (e.g., young, old)')
    parser.add_argument('--lam', type=int, required=True, help='lamda value (e.g. 1, 3)')
    
    args = parser.parse_args()
    
    main(args.tissue, args.age, args.lam)