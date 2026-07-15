import argparse
from importlib import reload
import numpy as np
import torch
import sys
import h5py
import os
import functools
from tqdm import tqdm
print = functools.partial(print, flush=True)

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
    #tf_samples.index_copy_(0, torch.stack((row_idxs, chain_arr), dim=1), new_spins)
    return tf_samples

def generate_samples_mc(theta, tf_samples, m, tf_idx, int_burn=10000, nSamples=1000, int_save=1000, add=False):
    nTF, nChain = tf_samples.shape


    tf_samples[tf_idx] = 1

    #print('nTF', nTF)
    if nSamples < nChain:
        nSamples = nChain
    stored_samples = []
    if add == True:  
        int_burn = int_save
        #print('We are adding, iters is', iters)
    iters = (nSamples // nChain) * int_save + (int_burn - int_save)
    row_idx_all = torch.randint(nTF, (iters+1, nChain), device=tf_samples.device)

    #print(torch.any(row_idx_all == tf_idx))
    while(torch.any(row_idx_all==tf_idx)):
        row_idx_all[torch.where(row_idx_all == tf_idx)] = torch.randint(nTF, (torch.where(row_idx_all == tf_idx)[0].shape[0],), device=tf_samples.device)
        
   
    #print('shape is', tf_samples.shape, row_idx_all.shape)
    count = 0 
    for i in range(0, iters+1, 1000):  # Update in batches of 100
        row_idx_batch = row_idx_all[i:i+1000]  # Select batch of indices
        #print(f'on batch {i}')
        #if i % 40000 == 0:
        #    print(f'step {i}')
        for row_idxs in row_idx_batch:
            if np.any(row_idxs.cpu().numpy() == tf_idx):
                print('yikesies')
            tf_samples = update_samples(theta, tf_samples, m, row_idxs)
            if ((count >= int_burn) and (count % int_save == 0)) or (count==int_burn):
                stored_samples.append(tf_samples.clone().cpu())
            count +=1
    
    samples = torch.hstack(stored_samples)
    samples = samples.to(tf_samples.device)

    if torch.any(samples[tf_idx] != 1):
        print("YOU FAILED")

    del tf_samples
    torch.cuda.empty_cache()

    # samples shape: [nTF, total_samples]
    #print('Shared sample bank shape is', samples.shape)
    return samples


def generate_samples_mc_many(theta, tf_samples, m, tf_idxs_ki=[], tf_idxs_ko=[], int_burn=10000, nSamples=1000, int_save=1000, add=False):
    nTF, nChain = tf_samples.shape

    if not list(tf_idxs_ki):
        return generate_samples_mc_many_ko(theta, tf_samples, m, tf_idxs_ko, int_burn, nSamples, int_save, add)
    if not list(tf_idxs_ko):
        return generate_samples_mc_many_ki(theta, tf_samples, m, tf_idxs_ki, int_burn, nSamples, int_save, add)

    tf_idxs_ki = tf_idxs_ki.copy()
    tf_idxs_ko = tf_idxs_ko.copy()

    tf_samples[tf_idxs_ki] = 1
    tf_samples[tf_idxs_ko] = 0

    tf_idxs = np.append(tf_idxs_ki, tf_idxs_ko)

    #print('nTF', nTF)
    if nSamples < nChain:
        nSamples = nChain
    stored_samples = []
    if add == True:  
        int_burn = int_save
        #print('We are adding, iters is', iters)
    iters = (nSamples // nChain) * int_save + (int_burn - int_save)

    all_rows   = np.arange(nTF)
    valid_rows = np.setdiff1d(all_rows, tf_idxs, assume_unique=True)
    
    row_idx_all = np.random.choice(valid_rows, size=(iters+1, nChain), replace=True)
    row_idx_all = torch.from_numpy(row_idx_all)

    print('shape is', tf_samples.shape, row_idx_all.shape)
    count = 0 
    for i in range(0, iters+1, 500):  # Update in batches of 100
        row_idx_batch = row_idx_all[i:i+500]  # Select batch of indices
        if i%5000 == 0:
            print(f'on batch {i} out of {iters}')
        for row_idxs in row_idx_batch:                    
            tf_samples = update_samples(theta, tf_samples, m, row_idxs)
            if ((count >= int_burn) and (count % int_save == 0)) or (count==int_burn):
                stored_samples.append(tf_samples.clone().cpu())
            count +=1
    
    samples = torch.hstack(stored_samples)
    samples = samples.to(tf_samples.device)

    if torch.any(samples[tf_idxs_ko] != 0):
        print("YOU FAILED KO")
        if torch.any(samples[tf_idxs_ki] != 1):
            print("YOU FAILED KI and KO")
    if torch.any(samples[tf_idxs_ki] != 1):
        print("YOU FAILED KI")

    del tf_samples
    torch.cuda.empty_cache()

    # samples shape: [nTF, total_samples]
    #print('Shared sample bank shape is', samples.shape)
    return samples


def generate_samples_mc_many_ki(theta, tf_samples, m, tf_idxs, int_burn=10000, nSamples=1000, int_save=1000, add=False):
    nTF, nChain = tf_samples.shape
    tf_idxs = tf_idxs.copy()
    tf_samples[tf_idxs] = 1

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
    all_rows   = np.arange(nTF)
    valid_rows = np.setdiff1d(all_rows, tf_idxs, assume_unique=True)

    row_idx_all = np.random.choice(valid_rows, size=(iters+1, nChain), replace=True)
    row_idx_all = torch.from_numpy(row_idx_all)
   
    print('shape is', tf_samples.shape, row_idx_all.shape)
    count = 0 
    for i in range(0, iters+1, 500):  # Update in batches of 100
        row_idx_batch = row_idx_all[i:i+500]  # Select batch of indices
        if i%5000 == 0:
            print(f'on batch {i} out of {iters}')
        for row_idxs in row_idx_batch:
            for i in range(len(tf_idxs)):    
                if torch.any(row_idxs == tf_idxs[i]):
                    print('yikesies')
                    
            tf_samples = update_samples(theta, tf_samples, m, row_idxs)
            if ((count >= int_burn) and (count % int_save == 0)) or (count==int_burn):
                stored_samples.append(tf_samples.clone().cpu())
            count +=1
    
    samples = torch.hstack(stored_samples)
    samples = samples.to(tf_samples.device)

    if torch.any(samples[tf_idxs] != 1):
        print("YOU FAILED")

    del tf_samples
    torch.cuda.empty_cache()

    # samples shape: [nTF, total_samples]
    #print('Shared sample bank shape is', samples.shape)
    return samples


def generate_samples_mc_many_ko(theta, tf_samples, m, tf_idxs, int_burn=10000, nSamples=1000, int_save=1000, add=False):
    nTF, nChain = tf_samples.shape
    tf_idxs = tf_idxs.copy()
    tf_samples[tf_idxs] = 0

    #print('nTF', nTF)
    if nSamples < nChain:
        nSamples = nChain
    stored_samples = []
    if add == True:  
        int_burn = int_save
        #print('We are adding, iters is', iters)
    iters = (nSamples // nChain) * int_save + (int_burn - int_save)

    all_rows   = np.arange(nTF)
    valid_rows = np.setdiff1d(all_rows, tf_idxs, assume_unique=True)
    
    row_idx_all = np.random.choice(valid_rows, size=(iters+1, nChain), replace=True)
    row_idx_all = torch.from_numpy(row_idx_all)

    print('shape is', tf_samples.shape, row_idx_all.shape)
    count = 0 
    for i in range(0, iters+1, 500):  # Update in batches of 100
        row_idx_batch = row_idx_all[i:i+500]  # Select batch of indices
        if i%5000 == 0:
            print(f'on batch {i} out of {iters}')
        for row_idxs in row_idx_batch:
            for i in range(len(tf_idxs)):    
                if torch.any(row_idxs == tf_idxs[i]):
                    print('yikesies')
                    
            tf_samples = update_samples(theta, tf_samples, m, row_idxs)
            if ((count >= int_burn) and (count % int_save == 0)) or (count==int_burn):
                stored_samples.append(tf_samples.clone().cpu())
            count +=1
    
    samples = torch.hstack(stored_samples)
    samples = samples.to(tf_samples.device)

    if torch.any(samples[tf_idxs] != 0):
        print("YOU FAILED")

    del tf_samples
    torch.cuda.empty_cache()

    # samples shape: [nTF, total_samples]
    #print('Shared sample bank shape is', samples.shape)
    return samples

##############################
####### Knock In #########
##############################

def knock_in(theta, tf_samples, m, tf_idx, int_burn=10000, nSamples=1000, int_save=100):
    theta = torch.tensor(theta).to(torch.float).cuda()
    tf_samples = torch.tensor(tf_samples).to(torch.float).cuda()
    m = torch.tensor(m).to(torch.float).cuda()

    samples = generate_samples_mc(theta, tf_samples, m, tf_idx, int_burn=int_burn, nSamples=nSamples, int_save=int_save)
    
    if(np.any(samples.cpu().numpy()[tf_idx]==0)):
        print('BOOOOO')
        return 1

    return samples


def knock_in_many(theta, tf_samples, m, tf_idxs, int_burn=10000, nSamples=1000, int_save=100):
    theta = torch.tensor(theta).to(torch.float).cuda()
    tf_samples = torch.tensor(tf_samples).to(torch.float).cuda()
    m = torch.tensor(m).to(torch.float).cuda()

    samples = generate_samples_mc_many(theta, tf_samples, m, tf_idxs, int_burn=int_burn, nSamples=nSamples, int_save=int_save)
    
    if(np.any(samples.cpu().numpy()[tf_idxs]==0)):
        print('BOOOOO')
        return 1

    return samples

######################################################################
#######B KNOCK-OUT
####

def knock_out(theta, tf_samples, m, tf_idx, int_burn=10000, nSamples=1000, int_save=100):
    theta = torch.tensor(theta).to(torch.float).cuda()
    tf_samples = torch.tensor(tf_samples).to(torch.float).cuda()
    m = torch.tensor(m).to(torch.float).cuda()

    samples = generate_samples_mc_ko(theta, tf_samples, m, tf_idx, int_burn=int_burn, nSamples=nSamples, int_save=int_save)
    
    if(np.any(samples.cpu().numpy()[tf_idx]==1)):
        print('BOOOOO')
        return 1

    return samples

def knock_out_many(theta, tf_samples, m, tf_idxs, int_burn=10000, nSamples=1000, int_save=100):
    theta = torch.tensor(theta).to(torch.float).cuda()
    tf_samples = torch.tensor(tf_samples).to(torch.float).cuda()
    m = torch.tensor(m).to(torch.float).cuda()

    samples = generate_samples_mc_many_ko(theta, tf_samples, m, tf_idxs, int_burn=int_burn, nSamples=nSamples, int_save=int_save)
    
    if(np.any(samples.cpu().numpy()[tf_idxs]==1)):
        print('BOOOOO')
        return 1

    return samples


def knock_both_many(theta, tf_samples, m, tf_idxs_ki=[], tf_idxs_ko=[], int_burn=10000, nSamples=1000, int_save=100):
    theta = torch.tensor(theta).to(torch.float).cuda()
    tf_samples = torch.tensor(tf_samples).to(torch.float).cuda()
    m = torch.tensor(m).to(torch.float).cuda()

    samples = generate_samples_mc_many(theta, tf_samples, m, tf_idxs_ki, tf_idxs_ko, int_burn=int_burn, nSamples=nSamples, int_save=int_save)
    
    if(np.any(samples.cpu().numpy()[tf_idxs_ki]==0)):
        print('BOOOOO on ki')
        if(np.any(samples.cpu().numpy()[tf_idxs_ko]==1)):
            print('BOOOOO on ko too')
        return 1

    if(np.any(samples.cpu().numpy()[tf_idxs_ko]==1)):
        print('BOOOOO on ko')
        return 0

    return samples


