import torch
import numpy as np
import time
import knock_in_final as ki

def print_memory_usage(msg=""):
    print(f"\n[MEMORY CHECK] {msg}")
    if torch.cuda.is_available():
        print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB\n")
    else:
        print("CUDA not available.\n")


@torch.jit.script
# These are functions to generate tf distributions to approximate the sum over tf configurations
def update_samples(theta, tf_samples, m, row_idxs) -> torch.Tensor:
    '''
    A function to update samples at each iteration in MC process
    Input:
        theta: GRN matrix nTF x nTF
        tf_samples: the current tf configuration
        m: parameter that is tf specific bias
        row_idxs: pre chosen indices to update at each step. One number per chain
    Output: 
        tf_samples: updated samples 
    * Note, a common issue in tensor multiplication is that all tensors must have same type
        i.e. floating point or double *
    '''
    #print(theta.is_floating_point(), tf_samples.is_floating_point(), m.is_floating_point())
    exp = torch.mm(theta.transpose(0, 1), tf_samples) + m.unsqueeze(1)
    pi = torch.sigmoid(-exp)
    chain_arr = torch.arange(len(row_idxs), device=tf_samples.device)
    chosen_pi = pi[row_idxs, chain_arr]
    
    new_spins = torch.bernoulli(chosen_pi)
    new_spins = new_spins.to(tf_samples.dtype)
    tf_samples[row_idxs, chain_arr] = new_spins
    return tf_samples

def generate_samples_mc(
    theta,
    tf_samples,
    m,
    int_burn=10000,
    nSamples=1000,
    int_save=1000,
    add=False,
    batch_size=500,
    verbose=True,
):
    '''
    Function to generate samples through mc sampling. 
    Input:
        theta: GRN matrix
        tf_samples: starting chains (tf configurations)
        m: TF-specific bias term
        int_burn: number of iterations to wait until sampling begins, default=10000
        nSamples: number of samples to have at the end, default=1000
        int_save: after burn in, how often to draw samples from the chain
        add: if add=True, there is no burn-in. It is assumed given samples are from a previous mc process, default=False

    Output:
        samples: tf samples generated from mc process
    '''
    nTF, nChain = tf_samples.shape # each sample is treated as a starting chain

    if nSamples < nChain: # return a minimum of nChain samples
        nSamples = nChain

    stored_samples = [] # empty array to add samples
    if add == True:  
        print('Starting for pre-burned samples')
        int_burn = int_save

    n_collect = int(np.ceil(nSamples / nChain))
    iters = n_collect * int_save + (int_burn - int_save) # calculated the number of needed steps

    # Pre-generate all row indices for updating
    row_idx_all = torch.randint(nTF, (iters+1, nChain), device=tf_samples.device) # This just saves some time to pre-do it

    count = 0 
    t0 = time.perf_counter()
    for i in range(0, iters+1, batch_size):
        row_idx_batch = row_idx_all[i:i+batch_size]  # Select batch of indices
        if verbose and i % max(batch_size * 10, 1) == 0:
            dt = time.perf_counter() - t0
            print(f'on batch i {i} for {iters} iters (stored={len(stored_samples)}, {dt:.1f}s)')

        for row_idxs in row_idx_batch:
            tf_samples = update_samples(theta, tf_samples, m, row_idxs)
            if count >= int_burn and (count - int_burn) % int_save == 0:
                stored_samples.append(tf_samples.clone().cpu())
            count +=1

    if not stored_samples:
        raise RuntimeError("No samples were stored. Check int_burn/int_save/nSamples settings.")

    samples = torch.hstack(stored_samples) # turn into tensor
    samples = samples[:, :nSamples]
    samples = samples.to(tf_samples.device)

    del tf_samples # saves space
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return samples
def calc_entropy(p):
    '''
    function to calculate entropy
    Input:
        p: probability
    Output:
        shannon entropy
    '''
    return -p * torch.log2(p) - (1 - p) * torch.log2(1 - p)

def calc_MI(
    sigma_tf,
    theta_tf,
    m_tf,
    sigma_tg,
    theta_tg,
    m_tg,
    age_tf,
    age_tg,
    t,
    lam=0,
    n_tf_samples=10000,
    use_gpu_if_avail=True,
    samples_tf=None,
    int_burn=200000,
    int_save=500,
    mc_batch_size=500,
    verbose=True,
):
    '''
    This is a function that calculates MI between all tfs and a TG
    Input:
        sigma_tf: starting chains (configurations) for tf calculation
        theta_tf: TF-TF GRN
        m_tf: TF specific bias term
        sigma_tg: TG data
        theta_tf: TF-TG GRN
        m_tg: TG specific bias term
        age_tf: age for TF stuff (theta, sigma, m)->for file names
        age_tg: age for TG stuff (theta, sigma, m)->for file names
        lam: lambda used for lasso->again for file name
        n_tf_samples: number of tf configurations to get MI, default=10000
        use_gpu_if_avail: when True, use any available GPU, default=True
        samples_tf: if not none, these configurations are used to calculate MI, if none
                    samples are generated, default=None
    Output:
        MI: mutual information for each TG, vector of length TG
        entropy: entropy for each TG, vector of length TG
    '''
    
    _, nC = sigma_tg.shape
    device = torch.device("cuda:0" if (torch.cuda.is_available() and use_gpu_if_avail) else "cpu")
    if verbose:
        print(f'device is {device}')
    
    theta_tf = torch.as_tensor(theta_tf, dtype=torch.float, device=device)
    theta_tg = torch.as_tensor(theta_tg, dtype=torch.float, device=device)

    m_tf = torch.as_tensor(m_tf, dtype=torch.float, device=device)
    m_tg = torch.as_tensor(m_tg, dtype=torch.float, device=device)

    tf_samples = torch.as_tensor(sigma_tf, dtype=torch.float, device=device)
    if verbose:
        print_memory_usage("After tensor allocation") # checking memory issues when using gpu

    if samples_tf is None:
        if verbose:
            print('generating....')
        samples = generate_samples_mc(
            theta_tf,
            tf_samples,
            m_tf,
            int_burn=int_burn,
            nSamples=n_tf_samples,
            int_save=int_save,
            add=False,
            batch_size=mc_batch_size,
            verbose=verbose,
        )
        n_tf_samples = samples.shape[1]
        #np.save(f'samples/samples_MI_{t}_lam_{lam}_{age_tf}_in_{age_tg}_fact_p01', samples.cpu().numpy())
        
    else:
        samples = torch.as_tensor(samples_tf, dtype=torch.float, device=device)
        n_tf_samples = samples_tf.shape[1]
        if verbose:
            print('samples were provided')
  
    # This matrix multipication allow us to calculate the MI for each TG in parallel to shorten runtime by ALOT
    exp = torch.mm(theta_tg.transpose(0,1), samples)+ m_tg.unsqueeze(1)
    
    # these are the conditional probs
    # The i,jth element of pi1/0 is the P(gi|tj) where tj is a configuration of tfs mat is nTG x n_tf_samples
    pi1 = torch.sigmoid(-exp)
    pi0 = 1-pi1
    
    # pn_tot = p(g=n), marginalized prob
    p1_tot_raw = pi1.mean(axis=1) # Should be n_genes x 1, where p_1_tot[i] is p(gi): average p(g) over chains
    p0_tot_raw = 1 - p1_tot_raw
    
    entropy = torch.zeros_like(p1_tot_raw)
    valid_entropy = (p1_tot_raw > 0) & (p1_tot_raw < 1)
    entropy[valid_entropy] = calc_entropy(p1_tot_raw[valid_entropy])
    eps = 1e-8
    p1_tot = torch.clamp(p1_tot_raw, min=eps, max=1-eps)
    p0_tot = torch.clamp(p0_tot_raw, min=eps, max=1-eps)
    
    # MI for each configuration of a TG (0 or 1)
    term1 = pi1 * torch.log2((pi1 + eps)/p1_tot.unsqueeze(1))
    term2 = pi0 * torch.log2((pi0 + eps)/p0_tot.unsqueeze(1))
    tot = term1 + term2
    MI = tot.sum(axis=1)/tot.shape[1]

    # Clear memory
    del pi1
    del pi0
    del tot
    del tf_samples
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return MI, entropy
 
def calc_MI_ki(
    sigma_tf,
    theta_tf,
    m_tf,
    sigma_tg,
    theta_tg,
    m_tg,
    age_tf,
    age_tg,
    t,
    tf_idxs,
    lam=0,
    n_tf_samples=10000,
    use_gpu_if_avail=True,
    samples_tf=None,
    ki_int_burn=40000,
    ki_int_save=100,
    verbose=True,
):
    '''
    This is a function that calculates MI between all tfs and a TG
    Input:
        sigma_tf: starting chains (configurations) for tf calculation
        theta_tf: TF-TF GRN
        m_tf: TF specific bias term
        sigma_tg: TG data
        theta_tf: TF-TG GRN
        m_tg: TG specific bias term
        age_tf: age for TF stuff (theta, sigma, m)->for file names
        age_tg: age for TG stuff (theta, sigma, m)->for file names
        t: tissue -> for file writing
        tf_idxs: knock-in indices 
        lam: lambda used for lasso->again for file name
        n_tf_samples: number of tf configurations to get MI, default=10000
        use_gpu_if_avail: when True, use any available GPU, default=True
        samples_tf: if not none, these configurations are used to calculate MI, if none
                    samples are generated, default=None
    Output:
        MI: mutual information for each TG, vector of length TG
        entropy: entropy for each TG, vector of length TG
    '''
    
    device = torch.device("cuda:0" if (torch.cuda.is_available() and use_gpu_if_avail) else "cpu")
    if verbose:
        print(f'device is {device}')
    
    theta_tf = torch.as_tensor(theta_tf, dtype=torch.float, device=device)
    theta_tg = torch.as_tensor(theta_tg, dtype=torch.float, device=device)

    m_tf = torch.as_tensor(m_tf, dtype=torch.float, device=device)
    m_tg = torch.as_tensor(m_tg, dtype=torch.float, device=device)

    # These are what we use to train tg model
    tf_samples = torch.as_tensor(sigma_tf, dtype=torch.float, device=device)
    if verbose:
        print_memory_usage("After tensor allocation") # track memory usage for GPU

    if samples_tf is None:
        # Generate samples if none are given
        if verbose:
            print('generating....')
        # similar to calc_MI but uses ki function
        ki_samples = ki.knock_in_many(
            theta_tf,
            tf_samples,
            m_tf,
            tf_idxs,
            int_burn=ki_int_burn,
            nSamples=n_tf_samples,
            int_save=ki_int_save,
        )
        n_tf_samples = ki_samples.shape[1]
        samples = ki_samples
        if verbose:
            print('samples shape is', samples.shape)
        #np.save(f'../project/samples_MI_{t}_KI', samples.cpu().numpy()) # to save generated samples if you want
        
    else:
        samples = torch.as_tensor(samples_tf, dtype=torch.float, device=device)
        n_tf_samples = samples_tf.shape[1]
        if verbose:
            print('samples were provided')

    # This matrix multipication allow us to calculate the MI for each TG in parallel to shorten runtime by ALOT
    exp = torch.mm(theta_tg.transpose(0,1), samples)+ m_tg.unsqueeze(1)
    
    # these are the conditional probs
    # The i,jth element of pi1/0 is the P(gi|tj) where tj is a configuration of tfs mat is nTG x n_tf_samples
    pi1 = torch.sigmoid(-exp)
    pi0 = 1-pi1
    
    # pn_tot = p(g=n), marginalized prob
    p1_tot_raw = pi1.mean(axis=1) # Should be n_genes x 1, where p_1_tot[i] is p(gi): average p(g) over chains
    p0_tot_raw = 1 - p1_tot_raw
    
    entropy = torch.zeros_like(p1_tot_raw)
    valid_entropy = (p1_tot_raw > 0) & (p1_tot_raw < 1)
    entropy[valid_entropy] = calc_entropy(p1_tot_raw[valid_entropy])
    eps = 1e-8
    p1_tot = torch.clamp(p1_tot_raw, min=eps, max=1-eps)
    p0_tot = torch.clamp(p0_tot_raw, min=eps, max=1-eps)
    
    # MI for each configuration of a TG (0 or 1)
    term1 = pi1 * torch.log2((pi1 + eps)/p1_tot.unsqueeze(1))
    term2 = pi0 * torch.log2((pi0 + eps)/p0_tot.unsqueeze(1))
        
    tot = term1 + term2
    MI = tot.sum(axis=1)/tot.shape[1] # again averaging over number of samples. This takes care of the p(tbar)

    # Try and clear stuff out of memory when using GPU
    del pi1
    del pi0
    del tot
    del tf_samples
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return MI, entropy
    
    
        
