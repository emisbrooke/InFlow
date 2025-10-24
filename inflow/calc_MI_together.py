import torch
import numpy as np
from tqdm import tqdm
import knock_in_final as ki

def print_memory_usage(msg=""):
    print(f"\n[MEMORY CHECK] {msg}")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB\n")


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
    for i in range(0, iters+1, 500):  # Update in batches of 100
        row_idx_batch = row_idx_all[i:i+500]  # Select batch of indices
        if i % 5000 == 0:
            print(f'on batch i {i} for {iters} iters')
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
def calc_entropy(p):
    #if p == 0 or p == 1:
    #    return 0
    return -p * torch.log2(p) - (1 - p) * torch.log2(1 - p)

def calc_MI(sigma_tf, theta_tf, m_tf, sigma_tg, theta_tg, m_tg, age_tf, age_tg, t,  lam = 0, n_tf_samples = 10000, use_gpu_if_avail=True, samples_tf = None):
    nG, nC = sigma_tg.shape
    nTF, _ = sigma_tf.shape
    device = torch.device("cuda:0" if (torch.cuda.is_available() and use_gpu_if_avail) else "cpu")
    print(f'device is {device}')
    
    #if pi_tf.shape[0] != nTF:
    #    if pi_tf.shape[0] != (nTF + nG):
    #        print('ERRORRRRRRRR')
    #        return
    #    else:
    #        pi_tf = pi_tf[:nTF]
    
    theta_tf = torch.tensor(theta_tf).to(torch.float).to(device)
    theta_tg = torch.tensor(theta_tg).to(torch.float).to(device)

    m_tf = torch.tensor(m_tf).to(device).to(torch.float)
    m_tg = torch.tensor(m_tg).to(device).to(torch.float)

    # These are what we use to train tg model
    tf_samples = torch.tensor(sigma_tf).to(torch.float).to(device)
    print_memory_usage("After tensor allocation")

    print(f'tf_sigma shape is for real {tf_samples.shape}')

    int_save = 500
    int_burn = 200000
    count = 0
    N = int((int_save*n_tf_samples)/nC)
    stored_samples = []

    if samples_tf is None:
        print('generating....')
        samples = generate_samples_mc(theta_tf, tf_samples, m_tf, int_burn=int_burn, nSamples=n_tf_samples, int_save=int_save, add=False)
        n_tf_samples = samples.shape[1]
        print('samples shape is', samples.shape)
        #np.save(f'samples/samples_MI_{t}_lam_{lam}_{age_tf}_in_{age_tg}_fact_p01', samples.cpu().numpy())
        
    else:
        samples = torch.tensor(samples_tf).to(device)
        n_tf_samples = samples_tf.shape[1]
        #print('samples were provided')
    #print(theta_tg.shape, samples.shape, pi_tf.shape, m_tg.shape)
    samples = torch.tensor(samples).to(torch.float).to(device)
    
    exp = torch.mm(theta_tg.transpose(0,1), samples)+ m_tg.unsqueeze(1)
    pi1 = torch.exp(-exp)/ ( 1+torch.exp(-exp))
    pi0 = 1-pi1

    
    # The i,jth element of pi_tg is the P(gi|tj)

    p1_tot = pi1.sum(axis=1)/n_tf_samples # Should be n_genes x 1, where p_1_tot[i] is p(gi)
    p0_tot = pi0.sum(axis=1)/n_tf_samples # Should be n_genes x 1, where p_1_tot[i] is p(gi)
    
    #entropy = entropy.sum(axis=1)/n_tf_samples 
    entropy = calc_entropy(p1_tot)
    entropy[p1_tot==0] = 0
    entropy[p1_tot==1] = 0
    
    #print(f'p(g) shape is {p1_tot.shape}, {p0_tot.shape}')
    
    '''
    MI1 = torch.zeros(10).to(device)
    
    for gene in tqdm(range(10)):
        for config in range(n_tf_samples):
            term1 = pi_tg1[gene, config] * torch.log(pi_tg1[gene, config]/p1_tot[gene])
            term2 = pi_tg0[gene, config] * torch.log(pi_tg0[gene, config]/p0_tot[gene])
            MI1[gene] += term1 + term2
        MI1[gene]/=n_tf_samples
    '''
        
    term1 = (pi1 * torch.log2(pi1/p1_tot.unsqueeze(1))) 
    term2 = (pi0 * torch.log2(pi0/p0_tot.unsqueeze(1)))
    
    term1[pi1==0]=0
    term2[pi0==0]=0
        
    tot = term1 + term2
    #print('tot shape is', tot.shape)
    MI = tot.sum(axis=1)/tot.shape[1]
    del pi1
    del pi0
    del tot
    del tf_samples
    #print('finally' , torch.all(MI1 == MI2[:10]))
    
    return MI, entropy
 
def calc_MI_ki(sigma_tf, theta_tf, m_tf, sigma_tg, theta_tg, m_tg, age_tf, age_tg, t, tf_idxs, lam = 0, n_tf_samples = 10000, use_gpu_if_avail=True, samples_tf = None):
    nG, nC = sigma_tg.shape
    nTF, _ = sigma_tf.shape
    device = torch.device("cuda:0" if (torch.cuda.is_available() and use_gpu_if_avail) else "cpu")
    print(f'device is {device}')
    
    #if pi_tf.shape[0] != nTF:
    #    if pi_tf.shape[0] != (nTF + nG):
    #        print('ERRORRRRRRRR')
    #        return
    #    else:
    #        pi_tf = pi_tf[:nTF]
    
    theta_tf = torch.tensor(theta_tf).to(torch.float).to(device)
    theta_tg = torch.tensor(theta_tg).to(torch.float).to(device)

    m_tf = torch.tensor(m_tf).to(device).to(torch.float)
    m_tg = torch.tensor(m_tg).to(device).to(torch.float)

    # These are what we use to train tg model
    tf_samples = torch.tensor(sigma_tf).to(torch.float).to(device)
    print_memory_usage("After tensor allocation")

    print(f'tf_sigma shape is for real {tf_samples.shape}')

    int_save = 500
    int_burn = 200000
    count = 0
    N = int((int_save*n_tf_samples)/nC)
    stored_samples = []

    if samples_tf is None:
        print('generating....')
        ki_samples = ki.knock_in_many(theta_tf, tf_samples, m_tf, tf_idxs, int_burn=40000, nSamples=n_tf_samples, int_save=100)

        n_tf_samples = ki_samples.shape[1]
        samples = ki_samples
        print('samples shape is', samples.shape)
        np.save(f'../project/samples_MI_{t}_KI', samples.cpu().numpy())
        
    else:
        samples = torch.tensor(samples_tf).to(device)
        n_tf_samples = samples_tf.shape[1]
        #print('samples were provided')
    #print(theta_tg.shape, samples.shape, pi_tf.shape, m_tg.shape)
    samples = torch.tensor(samples).to(torch.float).to(device)
    
    exp = torch.mm(theta_tg.transpose(0,1), samples)+ m_tg.unsqueeze(1)
    pi1 = torch.exp(-exp)/ ( 1+torch.exp(-exp))
    pi0 = 1-pi1
    
    pi_tg1 = pi1
    pi_tg0 = 1-pi_tg1
    
    # The i,jth element of pi_tg is the P(gi|tj)

    p1_tot = pi1.sum(axis=1)/n_tf_samples # Should be n_genes x 1, where p_1_tot[i] is p(gi)
    p0_tot = pi0.sum(axis=1)/n_tf_samples # Should be n_genes x 1, where p_1_tot[i] is p(gi)
    
    #entropy = entropy.sum(axis=1)/n_tf_samples 
    entropy = calc_entropy(p1_tot)
    entropy[p1_tot==0] = 0
    entropy[p1_tot==1] = 0
    
    #print(f'p(g) shape is {p1_tot.shape}, {p0_tot.shape}')
    
    '''
    MI1 = torch.zeros(10).to(device)
    
    for gene in tqdm(range(10)):
        for config in range(n_tf_samples):
            term1 = pi_tg1[gene, config] * torch.log(pi_tg1[gene, config]/p1_tot[gene])
            term2 = pi_tg0[gene, config] * torch.log(pi_tg0[gene, config]/p0_tot[gene])
            MI1[gene] += term1 + term2
        MI1[gene]/=n_tf_samples
    '''
        
    term1 = (pi1 * torch.log2(pi1/p1_tot.unsqueeze(1))) 
    term2 = (pi0 * torch.log2(pi0/p0_tot.unsqueeze(1)))
    
    term1[pi1==0]=0
    term2[pi0==0]=0
        
    tot = term1 + term2
    #print('tot shape is', tot.shape)
    MI = tot.sum(axis=1)/tot.shape[1]
    del pi1
    del pi0
    del tot
    del pi_tg1
    del tf_samples
    #print('finally' , torch.all(MI1 == MI2[:10]))
    
    return MI, entropy
    
    
        