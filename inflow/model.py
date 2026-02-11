import torch
import numpy as np
import time
import math
import warnings
import inference_funcs_tf as tf
import inference_funcs_tg as tg
from importlib import reload

reload(tf)

'''
A file to learn the TF part of the model. It is very similar to the TG one, but the sampling and generating samples is different
'''

class TFModel:
    def __init__(self, age, tf_data, data, gene_type='TF', init_min=-1, init_max=1):
        #self.data_sets = data_sets
        '''
        Initialized the TF model 
        Input:
        ============================
            age: age of the mice for which we are learning
            tf_data: a matrix of since num TFs X num Cells (nTF, nC) containing the TF configurations in the data
            data: data to learn (same if learning the TF model)
            gene_type: Either 'TF' or 'TG' to use appropriate inference functions 
            init_min/init_max: default: -1, 1, how to initialize the matrix to learn
        '''
        self.nTF, self.nC = tf_data.shape
        self.nG, _ = data.shape # is gene type is TF nG is just nTF 

        self.gene_type = gene_type.upper()
        if self.gene_type not in {"TF", "TG"}:
            raise ValueError("gene_type must be either 'TF' or 'TG'")
        if self.nG == self.nTF and self.gene_type != 'TF':
            warnings.warn(
                "Input dimensions are ambiguous (nG == nTF). Continuing with TG mode as requested.",
                stacklevel=2,
            )
        if self.nG != self.nTF and self.gene_type == 'TF':
            raise ValueError("Input data appears to be TGs, but gene_type is TF")

        self.inf = tf if self.gene_type == "TF" else tg

        self.sigma = torch.as_tensor(data, dtype=torch.double)
        self.tf_data = torch.as_tensor(tf_data, dtype=torch.double)
        self.age = age
         
        theta = (init_max - init_min) * torch.rand(self.nTF, self.nG, dtype=torch.double) + init_min
        
        if self.gene_type == 'TF': # ==== TFs cannot regulate themselves =======
            idxs = torch.arange(self.nTF)
            theta[idxs,idxs] = 0
        
        theta = theta / torch.linalg.norm(theta)
        m = (init_max - init_min) * torch.rand(self.nG, dtype=torch.double) + init_min
        m = m / torch.linalg.norm(m)

        exp = torch.mm(theta.transpose(0, 1), self.tf_data) + m.unsqueeze(1)
        pi = torch.sigmoid(-exp)
        print(pi.shape, torch.any(torch.isnan(pi)), self.inf.calc_loglikelihood(self.sigma, pi))
        
        self.theta = theta
        self.m = m 
        self.pi = pi
        self.Larr = np.array([])


    def train(self, steps=int(10e7), thresh=0.5, patience=1000, min_delta_ll=50, optimizer='adam', use_gpu_if_avail=True, 
              alpha=.01, beta1=.9, beta2=.999, eps=1e-8, verbose=True, log_int=500, lambda_=5, eta=3e-4):
        '''
        A function to train our model. It is a function of the class GeneModel
        steps: default=10e7, max number of training steps
        thresh: default=0.5, threshold for the relative gradient. A possible stopping criteria: relative grad (|grad X|/|X|) < thresh
        patience: default=1000, number of steps to wait for likelikhood improvement 
        min_delta_ll: default=50, minimum increase in likelihood to be considered improvement
        optimizer: default='adam', can choose adam, yogi, or uses standard learning with learning rate eta
        use_gpu_if_avail: default=True, if there is a GPU, will use it when set to True
        alpha, beta1, beta2, eps: all the parameters for the optimizer if not using standard
        verbose: default=True, if True, more information printed through model learning
        log_int: number of iterations between storing learned parameters
        lambda_: default=5, the weight of Lasso regularization
        eta: default=3e-4, standard learning rate
        '''

        t0 = time.time() # to keep track of model speed
        device = torch.device("cuda:0" if (torch.cuda.is_available() and use_gpu_if_avail) else "cpu") # switch to gpu if available and requested
        
        # Set all arrays to tensors on desired device 
        sigma = self.sigma.to(torch.double).to(device); theta = self.theta.to(device); tf_data = self.tf_data.to(device)
        m = self.m.to(device)
        Larr = torch.zeros(steps, dtype=torch.double, device=device)
        
        # Initialize probability distribution
        exp = torch.mm(theta.transpose(0,1), tf_data) + m.unsqueeze(1)
        pi = torch.sigmoid(-exp)

        # Objective function penalized for regularization 
        penalized_obj = self.inf.calc_loglikelihood(sigma, pi) - lambda_ * torch.sum(torch.abs(theta))
        Larr[0] = penalized_obj
        print('l0', Larr[0],self.inf.calc_loglikelihood(sigma, pi), lambda_ * torch.sum(torch.abs(theta))) # initial likelihood

        stopper = HybridStopping(gradient_thresh=thresh, patience=patience, min_delta=min_delta_ll) # Initialize stopper for model
        tot_steps = steps        
        
        if verbose:
            print(f'running on {device}, lambda is {lambda_}')

        # Initialize optimizer parameters
        if optimizer == 'adam' or optimizer == 'yogi':
            mtheta = torch.zeros(theta.shape, dtype=torch.double).to(device)
            vtheta = torch.zeros(theta.shape, dtype=torch.double).to(device)
            mm = torch.zeros(m.shape, dtype=torch.double).to(device) 
            vm = torch.zeros(m.shape, dtype=torch.double).to(device) 
        
        print("Starting Inference...")
        for step in range(1, steps):        
            dLdtheta, dLdm = self.inf.calc_deri(tf_data, theta, m, sigma, lambda_=lambda_)
            tng = torch.linalg.norm(dLdtheta) / torch.linalg.norm(theta) # relative grad
            stop = stopper(tng, Larr[step - 1]) # check for stopping
            if stop:
                print(f'Stopping criterion met: tng={tng.item():.6f}')
                tot_steps = step
                break

            if optimizer == 'adam' or optimizer == 'yogi':
                m, theta, mm, mtheta, vm, vtheta = self.inf.adaptive_newparams(
                    m, theta, mm, mtheta, vm, vtheta, 
                    dLdm, dLdtheta, beta1=beta1, beta2=beta2, alpha=alpha, i=step, eps=eps, optimizer=optimizer
                )
                
            else:
                theta = theta + eta * dLdtheta
                m = m + eta * dLdm

            exp = torch.mm(theta.transpose(0,1), tf_data) + m.unsqueeze(1)
            pi = torch.sigmoid(-exp)
            
            penalized_obj = self.inf.calc_loglikelihood(sigma, pi) - lambda_ * torch.sum(torch.abs(theta))
            Larr[step] = penalized_obj
            
            if step % log_int == 0:
                if verbose:
                    print(f'After {step} iterations:')
                    print(f'Likelihood = {Larr[step]}, penalty={lambda_ * torch.sum(torch.abs(theta))}, tng={tng}, lambda={lambda_}')
                self.Larr = Larr[:step + 1]; self.pi=pi; self.sigma=sigma; self.theta=theta; self.m=m
                self.save_state()

        t1 = time.time()
        if verbose:
            print(f'Inference is over in {(t1-t0):.4f} seconds, {tot_steps} steps\nStep on avg takes {(t1-t0)/tot_steps:.4f} seconds')
        
        self.Larr = Larr[:tot_steps]; self.pi=pi; self.sigma=sigma; self.theta=theta; self.m=m
        self.save_state()
        

    def save_state(self):
        # This is intended as a backup if you lose your current model. Change the filename however you choose. It isn't meant as a final saving
        np.save(f'larr_tf_{self.age}m', self.Larr.cpu())
        np.save(f'pi_tf_{self.age}m', self.pi.cpu())
        np.save(f'sigmas_tf_{self.age}m', self.sigma.cpu())
        np.save(f'theta_tf_{self.age}m', self.theta.cpu())
        np.save(f'm_tf_{self.age}m', self.m.cpu())

    @staticmethod
    def update_samples(theta, tf_samples, m, row_idxs):
        '''
        A function that updates all chains given for one step 
        theta: theta learned
        tf_samples: samples to start from
        m: learned m
        row_idxs: TFs to try an update
        '''
        exp = torch.mm(theta.transpose(0, 1), tf_samples.to(dtype=theta.dtype, device=theta.device)) + m.unsqueeze(1) # Recalculate prob given other TFs
        pi = torch.sigmoid(-exp)

        # Randomly pick a row output and calculate flip prob
        chain_arr = torch.arange(len(row_idxs), device=tf_samples.device) # allows indexing for working with chains simulataneously
        chosen_pi = pi[row_idxs, chain_arr]
        
        new_spins = torch.bernoulli(chosen_pi) # decide to flip
        new_spins = new_spins.to(tf_samples.dtype)
        tf_samples[row_idxs, chain_arr] = new_spins

        return tf_samples


    def generate(self, int_burn=10000, nSamples=1000, int_save=1000, batch_size=1000):
        '''
        int_burn: default=10000, number of iterations to wait before sampling
        nSamples: default=1000, number of samples to generate
        int_save: default=1000, after burn in, how long to wait between sampling
        batch_size: default=1000, number of iterations to do at once. This 
        '''
        nTF, nChain = self.tf_data.shape # choose the number of starting chains to be the same as nC
        theta = self.theta
        m = self.m
        device = theta.device
        tf_samples = self.tf_data.to(device).clone()

        if nSamples < nChain: # if they want less samples than chosen chains, we can just do a chain for each sample
            nSamples = nChain
        stored_samples = []
            
        n_collect = math.ceil(nSamples / nChain)
        iters = n_collect * int_save + (int_burn - int_save) # number of iterations needed to obtain samples
        print('iters is ', iters, int_burn, int_save, nSamples)


        # Pre-generate all row indices for updating
        row_idx_all = torch.randint(nTF, (iters+1, nChain), device=tf_samples.device) # chooses which TF to try and activate/deactivate at each step
        
        count = 0 
        for i in range(0, iters+1, batch_size):  # Update in batches of 1000, this is limited by the GPU
            row_idx_batch = row_idx_all[i:i + batch_size]  # Select batch of indices
            print(f'on batch {i}')
            for row_idxs in row_idx_batch:
                tf_samples = self.update_samples(theta, tf_samples, m, row_idxs)
                if ((count >= int_burn) and (count % int_save == 0)) or (count==int_burn):
                    stored_samples.append(tf_samples.clone().cpu())
                count +=1

        if not stored_samples:
            raise RuntimeError("No samples were stored. Check int_burn/int_save/nSamples settings.")

        samples = torch.hstack(stored_samples)
        samples = samples[:, :nSamples]
        samples = samples.to(tf_samples.device)

        del tf_samples # free up some space 
        if torch.cuda.is_available():
            torch.cuda.empty_cache() # empty the cache

        return samples # should have shape nTF x nSamples 
    
class HybridStopping:
    '''
        defines ways that learning will be halted
        1. Early stop if relative gradient falls below threshold
        2. Patience-based stop if likelihood plateaus
    '''
    def __init__(self, gradient_thresh=0.03, patience=1000, min_delta=100):
        self.gradient_thresh = gradient_thresh
        self.patience = patience
        self.min_delta = min_delta
        self.best_ll = float('-inf')
        self.counter = 0
        
    def reset(self):
        self.best_ll = float('-inf')
        self.counter = 0
            
    def __call__(self, relative_grad, current_ll):
        
        grad_value = float(relative_grad.item() if torch.is_tensor(relative_grad) else relative_grad)
        if (grad_value < self.gradient_thresh) and (self.counter > self.patience):
            return True
        
        ll_value = float(current_ll.item() if torch.is_tensor(current_ll) else current_ll)
        if ll_value > self.best_ll + self.min_delta:
            self.best_ll = ll_value
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            return True
        
        return False
