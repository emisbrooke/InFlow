import torch 
import numpy as np
import time
import inference_funcs_tf as tg
#import inference_funcs_tf as tf
from importlib import reload
from tqdm import tqdm
reload(tg)

class GeneModel:
    def __init__(self, age, tf_data, data, init_min=-1, init_max=1):
        #self.data_sets = data_sets
        '''
        age: age of the mice for which we are learning
        tf_data: a matrix of since num TFs X num Cells (nTF, nC) containing the TF configurations in the data
        data: data to learn (same if learning the TF model)
        init_min/init_max: default: -1, 1, how to initialize the matrix to learn

        '''
        self.nTF, self.nC = tf_data.shape
        self.nG, _ = data.shape
        self.sigma = torch.tensor(data)
        self.tf_data = torch.tensor(tf_data)
        self.age = age
        idxs = torch.arange(self.nTF)
         
        theta = (init_max - init_min) * torch.rand(self.nTF, self.nG, dtype=torch.double) + init_min
        theta = theta / torch.linalg.norm(theta)
        theta[idxs,idxs] = 0
        
        m = (init_max - init_min) * torch.rand(self.nG, dtype=torch.double) + init_min
        m = m / torch.linalg.norm(m)
        
        print(self.tf_data.transpose(0,1).shape, theta.shape, m.unsqueeze(1).shape)
        exp = torch.mm(theta.transpose(0,1), self.tf_data)+ m.unsqueeze(1)
        pi = torch.exp(-exp)/ ( 1+torch.exp(-exp)) 
        print(pi.shape, torch.any(torch.isnan(pi)), tg.calc_loglikelihood(self.sigma, pi))
        
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
        ****** lambda_int: leftover boooo
        eta: default=3e-4, standard learning rate
        '''
        t0 = time.time() # to keep track of model speed
        device = torch.device("cuda:0" if (torch.cuda.is_available() and use_gpu_if_avail) else "cpu") # switch to gpu if available and requested
        
        # Set all arrays to tensors on desired device 
        sigma = self.sigma.to(torch.double).to(device); theta = self.theta.to(device); tf_data = self.tf_data.to(device)
        m = self.m.to(device)
        Larr = torch.zeros(steps).to(device)
        
        # Initialize probability distribution
        exp = torch.mm(theta.transpose(0,1), tf_data) + m.unsqueeze(1)
        pi = torch.exp(-exp)/ ( 1+torch.exp(-exp))

        # Objective function penalized for regularization 
        penalized_obj = tg.calc_loglikelihood(sigma, pi) - lambda_ * torch.sum(torch.abs(theta))
        Larr[0] = penalized_obj
        print('l0', Larr[0],tg.calc_loglikelihood(sigma, pi), lambda_ * torch.sum(torch.abs(theta))) # initial likelihood

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
            dLdtheta, dLdm = tg.calc_deri(tf_data, theta, m, sigma, lambda_=lambda_)
            tng = torch.linalg.norm(dLdtheta) / torch.linalg.norm(theta) # relative grad
            stop = stopper(tng, Larr[step-1]) # check for stopping
            if stop:
                print(f'Stopping criterion met', {tng})
                tot_steps = step
                break

            if optimizer == 'adam' or optimizer == 'yogi':
                #print('dfasdfs')
                m, theta, mm, mtheta, vm, vtheta = tg.adaptive_newparams(
                    m, theta, mm, mtheta, vm, vtheta, 
                    dLdm, dLdtheta, beta1=beta1, beta2=beta2, alpha=alpha, eps=eps, optimizer=optimizer
                )
                
            else:
                #print('excuse me')
                theta = theta + eta * dLdtheta
                m = m + eta * dLdm

            exp = torch.mm(theta.transpose(0,1), tf_data) + m.unsqueeze(1)
            pi = torch.exp(-exp)/ ( 1+torch.exp(-exp))
            
            penalized_obj = tg.calc_loglikelihood(sigma, pi) - lambda_ * torch.sum(torch.abs(theta))
            Larr[step] = penalized_obj
            
            if step % log_int == 0:
                if verbose:
                    print(f'After {step} iterations:')
                    print(f'Likelihood = {Larr[step]}, penalty={lambda_ * torch.sum(torch.abs(theta))}, tng={tng}, lambda={lambda_}')
                self.Larr = Larr[:step]; self.pi=pi; self.sigma=sigma; self.theta=theta; self.m=m
                self.save_state()

        t1 = time.time()
        if verbose:
            print(f'Inference is over in {(t1-t0):.4f} seconds, {tot_steps} steps\nStep on avg takes {(t1-t0)/tot_steps:.4f} seconds')
        
        self.Larr = Larr[:tot_steps]; self.pi=pi; self.sigma=sigma; self.theta=theta; self.m=m
        self.save_state()
        

    def save_state(self):
        # This is intended as a backup if you lose your current model. Change the filename however you choose. It isn't meant as a final saving
        np.save(f'larr_tg_{self.age}m', self.Larr.cpu())
        np.save(f'pi_tg_{self.age}m', self.pi.cpu())
        np.save(f'sigmas_tg_{self.age}m', self.sigma.cpu())
        np.save(f'theta_tg_{self.age}m', self.theta.cpu())
        np.save(f'm_tg_{self.age}m', self.m.cpu())


    def generate(int_burn=10000, nSamples=1000, int_save=1000, batch_size=1000):
        '''
        int_burn: default=10000, number of iterations to wait before sampling
        nSamples: default=1000, number of samples to generate
        int_save: default=1000, after burn in, how long to wait between sampling
        batch_size: default=1000, number of iterations to do at once. This 
        '''
        nTF, nChain = self.tf_data.shape # choose the number of starting chains to be the same as nC
        #print('nTF', nTF)
        tf_samples = tf_data.clone()

        if nSamples < nChain: # if they want less samples than chosen chains, we can just do a chain for each sample
            nSamples = nChain
        stored_samples = []
            
        iters = (nSamples // nChain) * int_save + (int_burn - int_save) # number of iterations needed to obtain samples
        print('iters is ', iters, int_burn, int_save, nSamples)


        # Pre-generate all row indices for updating
        row_idx_all = torch.randint(nTF, (iters+1, nChain), device=tf_samples.device) # chooses which TF to try and activate/deactivate at each step
        
        count = 0 
        for i in range(0, iters+1, batch_size):  # Update in batches of 1000, this is limited by the GPU
            row_idx_batch = row_idx_all[i:i+1000]  # Select batch of indices
            print(f'on batch {i}')
            for row_idxs in row_idx_batch:
                tf_samples = update_samples(self.theta, tf_samples, self.m, row_idxs)
                if ((count >= int_burn) and (count % int_save == 0)) or (count==int_burn):
                    stored_samples.append(tf_samples.clone().cpu())
                count +=1
        
        samples = torch.hstack(stored_samples)
        samples = samples.to(tf_samples.device)

        del tf_samples # free up some space 
        torch.cuda.empty_cache() # empty the cache

        return samples # should have shape nTF x nSamples
    
    @torch.jit.script
    def update_samples(theta, tf_samples, m, row_idxs) -> torch.Tensor:
        '''
        A function that updates all chains given for one step 
        theta: theta learned
        tf_samples: samples to start from
        m: learned m
        row_idxs: TFs to try an update
        '''
        exp = torch.mm(theta.transpose(0, 1), tf_samples.to(torch.float)) + m.unsqueeze(1) # Recaculate prob given other TFs
        pi = torch.sigmoid(-exp)
        #print(pi.is_floating_point())

        # Randomly pick a row of put and calculate flip prob
        chain_arr = torch.arange(len(row_idxs), device=tf_samples.device) # allows indexing for working with chains simulataneously
        chosen_pi = pi[row_idxs, chain_arr]
        
        new_spins = torch.bernoulli(chosen_pi) # decide to flip
        new_spins = new_spins.to(tf_samples.dtype)
        tf_samples[row_idxs, chain_arr] = new_spins

        return tf_samples
    
    
class HybridStopping:
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
        
        if (relative_grad < self.gradient_thresh) and (self.counter > self.patience):
            return True
        
        if current_ll > self.best_ll + self.min_delta:
            self.best_ll = current_ll
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            return True
        
        return False
