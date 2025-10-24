import torch 
import numpy as np
import time
import inference_funcs_tf_final as tg
#import inference_funcs_tf as tf
from importlib import reload
from tqdm import tqdm
reload(tg)

class GeneModel:
    def __init__(self, age, tf_data, data, init_min=-1, init_max=1, exclude_min=-0.01, exclude_max=0.01):
        #self.data_sets = data_sets
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
              alpha=.01, beta1=.9, beta2=.999, eps=1e-8, verbose=True, log_int=500, lambda_=5, lambda_int=1000 ,eta=3e-4):
        t0 = time.time()
        device = torch.device("cuda:0" if (torch.cuda.is_available() and use_gpu_if_avail) else "cpu")
        
        sigma = self.sigma.to(torch.double).to(device); theta = self.theta.to(device); tf_data = self.tf_data.to(device)
        m = self.m.to(device)
        Larr = torch.zeros(steps).to(device)
            
        exp = torch.mm(theta.transpose(0,1), tf_data) + m.unsqueeze(1)
        pi = torch.exp(-exp)/ ( 1+torch.exp(-exp))

        penalized_obj = tg.calc_loglikelihood(sigma, pi) - lambda_ * torch.sum(torch.abs(theta))
        Larr[0] = penalized_obj
        print('l0', Larr[0],tg.calc_loglikelihood(sigma, pi), lambda_ * torch.sum(torch.abs(theta)))

        lambda_current = lambda_
        lambda_met = True
        #max_lambda = 25
        stopper = HybridStopping(gradient_thresh=thresh, patience=patience, min_delta=min_delta_ll)
        tot_steps = steps        
        
        if verbose:
            print(f'running on {device}, lambda is {lambda_}')

        if optimizer == 'adam' or optimizer == 'yogi':
            mtheta = torch.zeros(theta.shape, dtype=torch.double).to(device)
            vtheta = torch.zeros(theta.shape, dtype=torch.double).to(device)
            mm = torch.zeros(m.shape, dtype=torch.double).to(device) 
            vm = torch.zeros(m.shape, dtype=torch.double).to(device) 
        
        print("Starting Inference...")
        for step in range(1, steps):        
            dLdtheta, dLdm = tg.calc_deri(tf_data, theta, m, sigma, lambda_=lambda_current)
            tng = torch.linalg.norm(dLdtheta) / torch.linalg.norm(theta)
            stop = stopper(tng, Larr[step-1])
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
                    print(f'Likelihood = {Larr[step]}, penalty={lambda_current * torch.sum(torch.abs(theta))}, tng={tng}, lambda={lambda_current}')
                self.Larr = Larr[:step]; self.pi=pi; self.sigma=sigma; self.theta=theta; self.m=m
                self.save_state()

        t1 = time.time()
        if verbose:
            print(f'Inference is over in {(t1-t0):.4f} seconds, {tot_steps} steps\nStep on avg takes {(t1-t0)/tot_steps:.4f} seconds')
        
        self.Larr = Larr[:tot_steps]; self.pi=pi; self.sigma=sigma; self.theta=theta; self.m=m
        self.save_state()
        

    def save_state(self):
        #save_file = f'output_final/{self.age}m'
        save_file = 'output_tg_thymus'
        np.save(f'larr_tg_{self.age}m', self.Larr.cpu())
        np.save(f'pi_tg_{self.age}m', self.pi.cpu())
        np.save(f'sigmas_tg_{self.age}m', self.sigma.cpu())
        np.save(f'theta_tg_{self.age}m', self.theta.cpu())
        np.save(f'm_tg_{self.age}m', self.m.cpu())

    def generate(self, n_samples):
        idxs = torch.randint(self.nC, (n_samples,))
        samples = torch.bernoulli(self.pi[:,idxs])
        return samples
    
    
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
