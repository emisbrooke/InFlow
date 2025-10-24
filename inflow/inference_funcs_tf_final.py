import torch 
def calc_deri(t,theta, m, sigmas, lambda_ = 0.05):
    """Get the derivative of the log likelihood with respect to theta and theta
    input: t : latents-TFs, dim should be #TF x #cells 
           theta: dim is #TF x #non-TF genes
    """ 
    nTF = t.shape[0]
    idxs = torch.arange(nTF)
    #exp = torch.mm(theta.transpose(0,1), t) + m.sum(axis=0).unsqueeze(1)
    exp = torch.mm(theta.transpose(0,1), t) + m.unsqueeze(1)    
    #print(exp.mean(),theta.mean(), exp.min())
    pi = torch.exp(-exp)/ ( 1+torch.exp(-exp))  
    #print('okok', pi.mean())
    error = pi - sigmas
    
    ''' 
        tf1  tf2  tf3 .  .  .  tfN |
    tf1  0                         |                           
    tf2       0                    |                           
    tf3            0               |
    .                 .            |                           
    .                    .         |                           
    .                         .    |                           
    tfN                          0 |                           
    -------------------------------
'''
    
    dLdtheta = torch.mm(t, error.transpose(0,1)) - lambda_* torch.sign(theta)
    #dLdtheta[theta==0] = 0
    dLdtheta[idxs,idxs] = 0
    
    dLdm = error.sum(axis=1)

    return dLdtheta, dLdm

def calc_loglikelihood(sigmas,pi):
    """calculates log likelihood likelihood """
    #print(sigmas.dtype, pi.dtype)
    #print(sigmas.dtype, pi.dtype)
    L = torch.sum(torch.multiply(sigmas, torch.log(pi+1e-8))) + torch.sum(torch.multiply((1-sigmas), torch.log(1-pi +1e-8)))
    
    return L

def adaptive_newparams(m, theta, mm, mtheta, vm,vtheta, dLdm, dLdtheta, beta1 =.9 , beta2  = .999, alpha = .1, i = 1, eps = 1e-8, optimizer  = 'adam'):
    """Calculates the new parameters using the old parameters through ADAM algorithm"""
    
    mm = beta1*mm + (1-beta1)*dLdm
    mtheta = beta1*mtheta + (1-beta1)*dLdtheta
    
    if optimizer == 'adam':
        vm = beta2*vm + (1-beta2)*dLdm**2
        vtheta = beta2*vtheta + (1-beta2)*dLdtheta**2
    elif optimizer == 'yogi':
        vm = vm - (1-beta2)*torch.sign(vm - dLdm.unsqueeze(0)**2)*dLdm.unsqueeze(0)**2
        vtheta = vtheta - (1-beta2)*torch.sign(vtheta - dLdtheta**2)*dLdtheta**2
        pass
   
    ##### unbias 
    mmhat = mm/(1-beta1**(i+1))
    mthetahat = mtheta/(1-beta1**(i+1))

    vmhat = vm/(1-beta2**(i+1))
    vthetahat = vtheta/(1-beta2**(i+1))
    
    m = m + alpha*mmhat/(torch.sqrt(vmhat) + eps)
    
    # Keep track of sparsity
    theta_zero = torch.where(theta==0)
    theta = theta + alpha*mthetahat/(torch.sqrt(vthetahat) + eps)
    
    # Make sure the 0s stay 0s
    theta[theta_zero]=0
    
    # Handle the NaN situation, e.g., by resetting theta or stopping the optimization
    
    return m,theta , mm, mtheta, vm, vtheta

def enforce_sparsity(theta, factor=1e-5):
    non_zero = theta[theta != 0]
    theta_mean = torch.mean(torch.abs(non_zero))
    #print(theta_mean)
    theta[torch.abs(theta) < (factor * theta_mean)] = 0

    sparsity = torch.count_nonzero(theta).item() / (theta.shape[0] * theta.shape[1])
    return theta, sparsity