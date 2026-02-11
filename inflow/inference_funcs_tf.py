import torch 
def calc_deri(t, theta, m, sigmas, lambda_ = 0.05):
    """
        Get the derivative of the log likelihood with respect to theta and m. Difference b/w tg is that diagonal entries must be 0
        While the TF and TG models are found seperately, you can write the thetas together as one GRN with dimensions nTFx(nTF+nTG)
        and we learn a sub-matrix with the constraint that TFs cannot regulate themselves 
        Inputs-
            t : latents, i.e. TFs, dim should be #TF x #cells 
            theta: learned GRN dim is #TF_regulating x #TF_regulated genes here
            m: parameter to help match means, dim nG
            sigmas: binary data, nTF x nC. For the TF model, should be the same as t

        Returns-
            dLdtheta: gradient of likelihood w.r.t. theta
            dLdm: radient of likelihood w.r.t. m
    """ 
    nTF = t.shape[0]
    idxs = torch.arange(nTF, device=theta.device) # To keep the diagonal entries 0 (can't regulate themselves)
    
    exp = torch.mm(theta.transpose(0,1), t) + m.unsqueeze(1)    
    pi = torch.sigmoid(-exp)  # Newly calculate probabilites
    error = pi - sigmas # Error between predicted and actual
    

    ''' 
    This is what the theta_{alpha <- beta}.T looks like
        -- we want tf_alpha = sum_beta(tf_beta)=theta.T*t + m_alpha 
        -- its the sum of the effects of all the TFs (but alpha) on tf_alpha plus a term to learn mean of tf_alpha

                        tf1  tf2  tf_beta ...tf_nTF|
                    tf1  0                         |                           
                    tf2       0                    |                           
                    tf3_alpha      0               |
        theta_.T =  .                 .            |                           
                    .                    .         |                           
                    .                         .    |                           
                    tf_nTF                       0 |                           
                    --------------------------------

    '''
    
    dLdtheta = torch.mm(t, error.transpose(0,1)) - lambda_* torch.sign(theta) # grad of theta with lasso reg
    dLdtheta[idxs,idxs] = 0 # ===== set diagonal entries to 0 =======
    
    dLdm = error.sum(axis=1)

    return dLdtheta, dLdm

def calc_loglikelihood(sigmas,pi):
    """calculates log likelihood likelihood 
    Input--
        sigmas: data
        pi: model probabilities 
        the factor 1e-8 is here to ensure there are no nans in the calculation
    Output--
        L: log likelihood
    """
    L = torch.sum(torch.multiply(sigmas, torch.log(pi+1e-8))) + torch.sum(torch.multiply((1-sigmas), torch.log(1-pi +1e-8)))  
    return L

def adaptive_newparams(m, theta, mm, mtheta, vm,vtheta, dLdm, dLdtheta, beta1 =.9 , beta2  = .999, alpha = .1, i = 1, eps = 1e-8, optimizer  = 'adam'):
    """Calculates the new parameters using the old parameters through ADAM/yogi algorithm (standard calculation)"""
    
    mm = beta1*mm + (1-beta1)*dLdm
    mtheta = beta1*mtheta + (1-beta1)*dLdtheta
    
    if optimizer == 'adam':
        vm = beta2*vm + (1-beta2)*dLdm**2
        vtheta = beta2*vtheta + (1-beta2)*dLdtheta**2
    elif optimizer == 'yogi':
        vm = vm - (1-beta2)*torch.sign(vm - dLdm.unsqueeze(0)**2)*dLdm.unsqueeze(0)**2
        vtheta = vtheta - (1-beta2)*torch.sign(vtheta - dLdtheta**2)*dLdtheta**2
        pass
   
    ##### opt learning 
    mmhat = mm/(1-beta1**(i+1))
    mthetahat = mtheta/(1-beta1**(i+1))

    vmhat = vm/(1-beta2**(i+1))
    vthetahat = vtheta/(1-beta2**(i+1))
    
    m = m + alpha*mmhat/(torch.sqrt(vmhat) + eps)
    theta = theta + alpha*mthetahat/(torch.sqrt(vthetahat) + eps)

    return m,theta , mm, mtheta, vm, vtheta
