import torch 
def calc_deri(t,theta, m, sigmas, lambda_ = 0.05):
    """
        Get the derivative of the log likelihood with respect to theta and m
        While the TF and TG models are found seperately, you can write the thetas together as one GRN with dimensions nTFx(nTF+nTG)
        and we learn a sub-matrix with the constraint that TG cannot regulate TFs
        Inputs: 
            t : latents, i.e. TFs, dim should be #TF x #cells 
            m: parameter to help match means, dim nG
            theta: learned GRN dim is #TF (nTF) x #non-TF genes (nG)
            sigmas: binary data, #TGs x #cells. For the TG model this is the actual TG data

        Returns-
            dLdtheta: gradient of likelihood w.r.t. theta
            dLdm: radient of likelihood w.r.t. m
    """ 
    
    exp = torch.mm(theta.transpose(0,1), t) + m.unsqueeze(1)    
    pi = torch.sigmoid(-exp)
    error = pi - sigmas

    ''' 
    This is what the theta_{i<-alpha}.T looks like, the 0,0 element is how much influence tf0 has on tg0
        -- we want tg_i = sum_alpha(tf_alpha) = theta.T*t + m_i 
        -- its the sum of the effects of all the TFs on tg_i plus a term to learn mean of tg_i

                    tf1  tf2 tf_alpha...tf_nTF|
                tg0   .                       |                           
                tg1        .                  |                           
                tg_i           .              |
        theta.T = .                 .         |                           
                .                    .        |                           
                .                         .   |                           
                tg_nG                         |                           
                -------------------------------

    '''
    
    dLdtheta = torch.mm(t, error.transpose(0,1)) - lambda_* torch.sign(theta) # grad of theta with lasso reg
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
    """Calculates the new parameters using the old parameters through ADAM/ypgi algorithm"""
    
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
