import torch
import numpy as np

def soft_min(x,y,gamma=5):
    if isinstance(x,torch.Tensor):
        expfun = torch.exp
    elif isinstance(x,np.ndarray):
        expfun = np.exp
    exp1 = expfun((y-x)/2)
    exp2 = expfun((x-y)/2)
    return (exp1*x+exp2*y)/(exp1+exp2)

def soft_max(x,y,gamma=5):
    if isinstance(x,torch.Tensor):
        expfun = torch.exp
    elif isinstance(x,np.ndarray):
        expfun = np.exp
    exp1 = expfun((x-y)/2)
    exp2 = expfun((y-x)/2)
    return (exp1*x+exp2*y)/(exp1+exp2)

def soft_sat(x,x_min=None,x_max=None,gamma=5):
    if x_min is None and x_max is None:
        return x
    elif x_min is None and x_max is not None:
        return soft_min(x,x_max,gamma)
    elif x_min is not None and x_max is None:
        return soft_max(x,x_min,gamma)
    else:
        if isinstance(x_min,torch.Tensor) or isinstance(x_min,np.ndarray):
            assert (x_max>x_min).all()
        else:
            assert x_max>x_min
        xc = x - (x_min+x_max)/2
        if isinstance(x,torch.Tensor):
            return xc/(torch.pow(1+torch.pow(torch.abs(xc*2/(x_max-x_min)),gamma),1/gamma))+(x_min+x_max)/2
        elif isinstance(x,np.ndarray):
            return xc/(np.power(1+np.power(np.abs(xc*2/(x_max-x_min)),gamma),1/gamma))+(x_min+x_max)/2
        else:
             raise Exception("data type not supported")


def Gaussian_importance_sampling(mu0,var0,mu1,var1,num_samples=1):
    """ perform importance sampling between two Gaussian distributions

    Args:
        mu0 (torch.Tensor): [B,D]: mean of the target Gaussian distribution
        var0 (torch.Tensor): [B,D]: variance of the target Gaussian distribution
        mu1 (torch.Tensor): [B,D]: mean of the proposal Gaussian distribution
        var1 (torch.Tensor): [B,D]: variance of the proposal Gaussian distribution
        num_samples (int, optional): number of samples. Defaults to 1.

    Returns:
        samples: [B,num_samples,D]: samples from the proposal Gaussian distribution
        log_weights: [B,num_samples]: log weights of the samples
    """
    samples = torch.randn([*mu1.shape[:-1],num_samples,mu1.shape[-1]],device=mu1.device)*torch.sqrt(var1).unsqueeze(-2)+mu1.unsqueeze(-2)
    log_weights = -0.5*torch.log(2*np.pi*var0.unsqueeze(-2))-0.5*torch.pow(samples-mu0.unsqueeze(-2),2)/var0.unsqueeze(-2)+0.5*torch.log(2*np.pi*var1.unsqueeze(-2))+0.5*torch.pow(samples-mu1.unsqueeze(-2),2)/var1.unsqueeze(-2)
    return samples,log_weights
