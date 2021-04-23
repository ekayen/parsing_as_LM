import torch
import numpy as np
import random

def format_elapsed(time2,time1):
    """
    From Trang Tran's prosody_nlp repo
    """
    elapsed_time = time2 - time1
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(int(hours), int(minutes), int(seconds))
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string


def masked_softmax(vec, mask=None, dim=1, epsilon=1e-5,mask_val=-100): # using this as a mask value cause BERT does?
    if dim==1:
        vec = vec.view(vec.shape[-1])
    vec[~mask.bool()] = mask_val
    if dim==1:
        vec = vec.view(1,vec.shape[-1])
    masked_exps = torch.exp(vec)
    masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
    masked_soft = masked_exps/masked_sums
    return masked_soft

def masked_softmax_by_idx(vec,idxs,dim=1,epsilon=1e-5):
    idxs = np.array(idxs)
    mask = torch.zeros(vec.shape[dim])
    mask[idxs] = 1
    return masked_softmax(vec,mask,dim=dim,epsilon=epsilon)

def mask_by_idx(vec,idxs,dim=1,mask_val=-10000):
    idxs = np.array(idxs)
    mask = torch.zeros(vec.shape[dim])
    mask[idxs] = 1
    if dim==1:
        vec = vec.view(vec.shape[-1])
    vec[~mask.bool()] = mask_val
    if dim==1:
        vec = vec.view(1,vec.shape[-1])
    return vec

def set_seeds(seed,verbose=False):
    if verbose: print(f'setting seed to {seed}')
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__=="__main__":
    vec = torch.FloatTensor([[0,1,2,3,4,5,6,7,8,9]]).long()
    idxs = [4,5,6,7,8]
    msft = masked_softmax_by_idx(vec,idxs,dim=1)
    mask = torch.FloatTensor([0,0,0,0,1,1,1,1,1,0])
    sft = masked_softmax(vec,mask,dim=1)
    masked_v = mask_by_idx(vec,idxs,dim=1)
    import pdb;pdb.set_trace()
    
