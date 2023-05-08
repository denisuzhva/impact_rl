import torch
import numpy as np



def bit_crush(data, bits_min=1, bits_max=9):
    bits_new = np.random.randint(bits_min, bits_max)
    mcv_new = 2**bits_new - 1
    data_crushed = torch.round(data * mcv_new) / mcv_new    
    return data_crushed