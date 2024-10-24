import torch
import torchaudio
import numpy as np

# tenzor audio signalu o tvaru (500tis x 1)
shape = (500000,1,)
tensor = torch.rand(shape)
print(f"Tensor Shape:", tensor.shape)

# chceme dostat tvar (batch size, 2^16, 1)
# matematika: 2^16=65536; 500 000/65536=7,6; zaokrouhlime a ziskame velikost: batch = 8;
# 8*2^16=524 288-500 000 = 24 288; abychom zaplnili batch, musime rozsirit tenzor 500tis na 524 288;

pad = (0,0,0,24288)
padded_tensor = torch.nn.functional.pad(tensor,pad, mode='constant', value=0)
print(f"Padded Tensor Shape:", padded_tensor.shape)

# reshape padded tenzoru do pozadovaneho tvaru (batch, 2^16, 1)
reshaped_tensor = padded_tensor.reshape((8,65536,1))
print(f"Reshaped Tensor Shape:", reshaped_tensor.shape)