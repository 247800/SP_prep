import torch
import torchaudio
import numpy as np

# tenzor audio signalu o tvaru (500tis x 1)
shape = (500000,1,)
tensor = torch.rand(shape)
print(f"Tensor Shape:", tensor.shape)

# pokud chceme tenzor ve tvaru (batch size, 2^16, 1)
# 500 000/(2^16)=7,6; zaokrouhlime a ziskame velikost: batch = 8;
# 8*(2^16)=524 288; abychom zaplnili batch, musime rozsirit tenzor z 500tis na 524 288;
pad = (0,0,0,24288)
padded_tensor = torch.nn.functional.pad(tensor,pad, mode='constant', value=0)
print(f"Padded Tensor Shape:", padded_tensor.shape)

# reshape padded tenzoru do pozadovaneho tvaru (batch, 2^16, 1)
reshaped_tensor = padded_tensor.reshape((8,65536,1))
print(f"Reshaped Tensor Shape:", reshaped_tensor.shape)

# zkraceni puvodniho tenzoru: chceme zkratit cas nahravky, potrebujeme tedy snizit pocet vzorku
trimmed_tensor = tensor[0:250000,:]
print(f"Trimmed Tensor Shape:", trimmed_tensor.shape)
