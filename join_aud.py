import torch
import torchaudio
import matplotlib.pyplot as plt

waveform, sample_rate = torchaudio.load("synth.wav")
print("Tensor shape:", waveform.shape)

joined_tensor = torch.cat((waveform, waveform), 1)
print("Joined Tensor shape:", joined_tensor.shape)

plt.figure()
plt.plot(joined_tensor.t().numpy())
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.title("Joined Tensor Waveform")
plt.show()
