from hear21passt.base import AugmentMelSTFT, PasstBasicWrapper
from models.passt import PaSST
import torch

sr = 32000

mel = AugmentMelSTFT(
    n_mels=128, sr=sr, win_length=800, hopsize=320, n_fft=1024, freqm=48,
    timem=192,
    htk=False, fmin=0.0, fmax=None, norm=1, fmin_aug_range=10,
    fmax_aug_range=2000,
)
net = PaSST()
model = PasstBasicWrapper(mel=mel, net=net, mode="logits")


print(model.mel) # Extracts mel spectrogram from raw waveforms.
print(model.net) # the transformer network.
# example inference

model.eval()
model = model.cuda()
with torch.no_grad():
    # audio_wave has the shape of [batch, seconds*32000] sampling rate is 32k
    # example audio_wave of batch=3 and 10 seconds
    audio = torch.ones((3, sr * 10))*0.5
    audio_wave = audio.cuda()
    logits=model(audio_wave) 