import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import cv2
import numpy as np
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'
device = "cuda"
print(device)

#bundle = torchaudio.pipelines.HUBERT_BASE
bundle = torchaudio.pipelines.HUBERT_LARGE
model = bundle.get_model()
model = model.to(device)
model.eval()

#读取音频
waveform, sample_rate = torchaudio.load("{}".format(audio_path))

wav = torchaudio.functional.resample(wav, sample_rate, bundle.sample_rate)
feat, _ = model.extract_features(wav.to(device))