import torch
import json
import torch.nn as nn
from torchvision import transforms
import os
import cv2
import time
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)
from typing import Dict
from PIL import Image

# Device on which to run the model
# Set to cuda to load on GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'
device = "cuda"
print(device)
# Pick a pretrained model and load the pretrained weights
model_name = "slowfast_16x8_r101_50_50"
#model_name = "slowfast_r50"
model = torch.hub.load("facebookresearch/pytorchvideo", model=model_name, pretrained=True)
model.blocks[-1].proj = nn.Identity()
#model.blocks[-1].output_pool = nn.Identity()
# Set to eval mode and move to desired device
model = model.to(device)
model = model.eval()
print(model)

sampling_rate = 2
frames_per_second = 24
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
side_size = 256
crop_size = 256#裁剪尺寸
num_frames = 64#采样帧数
alpha = 4#慢速

class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """
    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list

transform =  ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x/255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(
                size=side_size
            ),
            CenterCropVideo(crop_size),
            PackPathway()
        ]
    ),
)

class mySlowfast(torch.nn.Module):
    def __init__(self):
        super(mySlowfast, self).__init__()
        slowfast_pretrained_features = nn.Sequential(*list(model.children())[0])

        self.feature_extraction = torch.nn.Sequential()
        self.slow_avg_pool = torch.nn.Sequential()
        self.fast_avg_pool = torch.nn.Sequential()
        self.adp_avg_pool = torch.nn.Sequential()
        self.poolcon      = torch.nn.Sequential()
        self.reshead      = torch.nn.Sequential()


        for x in range(0, 5):
            self.feature_extraction.add_module(str(x), slowfast_pretrained_features[x])
        
        self.poolcon.add_module("poolcon", slowfast_pretrained_features[5])#添加第五个
        self.reshead.add_module("reshead", slowfast_pretrained_features[6])
        self.slow_avg_pool.add_module('slow_avg_pool', slowfast_pretrained_features[5].pool[0])
        self.fast_avg_pool.add_module('fast_avg_pool', slowfast_pretrained_features[5].pool[1])
        self.adp_avg_pool.add_module('adp_avg_pool', slowfast_pretrained_features[6].output_pool)

    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extraction(x)
            
            x1 = [x[0].clone(),x[1].clone()]
            
            x1 = self.poolcon(x1)
            x1 = self.reshead(x1)

        return x1

mysl = mySlowfast()


cap = cv2.VideoCapture("{}/{}".format(video_path,video_files[idx]))

start = 0
end   = 0

frames = get_frames(idx,cap,start,end)
video_tensor = get_video_tensor(frames,num_frames)
inputs = transform(video_tensor)
inputs = inputs["video"]

inputs = [i.to(device)[None, ...] for i in inputs]
slowfast_feature = mysl(inputs)


        
