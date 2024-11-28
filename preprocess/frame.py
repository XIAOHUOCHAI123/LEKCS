import cv2
import numpy as np
# import insightface
# from insightface.app import FaceAnalysis
# from insightface.data import get_image as ins_get_image
import os
import copy
import json
import random
import torch
import torchvision.models as models
from torchvision import transforms as trn
from torch.autograd import Variable as V
from torch.nn import functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.transforms import *
from resnet import ResNet
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'
device = get_device()
print(device)

arch = 'resnet50'
model_file = 'resnet50_places365.pth.tar'


model = models.__dict__[arch](num_classes=365).to(device)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()

centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


sframe = Image.open("image_path")
input_img = V(centre_crop(sframe).unsqueeze(0)).to(device)
out = modelout(input_img)