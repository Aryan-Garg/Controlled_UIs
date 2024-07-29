#!/usr/bin/env python3

import os
import torch
from PIL import Image
from torchvision import transforms
import sys
sys.path.append("../")
from cldm.model import create_model

ckpt_path = "v1-5-pruned.ckpt"
model = create_model(config_path='cldm_v15.yaml')
pretrained_weights = torch.load(ckpt_path, map_location='cpu')

# Print and write model to a text file
with open("SD_model.txt", "w") as f:
    f.write(str(model))