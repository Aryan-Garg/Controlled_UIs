#!/usr/bin/env python3
from share import *

import torch
import torch.nn as nn

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint

# Configs
resume_path = './models/control_sd15_ini.ckpt'
batch_size = 8
logger_freq = 300

######################## Changed from Base ControlNet ########################
learning_rate = 2e-6         
sd_locked = False           
##############################################################################

only_mid_control = False

# First use cpu to load models. 
# Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, 
                        num_workers=32, 
                        batch_size=batch_size, 
                        shuffle=True)

logger = ImageLogger(batch_frequency=logger_freq)

model_ckpt_logger = ModelCheckpoint(every_n_epochs=1)

trainer = pl.Trainer(gpus=1, 
                    #  default_root_dir="./model/path/"
                     precision=32, 
                     callbacks=[logger, model_ckpt_logger], 
                     max_epochs=-1, # Infinte epochs
                     accumulate_grad_batches=8)
# print(vars(trainer))
# Train!
trainer.fit(model, dataloader)