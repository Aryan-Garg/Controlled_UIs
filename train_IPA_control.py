#!/usr/bin/env python3
from share import *

import torch
import torch.nn as nn

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

import a1111_ipadapter
from a1111_ipadapter.image_proj_models import ImageProjModel

# Configs
resume_path = './models/control_sd15_ini.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


def load_pathGAN_discriminator():
    pass


def IPA_Integration(unet):
    image_encoder = load_pathGAN_discriminator()
    #ip-adapter
    image_proj_model = ImageProjModel()

    input_path = "./models/v1-5-pruned.ckpt"
    pretrained_weights = torch.load(input_path)
    if 'state_dict' in pretrained_weights:
        pretrained_weights = pretrained_weights['state_dict']
    # Add the CA layers
    target_dict = {}
    unet_sd = unet.state_dict()
    adapter_modules_list = nn.ModuleList()
    for name in unet_sd.keys():
        # print(name)
        # These will be added to the model.
        if "diffusion_model" in name and "transformer" in name and \
            "attn" in name and ("to_k" in name or "to_v" in name):
            # print("[/] Copying", name, "\nparam.shape:", v.shape)
            # print("------------------------")
            tar_str = ""
            for x in name.split(".")[:-1]:
                tar_str += x + "."
            tar_str = tar_str[:-1] + "_ip"

            cross_attention_dim = 1024 # Always send 1024 sized image vector.
            hidden_size = unet_sd[name].shape[0]

            target_dict[tar_str] = nn.Linear(cross_attention_dim, hidden_size, bias=False)
            target_dict[tar_str].weight.data = unet_sd[name].clone()

            adapter_modules_list.append(target_dict[tar_str])

        if name in pretrained_weights:
            target_dict[name] = pretrained_weights[name].clone() # keep original SD weights always.
        else:
            target_dict[name] = unet_sd[name].clone()

    # TODO: Insert the adapter modules into the unet at appropriate places and perform the addition of cross-attention layers.
    unet.load_state_dict(target_dict, strict=True)
    torch.save(unet.state_dict(), "./models/IPA_control_sd15_ini.ckpt")
    print('Done IPA Integration.')

    return unet, adapter_modules_list

if __name__ == '__main__':
    # First use cpu to load models. 
    # Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./models/cldm_v15_ipa.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control
    
    flow_input = True
    if flow_input:
        model, adapter_modules = IPA_Integration(model)
        with open("models/model_definition_txts/adapterModules.txt", "w") as f:
            f.write(str(adapter_modules))
    exit()

    # Misc
    dataset = MyDataset()
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=logger_freq)
    trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])

    # Train!
    trainer.fit(model, dataloader)
