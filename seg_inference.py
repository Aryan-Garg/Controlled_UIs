#!/usr/bin/env python3
from share import *
import config

import cv2
import einops
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.uniformer import UniformerDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

from PIL import Image
import matplotlib.pyplot as plt


apply_uniformer = UniformerDetector()

model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./models/control_sd15_seg.pth', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    with torch.no_grad():
        input_image = HWC3(input_image)

        # NOTE: Why change the input segmentation mask?
        # detected_map = apply_uniformer(resize_image(input_image, detect_resolution))
        detected_map = resize_image(input_image, detect_resolution)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [detected_map] + results



input_image = Image.open("./test_imgs/39_31_1608092176371.png").convert('RGB')
input_image = np.array(input_image, dtype=np.uint8)
prompt = "A realistic mobile user interface design. No text or numbers."

num_samples = 4
image_resolution = 512 # 256 to 768 
strength = 2.0 # 1 to 2 with 0.1 step
guess_mode = True if len(prompt) == 0 else False # Turn to true if no prompt is given
detect_resolution = 1024 # 256 to 1024
ddim_steps = 100 # 1 to 100
scale = 28 # 1 to 30 (prompt guidance scale)
seed = random.randint(0, 414313142)
eta = 0 # ddim
a_prompt = 'best quality, extremely detailed, creative, unique, high quality, high resolution, high res, high quality'
n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, smudge, blur, low resolution, low res'
ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta]
results = process(*ips)


# show results in a grid
fig, ax = plt.subplots(1, num_samples+1, figsize=(20, 10))
ax[0].imshow(results[0])
ax[0].set_title("Segmentation Map")
ax[0].axis('off')
for i in range(1, num_samples+1):
    ax[i].imshow(results[i])
    ax[i].set_title(f"Sample {i}")
    ax[i].axis('off')
plt.show()