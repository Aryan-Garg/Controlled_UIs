import json
import cv2
import numpy as np
from PIL import Image
import os
import random

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor


class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('/home/researcher/Documents/dataset/original_datasets/webui_prompts.json', 'rt') as f:
            self.data = json.load(f)
            # for line in f:
            #     self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        item = self.data[idx]
        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']
        # print(source_filename, target_filename, prompt[:10])
        # TODO: Pre-process segmentation masks (Remove Text fields. Put in separate dir)
        source = cv2.imread(source_filename, cv2.IMREAD_UNCHANGED)
        target = cv2.imread(target_filename, cv2.IMREAD_UNCHANGED)
        print("L>>>", source, source_filename)
        print(source.shape)
        # print("\nidx:", idx, "|", source.shape, target.shape, len(prompt))
        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        # target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        # Resize images to 512x256
        source = cv2.resize(source, (256, 512), interpolation=cv2.INTER_CUBIC)
        target = cv2.resize(target, (256, 512), interpolation=cv2.INTER_CUBIC)

        # except:
        #     item = self.data[1]
        #     source_filename = item['source']
        #     target_filename = item['target']
        #     prompt = item['prompt']
        #     print(source_filename)
        #     # TODO: Pre-process segmentation masks (Remove Text fields. Put in separate dir)
        #     source = cv2.imread(source_filename, cv2.IMREAD_UNCHANGED)
        #     target = cv2.imread(target_filename, cv2.IMREAD_UNCHANGED)

        #     # print("\nidx:", idx, "|", source.shape, target.shape, len(prompt))

        #     # Do not forget that OpenCV read images in BGR order.
        #     source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        #     # target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        #     # Resize images to 512x256
        #     source = cv2.resize(source, (256, 512), interpolation=cv2.INTER_CUBIC)
        #     target = cv2.resize(target, (256, 512), interpolation=cv2.INTER_CUBIC)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)


# Dataset
class IPA_and_Control_Dataset(Dataset):
    def __init__(self, json_file, size=512, t_drop_rate=0.05, 
                 i_drop_rate=0.05, ti_drop_rate=0.05, image_root_path=""):
        super().__init__()

        # self.tokenizer = tokenizer
        self.size = size
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path

        self.data = json.load(open(json_file)) # list of dict: [{"image_file": "1.png", "text": "A dog"}]

        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.clip_image_processor = CLIPImageProcessor()
        
        
    def __getitem__(self, idx):
        item = self.data[idx] 
        prompt = item["text"]
        image_file = item["image_file"]
        
        # read image
        raw_image = Image.open(os.path.join(self.image_root_path, image_file))
        image = self.transform(raw_image.convert("RGB"))
        clip_image = self.clip_image_processor(images=raw_image, return_tensors="pt").pixel_values
        
        # drop
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            prompt = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            prompt = ""
            drop_image_embed = 1

        # get text and tokenize
        # text_input_ids = self.tokenizer(
        #     prompt,
        #     max_length=self.tokenizer.model_max_length,
        #     padding="max_length",
        #     truncation=True,
        #     return_tensors="pt"
        # ).input_ids
        
        return {
            "image": image,
            "txt": prompt,
            "clip_image": clip_image,
            "drop_image_embed": drop_image_embed
        }
    

    def __len__(self):
        return len(self.data)
    

def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    clip_images = torch.cat([example["clip_image"] for example in data], dim=0)
    drop_image_embeds = [example["drop_image_embed"] for example in data]

    return {
        "images": images,
        "text_input_ids": text_input_ids,
        "clip_images": clip_images,
        "drop_image_embeds": drop_image_embeds
    }
    
