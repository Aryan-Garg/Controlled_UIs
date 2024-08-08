#!/usr/bin/env python3
import os
import random
import argparse
from pathlib import Path
import json
import itertools
import time
import ruamel.yaml as yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from ip_adapter.ip_adapter import ImageProjModel
from ip_adapter.utils import is_torch2_available
if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor

from models_eyeformer.model_tracking import TrackingTransformer
from pytorchSoftdtwCuda.soft_dtw_cuda import SoftDTW
from tqdm.auto import tqdm
# Flow Adapter
#########################################################################
softLoss = SoftDTW(use_cuda=True, gamma=0.1)
def get_soft_dtw_Loss(pred, target):
    loss = softLoss(pred, target)
    return loss


class FlowEncoder_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.add_module('encoder', nn.Sequential(
            nn.Linear(45, 128),
            nn.LeakyReLU(0.3),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.3),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.3),
            nn.Linear(512, 1024)))

        self.add_module('decoder', nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.3),
            nn.Linear(128, 45)))

    def forward(self, x):
      x = x.view(x.size(0), -1)
      x = self.encoder(x)
      x = self.decoder(x)
      return x
    

class FlowEncoder(nn.Module):
    def __init__(self, cross_attention_dim, clip_embeddings_dim, clip_extra_context_tokens, flow_latenizer):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.clip_embeddings_dim = clip_embeddings_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.flow_latenizer = flow_latenizer
    
    def forward(self, x):
        flow_embeds = x.view(x.size(0), -1)
        flow_embeds = self.flow_latenizer(flow_embeds)
        # print("Final flow_embeds Shape: ", flow_embeds.shape)
        return flow_embeds


class CorrectProjModel(nn.Module):
    """
    Correct the final flow embedding with a linear norm and final projection layer
    """

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


#########################################################################


# Dataset
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, json_file, tokenizer, size=256, 
                 t_drop_rate=0.05, i_drop_rate=0.05, 
                 ti_drop_rate=0.05, dataset_name="ueyes"):
        super().__init__()

        self.tokenizer = tokenizer
        self.size = size
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.dataset_name = dataset_name

        self.data = json.load(open(json_file)) # list of dict: [{"image_file": "1.png", "text": "A dog"}]

        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        # self.clip_image_processor = CLIPImageProcessor()
        

    def __getitem__(self, idx):
        item = self.data[idx] 
        text = item["prompt"]
        image_file = item["target"]
        
        # read image and flow vector
        raw_image = Image.open(image_file)
        image = self.transform(raw_image.convert("RGB"))

        # clip_image = self.clip_image_processor(images=raw_image, return_tensors="pt").pixel_values
        
        # drop
        drop_flow_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_flow_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_flow_embed = 1
        # get text and tokenize
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        # Added for flow-Adapter:
        if self.dataset_name == "ueyes": # NOT Implemented yet
            flow_input = item["flow_input"]
        else:
            flow_input = None

        return {
            "image": image,
            "text_input_ids": text_input_ids,
            # "clip_image": clip_image,
            "drop_flow_embed": drop_flow_embed,
            "flow_input": flow_input
        }


    def __len__(self):
        return len(self.data)
    

def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    # clip_images = torch.cat([example["clip_image"] for example in data], dim=0)
    drop_flow_embeds = [example["drop_flow_embed"] for example in data]

    return {
        "images": images,
        "text_input_ids": text_input_ids,
        # "clip_images": clip_images,
        "drop_flow_embeds": drop_flow_embeds
    }
    

class IPAdapter(torch.nn.Module):
    """IP-Adapter"""
    def __init__(self, unet, correct_proj_model, adapter_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.correct_proj_model = correct_proj_model
        self.adapter_modules = adapter_modules

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, flow_embeds):
        # print("flow_embeds shape: ", flow_embeds.shape)
        # print("encoder_hidden_states shape: ", encoder_hidden_states.shape)
        ip_tokens = self.correct_proj_model(flow_embeds)
        # print("ip_tokens shape: ", ip_tokens.shape)
        
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.correct_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for correct_proj_model and adapter_modules
        self.correct_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.correct_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of correct_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")

 
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default=None,
        help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--data_json_file",
        type=str,
        default="/home/researcher/Documents/dataset/original_datasets/webui_prompts.json",
        help="Training data",
    )

    # Isn't the json file enough? NOPE not any more
    # parser.add_argument(
    #     "--data_root_path",
    #     type=str,
    #     default="",
    #     required=True,
    #     help="Training data root path",
    # )
    # False for Flow-Adapter
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        required=False,
        help="Path to CLIP image encoder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-flow_adapter",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100) # TODO: Change this < 50
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=8000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    
    # Added for flow-Adapter
    parser.add_argument("--dataset_name", type=str, default="everything_else", 
                        help="The name of the dataset to use. [ueyes | everything_else]")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
    

def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    flowAE = FlowEncoder_MLP()
    flow_latenizer = flowAE.encoder

    if args.dataset_name == "ueyes":
        flowAE.load_state_dict(torch.load("/home/researcher/flowAE.pth")) # TODO: Put the right path here
        flow_latenizer.requires_grad_(False)
        eyeFormer = None
    else: 
        flow_latenizer.requires_grad_(True)
        # NOTE: x -> x = eyeFormer(x) -> flowEncoder(x) -> IP-Adapter pipeline 
        config = yaml.load(open("./configs/Tracking.yaml", 'r'), Loader=yaml.Loader)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

        eyeFormer = TrackingTransformer(config = config, init_deit=False)
        checkpointEF = torch.load("/home/researcher/Documents/aryan/asciProject/flowEncoder/weights/checkpoint_19.pth",
                                map_location='cpu')
        state_dict = checkpointEF['model']

        eyeFormer.load_state_dict(state_dict)
        eyeFormer.requires_grad_(False)
        

    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    #ip-adapter
    # correct_proj_model = ImageProjModel(
    #     cross_attention_dim=unet.config.cross_attention_dim,
    #     clip_embeddings_dim=1024,
    #     clip_extra_context_tokens=4,
    # )
    
    # init adapter modules
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim

        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]


        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, 
                                               cross_attention_dim=cross_attention_dim)
            attn_procs[name].load_state_dict(weights)
    
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    
    # with open("IPA_SD_model.txt", "w") as f:
    #     f.write(str(unet))
    # exit()

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    #unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    flow_latenizer.to(accelerator.device, dtype=weight_dtype)
    if eyeFormer is not None:
        eyeFormer.to(accelerator.device, dtype=weight_dtype)
    
    flow_encoder = FlowEncoder(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=1024,
        clip_extra_context_tokens=4,
        flow_latenizer=flow_latenizer
    )
    flow_encoder.requires_grad_(True)
    flow_projection_model = CorrectProjModel(
            cross_attention_dim=unet.config.cross_attention_dim,
            clip_embeddings_dim=1024,
            clip_extra_context_tokens=4
    )
    flow_projection_model.requires_grad_(True)
    ip_adapter = IPAdapter(unet, flow_projection_model, adapter_modules, args.pretrained_ip_adapter_path)
    
    # optimizer
    if args.dataset_name == "ueyes":
        params_to_opt = itertools.chain(
            ip_adapter.correct_proj_model.parameters(),  
            ip_adapter.adapter_modules.parameters())
    else:
        params_to_opt = itertools.chain(flow_encoder.parameters(), 
                                        ip_adapter.correct_proj_model.parameters(),  
                                        ip_adapter.adapter_modules.parameters())
        
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # dataloader
    train_dataset = MyDataset(args.data_json_file, 
                              tokenizer=tokenizer, 
                              size=args.resolution, 
                              dataset_name=args.dataset_name)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    
    # Prepare everything with our `accelerator`.
    ip_adapter, flow_encoder, optimizer, train_dataloader = accelerator.prepare(ip_adapter, 
                                                                                flow_encoder, 
                                                                                optimizer, 
                                                                                train_dataloader)
    
    global_step = 0

    for epoch in range(0, args.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(ip_adapter):
                # Convert images to latent space
                with torch.no_grad():
                    latents = vae.encode(batch["images"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                if args.dataset_name == "everything_else":
                    with torch.no_grad():
                        flow_path = eyeFormer(batch["images"].to(accelerator.device, dtype=weight_dtype))
                else: 
                    # flow_path = batch["flow"]
                    raise NotImplementedError

                flow_embeds = flow_encoder(flow_path) 
                flow_embeds_ = []
                for image_embed, drop_flow_embed in zip(flow_embeds, batch["drop_flow_embeds"]):
                    if drop_flow_embed == 1:
                        flow_embeds_.append(torch.zeros_like(image_embed))
                    else:
                        flow_embeds_.append(image_embed)
                flow_embeds = torch.stack(flow_embeds_)
            
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["text_input_ids"].to(accelerator.device))[0]
                
                noise_pred = ip_adapter(noisy_latents, timesteps, encoder_hidden_states, flow_embeds)
        
                loss_noise = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                # TODO: loss_flow 
                # Need to sample an image from the diffusion process and then calculate the loss
                # sampled_image = noise_scheduler.sample_image(noisy_latents, timesteps) # TODO: Check if this is correct!
                
                # loss_flow = get_soft_dtw_Loss(flow_proj_model.eyeFormer(sampled_image), batch["flow_input"])
                loss_flow = 0.
                # Weight scalers
                lambda_flow = .5
                loss = loss_noise + lambda_flow * loss_flow
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
                
                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                if (step+1) % 200 == 0 and accelerator.is_main_process:
                    print("Epoch {}, step {}, data_time: {}, time: {}, step_loss: {}".format(
                        epoch, step, load_data_time, time.perf_counter() - begin, avg_loss))
            
            global_step += 1
            
            if global_step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path, safe_serialization=False)
                custom_model_save_dict = {
                    "flow_proj": ip_adapter.correct_proj_model.state_dict(),
                    "ip_adapter": ip_adapter.adapter_modules.state_dict(),
                    "flow_encoder": flow_encoder.state_dict(),
                }
                torch.save(custom_model_save_dict, 
                           os.path.join(args.output_dir, f"custom_model_checkpoint-{global_step}.pt"))
                
            
            begin = time.perf_counter()
      
                
if __name__ == "__main__":
    main()    
