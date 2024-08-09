#!/usr/bin/env python3
import torch
from PIL import Image
import os
from tqdm import tqdm
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import json

def use_llava_next(imagepath, processor, model):
    """
        Llava 1.6 7b mistral
    """

    # prepare image and text prompt, using the appropriate prompt template
    
    # Logic:
    # While (done with all text boxes/labels):
    	# Ask what to write from Llava in that box (Llava input: real UI + mask where you're asking + prompt: hand-crafted)
    	# Ask TextDiffuser to put it there.
    		# NOTE: fComposition/Harmonization? 
    		
    		
    # For third ControlNet branch: 
    # Maybe optimize original ControlNet branches with a Character Aware loss OR UIClip loss OR (Ablation) just pre-trained for initial experiments.
    image = Image.open(imagepath).convert('RGB')
    prompt = "[INST] <image>\n Describe the semantic styles, colors and UI elements in this mobile UI in a single paragraph. [/INST]"
    inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")

    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=512)

    return processor.decode(output[0], skip_special_tokens=True)


# NOTE: Old. Use Llava NeXT instead.
def use_llava_1_5():
    """
        Llava-1.5 7b
    """
    from PIL import Image
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    prompt = "USER: <image>\nDescribe the styles, font, colors and semantics in this mobile user interface image. ASSISTANT:"
    image = Image.open("./Rico/combined/0.jpg").convert('RGB')

    inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")

    # Generate
    output = model.generate(**inputs, max_new_tokens=1024)
    print("\nLlava's response:\n", processor.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    
    # TODO: get captions for all IOS images to finetune ControlNet
    BASE = "/home/researcher/Documents/dataset/original_datasets/Graph4GUI_dataset/UI_images/"
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

    model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", 
                                                            torch_dtype=torch.float16, 
                                                            low_cpu_mem_usage=True)  
    model.generation_config.pad_token_id = model.generation_config.eos_token_id # Suppress eos token warning
    model.to("cuda:0")
    # use_llava_next(textDiffuser)
    with open("graph4gui_prompts.json", "w") as f:
        all_dicts = []
        for i in (pbar:=tqdm(os.listdir(BASE))):
            if i.endswith(".jpg") or i.endswith(".png"):
                # pbar.set_description(f"Processing {i}")
                response = use_llava_next(BASE + i, processor, model)
                clean_response = response[response.find("[/INST]")+8:] 
                # print(clean_response)
                dict_this = {
                    "source": f"/home/researcher/Documents/dataset/original_datasets/Graph4GUI_dataset/semantic_maps/{i}", 
                    "target": f"{BASE+i}", 
                    "prompt": clean_response
                }
                all_dicts.append(dict_this)

        json.dump(all_dicts, f)
    # use_llava_1_5()
