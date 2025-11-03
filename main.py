import torch
import argparse
from PIL import Image
from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoProcessor
import time
import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device : {device}")
dtype = torch.float16 if device.type == "cuda" else torch.float32

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, default='4B-inst')
parser.add_argument("--img_path", type=str, nargs="+", help="one or more image paths")
parser.add_argument("--prompt", type=str, default="What is Qwen?")
args = parser.parse_args()

# if args.version == '4B':
#     model_name = "Qwen/Qwen3-4B-Instruct-2507"  
if args.version == '2B-inst':
    model_name = "Qwen/Qwen3-VL-2B-Instruct" 
elif args.version == '2B-think':
    model_name = "Qwen/Qwen3-VL-2B-thinking"  
elif args.version == '4B-inst':
    model_name = "Qwen/Qwen3-VL-4B-Instruct"
elif args.version == '4B-think':
    model_name = "Qwen/Qwen3-VL-4B-thinking"
elif args.version == '8B-inst':
    model_name = "Qwen/Qwen3-VL-8B-Instruct"
elif args.version == '8B-think':
    model_name = "Qwen/Qwen3-VL-8B-thinking"
elif args.version == '32B-inst':
    model_name = "Qwen/Qwen3-VL-32B-Instruct"
elif args.version == '32B-think':
    model_name = "Qwen/Qwen3-VL-32B-thinking"
else:
    raise ValueError

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
model = AutoModelForImageTextToText.from_pretrained(model_name, dtype=dtype,trust_remote_code=True).to(device)
model.eval()

prompt = args.prompt
user_content = [{"type": "text", "text": prompt}]

if args.img_path is not None:
    images = [Image.open(p).convert("RGB") for p in args.img_path]

    for img in images:
        user_content.append({"type": "image", "image": img})
else :
    images = None

messages = [
    {"role": "system", "content": "You are a helpful vision-language assistant."},
    {"role": "user",   "content": user_content},
]

prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Processor packs text + images into tensors
inputs = processor(text=[prompt_text], images=images if images else None, return_tensors="pt")
inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

gen_tic = time.time()
with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256, # answer max length
        temperature=0.7, # general answer < 1.0 < creative answer
        do_sample=True # sample tokens stochastically / False is greedy token selection
    )
    utils.print_cuda_mem(note="Generation")
gen_toc = time.time()

print(processor.batch_decode(outputs, skip_special_tokens=True)[0])
print(f"Inference took {(gen_toc-gen_tic):.2f} second")