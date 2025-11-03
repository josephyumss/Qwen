import torch
import argparse
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device : {device}")
dtype = torch.float16 if device.type == "cuda" else torch.float32

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, default='4B')
parser.add_argument("--img_path", type=str, nargs="+", help="one or more image paths")
parser.add_argument("--prompt", type=str)
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
    model_name = "Qwen/Qwen3-VL-8B-think"
else:
    raise ValueError

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
model.eval()

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "심신 유명론에 대해서 설명해줘"}
]

images = [Image.open(p).convert("RGB") for p in args.img_path]

text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

inputs = tokenizer(text, return_tensors="pt").to(device)

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
