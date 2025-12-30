import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "lmsys/vicuna-7b-v1.5"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

prompt = "### User:\nTell me why some people have heart disease.\n\n### Assistant:\n"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=160,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.decode(out[0], skip_special_tokens=True))