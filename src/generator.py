import torch
from transformers import AutoTokenizer

from src.models.tiny_shakespeare_module import TinyShakespeareModule

ckpt_path = (
    "/home/apujari1/Documents/lightning-hydra-uv-template/"
    "lightning-hydra-uv-template/logs/train/runs/2026-04-19_01-57-17/"
    "checkpoints/last.ckpt"
)

model = TinyShakespeareModule.load_from_checkpoint(ckpt_path)
model.eval()

device = next(model.parameters()).device

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/smolLM-135M")

prompts = [
    "CORIOLANUS:",
    "MENENIUS:",
    "First Citizen:",
]

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    with torch.no_grad():
        output = model.net.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
        )

    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\n--- Prompt: {prompt} ---\n{generated}\n")
