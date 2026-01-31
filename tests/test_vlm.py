"""
Step 1: SmolVLM-500M text inference test
SmolVLM is a lightweight VLM designed for edge devices, ideal for Orin Nano 8GB
"""
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"

print("=" * 50)
print(f"Model: {MODEL_ID}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("=" * 50)

print("\nLoading processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

print("Loading model...")
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print(f"Model loaded on: {next(model.parameters()).device}")
print(f"Model dtype: {next(model.parameters()).dtype}")

# Simple text test
messages = [
    {"role": "user", "content": [{"type": "text", "text": "Say hello in Chinese in one short sentence."}]}
]

print("\nApplying chat template...")
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

print("Generating response...")
with torch.inference_mode():
    outputs = model.generate(
        inputs,
        max_new_tokens=32,
        do_sample=False,
    )

response = processor.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

print("\n" + "=" * 50)
print("Response:")
print(response)
print("=" * 50)
print("\n✅ Text inference test completed!")
