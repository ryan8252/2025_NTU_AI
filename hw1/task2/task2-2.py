import os
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor, AutoModelForCausalLM, GenerationConfig, CLIPTokenizer
)
from diffusers import AutoPipelineForImage2Image

# === 使用者設定 ===
test_mode = False
max_images = 3 if test_mode else 100

input_dir = "./content_image"
output_dir = "./output2-2"  
os.makedirs(output_dir, exist_ok=True)

# === 載入 Phi-4 ===
phi4_model_path = "microsoft/Phi-4-multimodal-instruct"
phi4_processor = AutoProcessor.from_pretrained(phi4_model_path, trust_remote_code=True)
phi4_model = AutoModelForCausalLM.from_pretrained(
    phi4_model_path,
    torch_dtype="auto",
    trust_remote_code=True,
    device_map="cuda",
    _attn_implementation="flash_attention_2"
)
phi4_gen_config = GenerationConfig.from_pretrained(phi4_model_path)

# === 載入 SD v1-5 (AutoPipeline) ===
pipe = AutoPipelineForImage2Image.from_pretrained(
    "sd-legacy/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    variant="fp16",
    safety_checker=None,
).to("cuda")
pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()

# === CLIP tokenizer（判斷長度用）===
clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# === Prompt 結構設定 ===
user_prompt = "<|user|>"
assistant_prompt = "<|assistant|>"
prompt_suffix = "<|end|>"

# ✅ Snoopy 風格描述（精簡版）
# style_description = "a cartoon character in Peanuts comic style, flat colors, thick outlines, simple shapes, cute and minimalist"
style_description = "Peanuts comic style"

# === Step 1: 用 Phi-4 描述人像特徵 ===
def get_person_description(image: Image.Image) -> str:
    instruction = "Describe the person’s appearance in detail, including hairstyle, face shape, and clothing."
    prompt = f"{user_prompt}<|image_1|>{instruction}{prompt_suffix}{assistant_prompt}"
    inputs = phi4_processor(text=prompt, images=image, return_tensors="pt").to("cuda")

    generate_ids = phi4_model.generate(
        **inputs,
        max_new_tokens=512,
        generation_config=phi4_gen_config
    )
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    description = phi4_processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
    return description.strip()

# === Step 2: 建構 prompt + 裁切內容 ===
def build_safe_prompt(content_description: str, max_tokens=77) -> str:
    while True:
        full_prompt = f"Content: {content_description}\nStyle: {style_description}"
        tokenized = pipe.tokenizer(full_prompt, return_tensors=None)
        if len(tokenized["input_ids"]) <= max_tokens:
            return full_prompt
        content_description = " ".join(content_description.split(" ")[:-1])  # 刪掉最後一個詞

# === 主流程 ===
image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".jpg")])[:max_images]
prompt_log_path = os.path.join(output_dir, "prompt_log.txt")

with open(prompt_log_path, "w", encoding="utf-8") as logf:
    for filename in tqdm(image_files, desc="Generating (Image-to-Image)"):
        try:
            input_path = os.path.join(input_dir, filename)
            image = Image.open(input_path).convert("RGB")

            # Step 1: 用 Phi-4 生內容描述
            content_desc = get_person_description(image)

            # Step 2: 組合 prompt（並裁切）
            final_prompt = build_safe_prompt(content_desc)

            # Step 3: Image-to-Image 推論
            cartoon = pipe(
                prompt=final_prompt,
                image=image.resize((512, 512)),
                strength=0.7,
                guidance_scale=7.5
            ).images[0]

            # Step 4: Resize 輸出圖為 224x224 並儲存
            cartoon = cartoon.resize((224, 224))
            cartoon.save(os.path.join(output_dir, filename))

            # Step 5: 紀錄 log
            logf.write(f"{filename}\tDesc: {content_desc}\tPrompt: {final_prompt.strip()}\n")

        except Exception as e:
            print(f"[ERROR] {filename}: {e}")

print("\n✅ All done! Stylized images saved to ./output/")



