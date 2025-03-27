import os
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor, AutoModelForCausalLM, GenerationConfig, CLIPTokenizer
)
from diffusers import StableDiffusion3Pipeline

# === 使用者設定 ===
test_mode = False                
max_images = 3 if test_mode else 100

input_dir = "./content_image"
output_dir = "./output2-1"             
os.makedirs(output_dir, exist_ok=True)

# === 載入 Phi-4 MLLM ===
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

# === 載入 Stable Diffusion 3 & CLIP tokenizer ===
sd3_pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    text_encoder_3=None,
    tokenizer_3=None,
    torch_dtype=torch.float16
)
sd3_pipe.enable_model_cpu_offload() 

clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# === Prompt 格式設定 ===
user_prompt = "<|user|>"
assistant_prompt = "<|assistant|>"
prompt_suffix = "<|end|>"

# === Step 1: 針對人像產生描述 ===
def get_person_description(image: Image.Image) -> str:
    instruction = "Use a sentence to describe the person’s appearance, including hairstyle, face shape, and clothing."
    prompt = f"{user_prompt}<|image_1|>{instruction}{prompt_suffix}{assistant_prompt}"
    inputs = phi4_processor(text=prompt, images=image, return_tensors="pt").to("cuda")

    generate_ids = phi4_model.generate(
        **inputs,
        max_new_tokens=45,
        generation_config=phi4_gen_config
    )
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    description = phi4_processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
    return description.strip()


# === Step 2: 用 Stable Diffusion 3 生成圖 ===
def generate_stylized_image(prompt: str) -> Image.Image:
    result = sd3_pipe(
        prompt=prompt,
        negative_prompt="",
        num_inference_steps=28,
        guidance_scale=7.0
    )
    return result.images[0]

# === 主流程 ===
image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".jpg")])[:max_images]
prompt_log_path = os.path.join(output_dir, "prompt_log.txt")

with open(prompt_log_path, "w", encoding="utf-8") as logf:
    for filename in tqdm(image_files, desc="Generating Stylized Images"):
        try:
            input_path = os.path.join(input_dir, filename)
            image = Image.open(input_path).convert("RGB")

           # Step 1: 產生人像描述
            content_description = get_person_description(image)

            # Step 2: 組合完整 multi-section prompt，並強制裁切
            style_description = "a cartoon character in Peanuts comic style, flat colors, thick outlines, simple shapes, cute and minimalist"
            full_prompt = f"content: {content_description} style: {style_description}"

            max_total_tokens = 77
            while True:
                tokenized = sd3_pipe.tokenizer(full_prompt, return_tensors=None)
                if len(tokenized["input_ids"]) <= max_total_tokens:
                    break
                # 裁掉描述句的最後一個詞
                content_description = " ".join(content_description.split(" ")[:-1])
                full_prompt = f"content: {content_description} style: {style_description}"

            # Step 3: 用 SD3 產生圖像
            cartoon = generate_stylized_image(full_prompt)

            # Step 4: Resize 並儲存
            cartoon = cartoon.resize((224, 224))
            cartoon.save(os.path.join(output_dir, filename))

            # Step 5: 記錄 log
            logf.write(f"{filename}\tDesc: {content_description}\tPrompt: {full_prompt}\n")

        except Exception as e:
            print(f"[ERROR] {filename}: {e}")

print("\n✅ Done! Images saved in ./output/")
