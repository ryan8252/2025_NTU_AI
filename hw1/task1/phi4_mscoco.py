import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO
import evaluate
from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")

model_path = "microsoft/Phi-4-multimodal-instruct"
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
    _attn_implementation='flash_attention_2'
).cuda()
generation_config = GenerationConfig.from_pretrained(model_path)


dataset = load_dataset("nlphuji/mscoco_2014_5k_test_image_text_retrieval")["test"]


# 儲存結果
all_generated_captions = []
all_reference_captions = []
first_five = []


with torch.no_grad():
    for i, data in tqdm(enumerate(dataset), total=len(dataset)):
        # if i >= 10:
        #     break

        img_data = data["image"]
        if isinstance(img_data, str):
            try:
                response = requests.get(img_data)
                response.raise_for_status()
                raw_image = Image.open(BytesIO(response.content)).convert("RGB")
            except Exception as e:
                print(f"讀取圖片失敗: {e}")
                continue
        elif isinstance(img_data, Image.Image):
            raw_image = img_data.convert("RGB")
        else:
            continue

        prompt = "<|user|><|image_1|>What is shown in this image?<|end|><|assistant|>"
        inputs = processor(text=prompt, images=raw_image, return_tensors="pt").to(device)

        out = model.generate(**inputs, max_new_tokens=512, generation_config=generation_config)
        out = out[:, inputs["input_ids"].shape[1]:]
        caption = processor.batch_decode(out, skip_special_tokens=True)[0]

        refs = data["caption"]
        all_generated_captions.append(caption)
        all_reference_captions.append(refs)

        if len(first_five) < 5:
            first_five.append((caption, refs))

# 計算評分
results = {
    "bleu": bleu.compute(predictions=all_generated_captions, references=all_reference_captions),
    "rouge": rouge.compute(predictions=all_generated_captions, references=all_reference_captions),
    "meteor": meteor.compute(predictions=all_generated_captions, references=all_reference_captions),
}

# 組裝輸出
summary = (
    "\n===== Phi-4 on MSCOCO 評估結果 =====\n"
    f"BLEU Score: {results['bleu']['bleu']:.4f}\n"
    f"ROUGE-1: {results['rouge']['rouge1']:.4f}\n"
    f"ROUGE-2: {results['rouge']['rouge2']:.4f}\n"
    f"METEOR Score: {results['meteor']['meteor']:.4f}\n\n"
    "===== 前五張圖片的 Caption =====\n\n"
)

for i, (gen, ref) in enumerate(first_five):
    summary += f"[{i+1}]\nGenerated: {gen}\nGround Truth: {ref}\n\n"

# 印出並儲存
print(summary)
with open("evaluation_results_phi4_mscoco.txt", "a", encoding="utf-8") as f:
    f.write(summary)
