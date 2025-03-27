import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
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

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

dataset = load_dataset("nlphuji/flickr30k")["test"]


all_preds, all_refs = [], []
first_five = []

for i, data in tqdm(enumerate(dataset), total=len(dataset)):

    img_data = data["image"]
    if isinstance(img_data, str):
        try:
            response = requests.get(img_data)
            raw_image = Image.open(BytesIO(response.content)).convert("RGB")
        except:
            continue
    elif isinstance(img_data, Image.Image):
        raw_image = img_data.convert("RGB")
    else:
        continue

    inputs = processor(raw_image, return_tensors="pt").to(device)
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    refs = data["caption"]

    all_preds.append(caption)
    all_refs.append(refs)

    if len(first_five) < 5:
        first_five.append((caption, refs))

# 評估
results = {
    "bleu": bleu.compute(predictions=all_preds, references=all_refs),
    "rouge": rouge.compute(predictions=all_preds, references=all_refs),
    "meteor": meteor.compute(predictions=all_preds, references=all_refs),
}

summary = (
    "\n===== BLIP on Flickr30k 評估結果 =====	\n"
    f"BLEU Score: {results['bleu']['bleu']:.4f}\n"
    f"ROUGE-1: {results['rouge']['rouge1']:.4f}\n"
    f"ROUGE-2: {results['rouge']['rouge2']:.4f}\n"
    f"METEOR Score: {results['meteor']['meteor']:.4f}\n\n"
    "===== 前五張圖片的 Caption =====\n\n"
)

for i, (gen, ref) in enumerate(first_five):
    summary += f"[{i+1}]\nGenerated: {gen}\nGround Truth: {ref}\n\n"

print(summary)

with open("evaluation_results_blip_flickr30k.txt", "a", encoding="utf-8") as f:
    f.write(summary)
