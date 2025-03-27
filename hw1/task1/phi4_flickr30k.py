import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from datasets import load_dataset
from PIL import Image
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

dataset = load_dataset("nlphuji/flickr30k")["test"]
batch_size = 6

all_preds, all_refs = [], []
first_five = []

with torch.no_grad():
    for i in tqdm(range(0, len(dataset), batch_size), total=(len(dataset) + batch_size - 1) // batch_size):
        # if i >= 10:
        #     break

        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        images = batch["image"]
        prompts = ['<|user|><|image_1|>What is shown in this image?<|end|><|assistant|>'] * len(images)

        inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(device)
        out = model.generate(**inputs, max_new_tokens=512, generation_config=generation_config)
        out = out[:, inputs["input_ids"].shape[1]:]
        captions = processor.batch_decode(out, skip_special_tokens=True)

        all_preds.extend(captions)
        all_refs.extend(batch["caption"])

        if len(first_five) < 5:
            for cap, ref in zip(captions, batch["caption"]):
                if len(first_five) < 5:
                    first_five.append((cap, ref))

results = {
    "bleu": bleu.compute(predictions=all_preds, references=all_refs),
    "rouge": rouge.compute(predictions=all_preds, references=all_refs),
    "meteor": meteor.compute(predictions=all_preds, references=all_refs),
}

summary = (
    "\n===== Phi-4 on Flickr30k 評估結果 =====\n"
    f"BLEU Score: {results['bleu']['bleu']:.4f}\n"
    f"ROUGE-1: {results['rouge']['rouge1']:.4f}\n"
    f"ROUGE-2: {results['rouge']['rouge2']:.4f}\n"
    f"METEOR Score: {results['meteor']['meteor']:.4f}\n\n"
    "===== 前五張圖片的 Caption =====\n\n"
)

for i, (gen, ref) in enumerate(first_five):
    summary += f"[{i+1}]\nGenerated: {gen}\nGround Truth: {ref}\n\n"

print(summary)

with open("evaluation_results_phi4_flickr30k.txt", "a", encoding="utf-8") as f:
    f.write(summary)
