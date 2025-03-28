# 匯入必要的套件
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from datasets import load_dataset
from PIL import Image
import evaluate
from tqdm import tqdm

# 設定裝置為 GPU（如可用），否則使用 CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 載入三個常見的自然語言生成評估指標
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")

# 載入 Flickr30k 測試資料集
dataset = load_dataset("nlphuji/flickr30k")["test"]

# 儲存生成與標註結果，以及前五張圖片對應的 caption
all_preds, all_refs = [], []
first_five = []

# 指定模型名稱（Phi-4 多模態模型）
model_path = "microsoft/Phi-4-multimodal-instruct"
# 載入模型的 Processor（處理圖像與文字的前處理工具）
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
# 載入 Phi-4 模型（使用 flash attention、device_map，自動分配 GPU）
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cuda",  # 支援多卡（如單卡仍可用）
    torch_dtype="auto",
    trust_remote_code=True,
    _attn_implementation='flash_attention_2'
).cuda()
# 載入生成設定（例如 temperature, top_k 等）
generation_config = GenerationConfig.from_pretrained(model_path)
batch_size = 6
# 禁用 gradient 計算，加快推論
with torch.no_grad():
    # 每次取一批資料
    for i in tqdm(range(0, len(dataset), batch_size), total=(len(dataset) + batch_size - 1) // batch_size):
        # 可選：測試時可以限制處理資料量
        # if i >= 1:
        #     break

        # 抓取當前批次的圖片與標註
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        images = batch["image"]

        # 為每張圖片建立 prompt（符合 Phi-4 格式）
        prompts = ['<|user|><|image_1|>What is shown in this image?<|end|><|assistant|>'] * len(images)

        # 前處理文字與圖片成為模型輸入（tensor 格式）
        inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(device)

        # 模型生成輸出（回答圖片描述）
        out = model.generate(
            **inputs,
            max_new_tokens=512,
            generation_config=generation_config
        )

        # 去除 prompt 部分，只保留模型生成的新文字
        out = out[:, inputs["input_ids"].shape[1]:]

        # 解碼成為可讀文字（跳過特殊符號）
        captions = processor.batch_decode(out, skip_special_tokens=True)

        # 儲存結果供後續計算評估指標
        all_preds.extend(captions)
        all_refs.extend(batch["caption"])

        # 前五張圖片的生成與標註結果，用來展示
        if len(first_five) < 5:
            for cap, ref in zip(captions, batch["caption"]):
                if len(first_five) < 5:
                    first_five.append((cap, ref))

# 使用 BLEU、ROUGE、METEOR 對 caption 做整體評估
results = {
    "bleu": bleu.compute(predictions=all_preds, references=all_refs),
    "rouge": rouge.compute(predictions=all_preds, references=all_refs),
    "meteor": meteor.compute(predictions=all_preds, references=all_refs),
}

# 組裝結果摘要
summary = (
    "\n===== Phi-4 on Flickr30k 評估結果 =====\n"
    f"BLEU Score: {results['bleu']['bleu']:.4f}\n"
    f"ROUGE-1: {results['rouge']['rouge1']:.4f}\n"
    f"ROUGE-2: {results['rouge']['rouge2']:.4f}\n"
    f"METEOR Score: {results['meteor']['meteor']:.4f}\n\n"
    "===== 前五張圖片的 Caption =====\n\n"
)

# 加入前五筆的生成 vs. ground truth 對照
for i, (gen, ref) in enumerate(first_five):
    summary += f"[{i+1}]\nGenerated: {gen}\nGround Truth: {ref}\n\n"

# 螢幕顯示結果
print(summary)

# 將結果寫入文字檔（附加模式）
with open("evaluation_results_phi4_flickr30k.txt", "a", encoding="utf-8") as f:
    f.write(summary)
