# 匯入必要的函式庫
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO
import evaluate
from tqdm import tqdm

# 設定使用的裝置（GPU 有的話優先用 GPU）
device = "cuda" if torch.cuda.is_available() else "cpu"

# 載入三個自動化評估指標：BLEU、ROUGE、METEOR
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")


# 載入 MSCOCO 2014 測試資料集（此為 5k retrieval 版本）
dataset = load_dataset("nlphuji/mscoco_2014_5k_test_image_text_retrieval")["test"]

# 儲存所有模型生成的描述與對應 ground-truth caption
all_preds, all_refs = [], []

# 儲存前五筆的 caption，作為範例展示
first_five = []

# 載入 BLIP 模型與對應的前處理器（captioning base 版本）
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# 主迴圈：對每張圖片生成 caption 並儲存結果
for i, data in tqdm(enumerate(dataset), total=len(dataset)):
    # if i >= 10:
    #     break
    img_data = data["image"]

    # 若 image 是 URL（字串），則嘗試下載並開啟為 PIL 圖片
    if isinstance(img_data, str):
        try:
            response = requests.get(img_data)
            raw_image = Image.open(BytesIO(response.content)).convert("RGB")
        except:
            continue  # 若圖片讀取失敗則跳過

    # 若 image 已經是 PIL 圖片，直接使用
    elif isinstance(img_data, Image.Image):
        raw_image = img_data.convert("RGB")
    else:
        continue  # 其他型態的資料略過

    # 將圖片轉成模型輸入格式（tensor），並移至 GPU（如可用）
    inputs = processor(raw_image, return_tensors="pt").to(device)

    # 生成 caption（採用 greedy decoding）
    output = model.generate(**inputs)

    # 解碼為文字（去除特殊符號）
    caption = processor.decode(output[0], skip_special_tokens=True)

    # 取得對應的 ground-truth captions（通常是 5 筆）
    refs = data["caption"]

    # 儲存預測與標註，供評估使用
    all_preds.append(caption)
    all_refs.append(refs)

    # 儲存前五筆結果作為範例展示
    if len(first_five) < 5:
        first_five.append((caption, refs))

# 計算三種自動化評估指標的結果
results = {
    "bleu": bleu.compute(predictions=all_preds, references=all_refs),
    "rouge": rouge.compute(predictions=all_preds, references=all_refs),
    "meteor": meteor.compute(predictions=all_preds, references=all_refs),
}

# 將評估結果與前五張圖片的 caption 組成報告文字
summary = (
    "\n===== BLIP on MSCOCO 評估結果 =====\n"
    f"BLEU Score: {results['bleu']['bleu']:.4f}\n"
    f"ROUGE-1: {results['rouge']['rouge1']:.4f}\n"
    f"ROUGE-2: {results['rouge']['rouge2']:.4f}\n"
    f"METEOR Score: {results['meteor']['meteor']:.4f}\n\n"
    "===== 前五張圖片的 Caption =====\n\n"
)

# 加入前五筆的生成描述與標註對照
for i, (gen, ref) in enumerate(first_five):
    summary += f"[{i+1}]\nGenerated: {gen}\nGround Truth: {ref}\n\n"

# 螢幕顯示總結內容
print(summary)

# 將結果寫入檔案（附加模式）
with open("evaluation_results_blip_mscoco.txt", "a", encoding="utf-8") as f:
    f.write(summary)

