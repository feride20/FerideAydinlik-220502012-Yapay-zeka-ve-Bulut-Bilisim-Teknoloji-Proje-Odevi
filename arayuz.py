import torch
import gradio as gr
import numpy as np
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification


MODEL_DIR = r"./fruit_vit_out/checkpoint-174"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = ViTImageProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
model = ViTForImageClassification.from_pretrained(MODEL_DIR, local_files_only=True).to(device)
model.eval()

id2label = model.config.id2label

TR_LABELS = {
    "rottenbanana": "Çürük Muz",
    "freshbanana": "Taze Muz",
    "rottenapples": "Çürük Elma",
    "freshapples": "Taze Elma",
    "rottenoranges": "Çürük Portakal",
    "freshoranges": "Taze Portakal",
}

def get_label(idx: int) -> str:
    if not id2label:
        return str(idx)
    first_key = next(iter(id2label.keys()))
    raw = id2label.get(str(idx)) if isinstance(first_key, str) else id2label.get(idx)
    if raw is None:
        raw = str(idx)
    return TR_LABELS.get(raw, raw)


def preprocess_image(image: Image.Image):
    if image is None:
        return None, " Lütfen bir görsel yükleyin."

    image = image.convert("RGB")

    size = processor.size.get("shortest_edge", 224) if isinstance(processor.size, dict) else 224
    crop = processor.crop_size.get("height", 224) if hasattr(processor, "crop_size") and isinstance(processor.crop_size, dict) else 224

    w, h = image.size
    if w < h:
        new_w = size
        new_h = int(h * (size / w))
    else:
        new_h = size
        new_w = int(w * (size / h))

    image_resized = image.resize((new_w, new_h))

    left = (new_w - crop) // 2
    top = (new_h - crop) // 2
    right = left + crop
    bottom = top + crop
    image_cropped = image_resized.crop((left, top, right, bottom))

    return image_cropped, f" Ön işleme: Resize({size}) + CenterCrop({crop}x{crop})"


def predict(image_processed: Image.Image):
    if image_processed is None:
        return {}, " Önce bir görsel yükleyin."

    inputs = processor(images=image_processed, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_id = int(np.argmax(probs))
    pred_label = get_label(pred_id)
    confidence = float(probs[pred_id])

    topk = np.argsort(probs)[::-1][:5]
    scores = {get_label(int(i)): float(probs[int(i)]) for i in topk}

    sonuc_metin = f" Tahmin: {pred_label}\n Güven: %{confidence*100:.2f}"
    return scores, sonuc_metin

def clear_all():
    return None, None, "", {}, ""


CUSTOM_CSS = """
/* Genel */
.gradio-container {max-width: 1200px !important;}
footer, #footer {display:none !important;}
/* Üstteki gereksiz boşlukları azalt */
.block {border-radius: 14px !important;}
/* Başlık bar */
.headerbar{
  display:flex; align-items:center; justify-content:space-between;
  padding:14px 16px; border:1px solid rgba(255,255,255,.08);
  border-radius:14px; margin-bottom:12px;
}
.headerbar .title{
  font-size:20px; font-weight:700;
}
.headerbar .subtitle{
  opacity:.75; font-size:12px;
}
/* Butonlar */
button{
  border-radius:12px !important;
  font-weight:600 !important;
}
/* Sonuç kutusu */
textarea {font-size:14px !important;}
"""

with gr.Blocks(css=CUSTOM_CSS, title="Meyve Sınıflandırma") as demo:

    
    gr.HTML("""
    <div class="headerbar">
      <div>
        <div class="title">Meyve Sınıflandırma</div>
        <div class="subtitle">Görsel yükle → ön işlem → tahmin</div>
      </div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            img_in = gr.Image(type="pil", label="📤 Görsel Yükle", height=360)
        with gr.Column(scale=1):
            img_preview = gr.Image(type="pil", label="🧩 Ön İşlem Önizleme (Resize + Crop)", height=360)

    preprocess_info = gr.Textbox(
        label="ℹ️ Ön İşleme Bilgisi",
        interactive=False,
        placeholder="Görsel yükleyince otomatik ön işleme uygulanır."
    )

    with gr.Row():
        btn_preprocess = gr.Button(" Ön İşleme Uygula", variant="secondary")
        btn_predict = gr.Button(" Tahmin Et", variant="primary")
        btn_clear = gr.Button(" Temizle", variant="stop")

    with gr.Row():
        top5_out = gr.Label(num_top_classes=5, label=" Top 5 Sonuç")
        result_text = gr.Textbox(label=" Sonuç (Sınıf + Güven)", lines=3)

    
    img_in.change(fn=preprocess_image, inputs=img_in, outputs=[img_preview, preprocess_info])

    
    btn_preprocess.click(fn=preprocess_image, inputs=img_in, outputs=[img_preview, preprocess_info])
    btn_predict.click(fn=predict, inputs=img_preview, outputs=[top5_out, result_text])
    btn_clear.click(fn=clear_all, inputs=[], outputs=[img_in, img_preview, preprocess_info, top5_out, result_text])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
