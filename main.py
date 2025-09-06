from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import io
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_ID = os.environ.get("MODEL_ID", "./models/convnextv2-base-22k-224")
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def pretty_label(x: str) -> str:
    x = x.replace("_", " ").strip()
    parts = [p.strip() for p in x.split(",")]
    s = " / ".join(parts)
    return s[:1].upper() + s[1:]

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><title>Image Classifier</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{color-scheme:dark light}
body{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto;padding:40px;background:#0b0b0c;color:#eaeaea}
.card{max-width:680px;margin:auto;background:#151518;border:1px solid #26262b;border-radius:16px;padding:24px;box-shadow:0 10px 30px rgba(0,0,0,.35)}
h1{font-size:20px;margin:0 0 16px}
#preview{width:100%;max-height:360px;object-fit:contain;border-radius:12px;border:1px dashed #333;margin-top:12px}
button{background:#4f46e5;color:white;border:none;padding:10px 16px;border-radius:10px;cursor:pointer}
button:disabled{opacity:.6;cursor:not-allowed}
.row{display:flex;gap:12px;align-items:center}
.badge{display:inline-block;background:#22242c;border:1px solid #2e3240;padding:6px 10px;border-radius:999px;font-size:12px}
.result{margin-top:16px;padding:16px;background:#101114;border:1px solid #24262f;border-radius:12px}
.bar{height:10px;background:#1e2533;border-radius:999px;overflow:hidden}
.fill{height:100%;background:#10b981;width:0%}
.small{opacity:.8;font-size:12px}
</style>
</head>
<body>
<div class="card">
  <h1>Upload image for classification</h1>
  <div class="row">
    <input type="file" id="file" accept="image/*">
    <button id="send" onclick="send()">Classify</button>
    <span id="meta" class="badge"></span>
  </div>
  <img id="preview" />
  <div id="out" class="result" style="display:none">
    <div id="label" style="font-size:22px;font-weight:700"></div>
    <div class="bar" style="margin:10px 0 6px"><div id="fill" class="fill"></div></div>
    <div id="conf" class="small"></div>
    <div id="other" class="small"></div>
  </div>
</div>
<script>
const f=document.getElementById("file");
const img=document.getElementById("preview");
const out=document.getElementById("out");
const lab=document.getElementById("label");
const fill=document.getElementById("fill");
const conf=document.getElementById("conf");
const other=document.getElementById("other");
const btn=document.getElementById("send");
const meta=document.getElementById("meta");

f.addEventListener("change",e=>{
  const file=f.files[0];
  if(!file){img.src="";return}
  const url=URL.createObjectURL(file);
  img.src=url;
  meta.textContent=file.name+" • "+Math.round(file.size/1024)+" KB";
});

async function send(){
  const file=f.files[0];
  if(!file){alert("Select an image.");return}
  btn.disabled=true;
  out.style.display="none";
  const fd=new FormData(); fd.append("file", file);
  const t0=performance.now();
  const res=await fetch("/classify", {method:"POST", body:fd});
  const t1=performance.now();
  const json=await res.json();
  btn.disabled=false;
  if(!res.ok){alert(json.detail||"Error");return}
  out.style.display="block";
  lab.textContent=json.pretty_label;
  fill.style.width=Math.round(json.confidence*100)+"%";
  conf.textContent="Confidence: "+(json.confidence*100).toFixed(1)+"% • Model: "+json.model_id+" • "+(t1-t0).toFixed(0)+" ms";
  other.textContent=json.message;
}
</script>
</body>
</html>
"""

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(status_code=415, detail="Only JPEG/PNG/WEBP images are supported.")
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=device.type=="cuda"):
            outputs = model(**inputs)
            probs = outputs.logits.softmax(dim=1)[0]
            score, idx = torch.max(probs, dim=0)
    label_raw = model.config.id2label[idx.item()]
    label = pretty_label(label_raw)
    msg = f"Looks like {label.lower()}." if float(score) >= 0.55 else f"Not sure. Closest match: {label.lower()}."
    return JSONResponse({
        "filename": file.filename,
        "model_id": MODEL_ID,
        "label": label_raw,
        "pretty_label": label,
        "confidence": float(score),
        "message": msg
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
