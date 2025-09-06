# Vision Classification API (FastAPI + ConvNeXtV2)

A production-ready image classification API built with **FastAPI** and **Hugging Face Transformers**, using the **ConvNeXtV2** model family. It includes a minimal, elegant web UI for quick testing and a JSON endpoint for programmatic access.

---

## ✨ Highlights

- **State-of-the-art model:** ConvNeXtV2 (ImageNet-22k pretraining).
- **FastAPI** backend with CORS enabled and a built-in dark UI at `/` for drag-and-drop testing.
- Simple **POST `/classify`** endpoint accepting JPEG/PNG/WEBP and returning top-1 label + confidence.
- GPU aware (mixed precision on CUDA, automatic CPU fallback).

---

## 📁 Project Structure

.
├── main.py # FastAPI app (endpoints, model loading, UI)
├── requirements.txt # Python dependencies
├── README.md # (you can replace with this file)
└── LICENSE # Your project license

yaml
Copy code

---

## 🚀 Quickstart

### 1) Python environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
Dependencies used by the app: fastapi, uvicorn[standard], transformers, torch, pillow, python-multipart, timm.

2) Download the model (once)
This project defaults to loading a local copy of the ConvNeXtV2 Base 22k 224 checkpoint under ./models/convnextv2-base-22k-224. You can clone it with Git LFS:

bash
Copy code
# Install Git LFS first: https://git-lfs.com
git lfs install
git clone https://huggingface.co/facebook/convnextv2-base-22k-224 ./models/convnextv2-base-22k-224
The app reads the model location from MODEL_ID (env var). If unset, it falls back to ./models/convnextv2-base-22k-224.

3) Run the API
bash
Copy code
export MODEL_ID=./models/convnextv2-base-22k-224  # optional (uses default if omitted)
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
Then open: http://127.0.0.1:8000

You’ll see a sleek test page where you can upload an image and view the predicted label + confidence.

🔌 API
## 🖥️ Web UI

The API includes a built-in **interactive web interface** at `/`:

- **Drag & Drop Upload**: Quickly test the model by dropping images directly into the browser.
- **Live Preview**: See the uploaded image before classification.
- **Confidence Meter**: Animated progress bar showing top-1 prediction confidence.
- **Metadata Panel**: Displays filename, model ID, and prediction summary.

This UI is optimized for **fast testing** and works on both desktop and mobile browsers.

POST /classify – Classify an image
Consumes: multipart/form-data with field file

Accepted types: image/jpeg, image/png, image/webp (others return HTTP 415)

Responses:

200 OK with JSON

400 Bad Request if image cannot be parsed

415 Unsupported Media Type for disallowed content types

Request (cURL):

bash
Copy code
curl -X POST http://127.0.0.1:8000/classify \
  -F "file=@path/to/your_image.jpg"
Response (JSON):

json
Copy code
{
  "filename": "your_image.jpg",
  "model_id": "./models/convnextv2-base-22k-224",
  "label": "Egyptian_cat",
  "pretty_label": "Egyptian cat",
  "confidence": 0.9723,
  "message": "Looks like egyptian cat."
}
🧠 Model & Inference
Model and processor are loaded from MODEL_ID via AutoModelForImageClassification and AutoImageProcessor.

On CUDA devices, inference uses torch.cuda.amp.autocast for mixed precision; otherwise runs on CPU.

Softmax is applied to logits; top-1 class is returned.

⚙️ Configuration
MODEL_ID (env var): path or HF Hub model id. Defaults to ./models/convnextv2-base-22k-224.

CORS: Wide open (all origins, methods, headers) to simplify testing. Adjust for production.

Host/Port: Configure via your uvicorn command or process manager (e.g., systemd, Supervisor, Docker).

🔒 Production Notes
Restrict CORS and consider an API key or auth layer before exposing publicly.

Put the service behind a reverse proxy (Nginx, Caddy) with HTTPS.

Pre-pull models on boot; warm up the app with a dummy request.

For scale: enable GPU, use a process manager, and consider batching or async queues.

🧪 Testing
Health/UI: open / in a browser and try the demo uploader.

API: use the cURL example above or a REST client (Insomnia/Postman).

🔧 Troubleshooting
415 Unsupported Media Type — Only JPEG/PNG/WEBP are accepted. Convert your image.

400 Invalid image file. — File cannot be parsed by Pillow; verify it's a valid image.

Model not found — Ensure MODEL_ID points to a valid path/model id and the folder contains the Hugging Face weights and config.

Slow inference — Verify CUDA is available (torch.cuda.is_available()), otherwise run on a GPU machine.

📜 License
See LICENSE for details.