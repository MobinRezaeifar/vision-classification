# Vision Classification API (FastAPI + ConvNeXtV2)

A production-ready image classification API built with **FastAPI** and **Hugging Face Transformers**, powered by the **ConvNeXtV2** model family.

---

## ✨ Features

- 🔍 **Top-tier Model**: Uses ConvNeXtV2 pretrained on ImageNet-22k.
- ⚡ **FastAPI Backend**: With automatic CUDA support and CPU fallback.
- 🎨 **Sleek Web UI**: Drag-and-drop testing interface available at `/`.
- 🔗 **Simple REST API**: `POST /classify` for programmatic image classification.

---

## 📁 Project Structure

```
.
├── main.py              # FastAPI app with endpoints and UI
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── LICENSE              # Project license
└── models/
    └── convnextv2-base-22k-224/   # Model weights (via git-lfs)
```

---

## 🚀 Quickstart

### 1. Set up Python environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> 🧩 Required packages: `fastapi`, `uvicorn[standard]`, `transformers`, `torch`, `pillow`, `python-multipart`, `timm`

---

### 2. Download the Model

Install Git LFS and clone the pretrained model:

```bash
# Install Git LFS: https://git-lfs.com
git lfs install
git clone https://huggingface.co/facebook/convnextv2-base-22k-224 ./models/convnextv2-base-22k-224
```

> Alternatively, set `MODEL_ID` to any Hugging Face model ID or local path.

---

### 3. Run the API

```bash
export MODEL_ID=./models/convnextv2-base-22k-224
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

## 🖼️ Web UI at `/`

A minimal, elegant HTML interface:

- Drag-and-drop or upload images (`.jpg`, `.png`, `.webp`)
- View model predictions in real-time
- Includes confidence score, filename, and top label

---

## 🔌 API Reference

### POST `/classify`

**Request:**

- Method: `POST`
- Content-Type: `multipart/form-data`
- Field: `file` (image)

Supported formats: `.jpeg`, `.png`, `.webp`

**Example (cURL):**

```bash
curl -X POST http://127.0.0.1:8000/classify   -F "file=@path/to/your_image.jpg"
```

**Response:**

```json
{
  "filename": "your_image.jpg",
  "model_id": "./models/convnextv2-base-22k-224",
  "label": "Egyptian_cat",
  "pretty_label": "Egyptian cat",
  "confidence": 0.9723,
  "message": "Looks like egyptian cat."
}
```

---

## ⚙️ Configuration Options

| Variable   | Description                         | Default                                 |
|------------|-------------------------------------|-----------------------------------------|
| `MODEL_ID` | Path or Hugging Face model ID       | `./models/convnextv2-base-22k-224`       |

---

## 🧪 Testing

- 🔍 Open `/` in browser to try the UI
- 🧪 Use Postman or curl to test `/classify`

---

## ⚠️ Troubleshooting

- `415 Unsupported Media Type`: Use only `.jpg`, `.png`, or `.webp`.
- `400 Invalid image`: Check image format and validity.
- **Model not loading**: Make sure `MODEL_ID` points to the correct local folder or HF model ID.

---

## 📜 License

See LICENSE for details.