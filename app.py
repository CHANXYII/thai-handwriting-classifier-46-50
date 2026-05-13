<<<<<<< HEAD
import base64
import io
import json
import os
from pathlib import Path

from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps


BASE_DIR = Path(__file__).resolve().parent
HISTORY_PATH = BASE_DIR / "logs" / "prediction_history.json"
DEFAULT_MODEL_PATH = BASE_DIR / "checkpoints" / "best_model.pt"
DEFAULT_MAPPING_PATH = BASE_DIR / "checkpoints" / "class_mapping.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvBNAct(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, c, drop=0.0):
        super().__init__()
        self.b1 = ConvBNAct(c, c)
        self.b2 = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.drop(self.b2(self.b1(x))) + x)


class ThaiCNN(nn.Module):
    def __init__(self, n_classes=5, drop=0.3):
        super().__init__()
        self.stem = ConvBNAct(3, 32, k=3, s=1)
        self.stage1 = nn.Sequential(ConvBNAct(32, 64, s=2), ResBlock(64, 0.05))
        self.stage2 = nn.Sequential(ConvBNAct(64, 128, s=2), ResBlock(128, 0.10))
        self.stage3 = nn.Sequential(ConvBNAct(128, 256, s=2), ResBlock(256, 0.15))
        self.stage4 = nn.Sequential(ConvBNAct(256, 384, s=2), ResBlock(384, 0.20))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(drop), nn.Linear(384, n_classes))

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(self.gap(x))


app = Flask(__name__)


def load_mapping(mapping_path: Path) -> dict:
    with mapping_path.open("r", encoding="utf-8") as f:
        return json.load(f)
    

RUNTIME = load_mapping(DEFAULT_MAPPING_PATH)
CLASSES = [str(name) for name in RUNTIME["classes"]]
IMG_SIZE = int(RUNTIME["img_size"])
MEAN = np.array(RUNTIME["normalize"]["mean"], dtype=np.float32)
STD = np.array(RUNTIME["normalize"]["std"], dtype=np.float32)


def load_model(model_path: Path) -> nn.Module:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = ThaiCNN(n_classes=len(CLASSES)).to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


MODEL = load_model(DEFAULT_MODEL_PATH)

def save_prediction_history(prediction_data):

    if not HISTORY_PATH.exists():
        HISTORY_PATH.write_text("[]", encoding="utf-8")

    with HISTORY_PATH.open("r", encoding="utf-8") as f:
        history = json.load(f)

    history.append({
        "id": datetime.now().timestamp(),
        "prediction": prediction_data["prediction"],
        "confidence": prediction_data["confidence"],
        "correct": None,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    with HISTORY_PATH.open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def resize_and_pad(image: Image.Image, img_size: int) -> np.ndarray:
    image = image.convert("RGB")
    src_w, src_h = image.size
    scale = min(img_size / src_w, img_size / src_h)
    new_w = max(1, round(src_w * scale))
    new_h = max(1, round(src_h * scale))

    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (img_size, img_size), (255, 255, 255))
    offset = ((img_size - new_w) // 2, (img_size - new_h) // 2)
    canvas.paste(resized, offset)
    return np.asarray(canvas, dtype=np.float32) / 255.0


def preprocess_pil_image(image: Image.Image) -> torch.Tensor:
    arr = resize_and_pad(image, IMG_SIZE)
    arr = (arr - MEAN) / STD
    arr = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
    return arr.to(DEVICE)


def decode_canvas_image(data_url: str) -> Image.Image:
    if not data_url:
        raise ValueError("Missing image payload")

    encoded = data_url.split(",", 1)[1] if "," in data_url else data_url
    raw = base64.b64decode(encoded)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def ink_ratio(image: Image.Image) -> float:
    gray = np.asarray(ImageOps.grayscale(image), dtype=np.uint8)
    return float(np.mean(gray < 250))


@torch.no_grad()
def predict_probs(image: Image.Image) -> np.ndarray:
    x = preprocess_pil_image(image)
    probs = F.softmax(MODEL(x), dim=1)[0].cpu().numpy()
    return probs.astype(np.float32)


def format_prediction(probs: np.ndarray) -> dict:
    top_idx = int(probs.argmax())
    return {
        "prediction": CLASSES[top_idx],
        "confidence": float(probs[top_idx]),
        "probabilities": {label: float(probs[i]) for i, label in enumerate(CLASSES)},
    }

@app.get("/admin")
def admin_page():
    return render_template("admin.html")

@app.get("/history")
def history_page():
    return render_template("history.html")

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image")

    try:
        image = decode_canvas_image(image_data)
    except Exception:
        return jsonify({"error": "Invalid image payload"}), 400

    if ink_ratio(image) < 0.003:
        return jsonify({"error": "Please draw a digit before predicting."}), 400

    result = format_prediction(predict_probs(image))

    save_prediction_history(result)

    with HISTORY_PATH.open("r", encoding="utf-8") as f:
        history = json.load(f)
    latest_item = history[-1]
    result["id"] = latest_item["id"]

    return jsonify(result)

@app.get("/history/data")
def history_data():

    if not HISTORY_PATH.exists():
        return jsonify([])

    with HISTORY_PATH.open("r", encoding="utf-8") as f:
        history = json.load(f)

    return jsonify(history)


@app.post("/feedback")
def feedback():

    data = request.get_json()

    history_id = data.get("id")
    correct = data.get("correct")

    if not HISTORY_PATH.exists():
        return jsonify({
            "error": "History not found"
        }), 404

    with HISTORY_PATH.open("r", encoding="utf-8") as f:
        history = json.load(f)

    updated = False

    for item in history:
        if item["id"] == history_id:

            item["correct"] = correct
            updated = True
            break

    if not updated:

        return jsonify({
            "error": "Prediction ID not found"
        }), 404

    with HISTORY_PATH.open("w", encoding="utf-8") as f:

        json.dump(
            history,
            f,
            ensure_ascii=False,
            indent=2
        )

    return jsonify({
        "success": True
    })


@app.route('/admin/upload-model', methods=['POST'])
def upload_model():

    global MODEL

    try:

        if 'model' not in request.files:

            return jsonify({
                'success': False,
                'error': 'No model uploaded'
            }), 400

        file = request.files['model']

        if file.filename == '':

            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400

        if not file.filename.endswith('.pt'):

            return jsonify({
                'success': False,
                'error': 'Only .pt file allowed'
            }), 400

        file.save(DEFAULT_MODEL_PATH)

        MODEL = load_model(DEFAULT_MODEL_PATH)

        return jsonify({
            'success': True,
            'message': 'Model updated successfully'
        })

    except Exception as e:

        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
=======
from flask import Flask, render_template, request, jsonify
import base64
import os
import time

app = Flask(__name__)

# Create folder for Dataset
DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

# First page for UI to capture image and label
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint for receiving image and label data from the web page
@app.route('/save-sample', methods=['POST'])
def save_sample():
    data = request.json
    label = data.get('label')
    image_data = data.get('image')

    if not label or not image_data:
        return jsonify({"success": False, "error": "ข้อมูลไม่ครบถ้วน"}), 400

    # Create subfolder based on Label (e.g., dataset/46/)
    label_dir = os.path.join(DATASET_DIR, label)
    os.makedirs(label_dir, exist_ok=True)

    # Split the Base64 header (data:image / png;base64,...)
    encoded_data = image_data.split(',')[1]
    
    # Create filename using Label and Timestamp to avoid duplicates
    filename = f"{label}_{int(time.time() * 1000)}.png"
    filepath = os.path.join(label_dir, filename)

    # Convert Base64 to image and save
    with open(filepath, "wb") as fh:
        fh.write(base64.b64decode(encoded_data))

    return jsonify({"success": True, "filename": filename})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
>>>>>>> b72b90006fb4dd2a815c66f212708ca4904abad7
