import io
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
from torchvision import transforms

from model import load_model

FASHION_CLASSES = [
    "T-shirt/Top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot",
]

WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "best_CNN_ReLU.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="Fashion-MNIST Classifier")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model(str(WEIGHTS_PATH), device=DEVICE)

# Fashion-MNIST: white object on black background, normalized to mean/std.
preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,)),
])


def prepare_image(raw: bytes, invert: bool = True) -> torch.Tensor:
    img = Image.open(io.BytesIO(raw)).convert("L")
    # Fashion-MNIST is white-on-black. User uploads are usually dark-on-light,
    # so invert by default. Heuristic: if mean pixel > 127, invert.
    if invert and sum(img.getdata()) / (img.size[0] * img.size[1]) > 127:
        img = ImageOps.invert(img)
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    return tensor


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    raw = await file.read()
    try:
        x = prepare_image(raw)
    except Exception as e:
        raise HTTPException(400, f"Could not read image: {e}")

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().tolist()

    top_id = int(max(range(10), key=lambda i: probs[i]))
    return {
        "classId": top_id,
        "className": FASHION_CLASSES[top_id],
        "confidence": probs[top_id],
        "allProbabilities": [
            {"classId": i, "className": FASHION_CLASSES[i], "probability": probs[i]}
            for i in range(10)
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
