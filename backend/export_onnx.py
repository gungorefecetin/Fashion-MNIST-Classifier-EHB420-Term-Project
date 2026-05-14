"""Export best_CNN_ReLU.pth to ONNX for in-browser inference.

Run: ./venv/bin/python export_onnx.py
Output: ../public/model.onnx
"""
from pathlib import Path

import torch

from model import load_model

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "best_CNN_ReLU.pth"
OUT = ROOT / "public" / "model.onnx"

model = load_model(str(WEIGHTS), device="cpu")
model.eval()

dummy = torch.zeros(1, 1, 28, 28)

torch.onnx.export(
    model,
    dummy,
    str(OUT),
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17,
)

print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")
