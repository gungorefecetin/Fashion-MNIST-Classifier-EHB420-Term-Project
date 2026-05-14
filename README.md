# Fashion-MNIST Image Classifier

> A comparative study of neural network architectures on the Fashion-MNIST dataset, with a live demo interface.

Term project for **EHB420E — Artificial Neural Networks** at Istanbul Technical University, Spring 2026.

---

## Overview

This project compares MLP and CNN architectures on Fashion-MNIST (10-class clothing classification), then drills into how activation function choice affects training dynamics. It ships with a Next.js demo where you can upload any clothing image and see the model's top-3 predictions in real time.

The core question: in a shallow architecture, does activation choice still matter the way textbooks suggest? Short answer — yes, but the effect lives in convergence speed, not final accuracy.

## Results

| Model | Validation Accuracy | Test Accuracy | Epochs to 90% |
|---|---|---|---|
| MLP (baseline) | 88.70% | — | never reached |
| **CNN + ReLU** | **92.47%** | **91.59%** | **3** |
| CNN + Tanh | 92.02% | — | 3 |
| CNN + Sigmoid | 90.47% | — | 12 |

**Three findings worth highlighting:**

1. **Spatial inductive bias wins.** CNN beat MLP by 3.77 points despite both being fed the same data — parameter sharing and local receptive fields matter more than raw capacity.

2. **Activation differences hide in convergence, not accuracy.** ReLU, Tanh, and Sigmoid land within 2 points of each other on final accuracy, but Sigmoid takes 4× more epochs to reach the same plateau. Gradient vanishing manifests as slowdown, not failure, in shallow networks.

3. **Data resolution is the real ceiling.** The worst-performing class is Shirt (F1 0.74) — but the failure mode is the 28×28 grayscale resolution losing the details that distinguish a shirt from a T-shirt or pullover, not a model architecture issue.

## Demo

A Next.js web app lets you upload any clothing image and see the model's prediction with confidence scores.

![Demo screenshot](./docs/demo-screenshot.png)

**Try it locally:**

```bash
cd web
npm install
npm run dev
# open http://localhost:3000
```

The backend runs the trained CNN-ReLU checkpoint and returns top-3 predictions per request.

## Architecture

**CNN (main model, ~421k parameters):**

```
Input (1×28×28)
├─ Conv2d(1→32, 3×3, padding=1)  →  ReLU  →  MaxPool 2×2   →  32×14×14
├─ Conv2d(32→64, 3×3, padding=1) →  ReLU  →  MaxPool 2×2   →  64×7×7
├─ Flatten                                                  →  3136
├─ Linear(3136→128)              →  ReLU  →  Dropout(0.3)
└─ Linear(128→10)                                           →  logits
```

**MLP baseline (~109k parameters):**

```
Input (784)  →  Linear(784→128) → ReLU → Dropout(0.2)
             →  Linear(128→64)  → ReLU
             →  Linear(64→10)   → logits
```

Both trained with Adam (lr=1e-3), batch size 64, max 15 epochs, early stopping on validation accuracy (patience=3), cross-entropy loss.

## Project Structure

```
fashion-mnist-classifier/
├── notebook/
│   └── fashion_mnist_project.ipynb   # Main training notebook (Colab-ready)
├── models/
│   ├── best_MLP.pth                  # Trained checkpoints
│   ├── best_CNN_ReLU.pth
│   ├── best_CNN_Tanh.pth
│   └── best_CNN_Sigmoid.pth
├── results/
│   ├── training_curves.png
│   ├── accuracy_comparison.png
│   ├── confusion_matrix.png
│   └── class_samples.png
├── web/                              # Next.js demo
│   ├── app/
│   ├── components/
│   └── package.json
├── docs/
│   ├── PRD.md                        # Full project requirements document
│   ├── PRD_TR.md                     # Turkish version
│   └── presentation.pptx             # Term presentation slides
└── README.md
```

## Reproducing the Training

### Option 1: Colab (recommended)

Open `notebook/fashion_mnist_project.ipynb` in Google Colab, switch to T4 GPU runtime, and run all cells. Total training time across the 4 models is roughly 15 minutes.

### Option 2: Local

```bash
pip install torch torchvision matplotlib seaborn scikit-learn
python train.py
```

CPU works but training takes ~2 hours instead of 15 minutes.

## Tech Stack

**Model:** PyTorch 2.x, torchvision
**Training:** Google Colab (T4 GPU)
**Demo frontend:** Next.js 14, Tailwind CSS, TypeScript
**Demo backend:** Python FastAPI (model inference)
**Visualization:** Matplotlib, Seaborn

## What We Learned

The most interesting finding wasn't in the accuracy numbers — it was watching Sigmoid CNN's training curve. We expected dramatic gradient vanishing (textbook scenario: model fails to learn). Instead, the model learned just fine, only much slower. This taught us that the textbook "ReLU good, sigmoid bad" framing oversimplifies — the real story is depth-dependent, and in a 2-conv-layer network the effect is subtle. Worth remembering when reading about modern architectures: most empirical claims are tied to specific scales.

The other thing that surprised us was how cleanly the confusion matrix told a structural story. The four upper-body categories (T-shirt, Pullover, Coat, Shirt) confuse each other massively, while the three footwear classes (Sandal, Sneaker, Ankle boot) stay clean. The model isn't doing anything wrong — the data just doesn't carry enough information at 28×28 grayscale to disambiguate those clothing types.

## Future Work

- Test the same activation comparison on a 5-6 layer CNN — We'd expect Sigmoid to actually fail there, not just slow down
- Add RBF network baseline (covered in the course syllabus but not in mainstream tutorials)
- Optimizer comparison (SGD vs Adam vs RMSprop)
- Move to higher resolution / RGB clothing dataset<img width="447" height="542" alt="Screenshot 2026-05-14 at 22 14 50" src="https://github.com/user-attachments/assets/bd80e8d3-cfcb-420f-b125-9e704e2e8f94" />
 for the production-grade version

## References

- Xiao et al. (2017). *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms*
- LeCun et al. (1998). *Gradient-Based Learning Applied to Document Recognition*
- Kingma & Ba (2014). *Adam: A Method for Stochastic Optimization*
- Bishop, C. (1995). *Neural Networks for Pattern Recognition*
- Haykin, S. (2009). *Neural Networks and Learning Machines*

## Author

**Güngör Efe Çetin & Yağız Çoban**
Electronics & Communications Engineering
Istanbul Technical University

EHB420E term project, May 2026. Instructor: Doç. Dr. İsa Yıldırım.

---

*Built as part of coursework. The model and demo are for educational purposes — not production-grade clothing classification.*
