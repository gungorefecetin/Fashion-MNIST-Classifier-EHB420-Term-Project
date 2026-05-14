# Fashion-MNIST-Classifier-EHB420-Term-Project
This repository contains a comparative study on multi-class image classification using the Fashion-MNIST dataset. Built with PyTorch, the project empirically evaluates how different architectural decisions (MLP vs. CNN) and activation functions (ReLU, Sigmoid, Tanh) affect model performance and convergence. By analyzing parameter sharing, spatial information preservation, and the gradient vanishing problem, this project bridges the gap between neural network theory and practical implementation.

Architecture Comparison: Evaluates a baseline Multi-Layer Perceptron (~109k params) against a LeNet-style Convolutional Neural Network (~421k params) to demonstrate the power of spatial inductive bias.
Activation Function Analysis: Compares the convergence speed and final accuracy of ReLU, Sigmoid, and Tanh to empirically manifest the gradient vanishing problem.
Robust Training Pipeline: Implements Mini-batch SGD (Adam), Cross-Entropy Loss, Dropout regularization, and Early Stopping to prevent overfitting.
Comprehensive Evaluation: Includes loss curves, per-class F1 scores, and confusion matrix analysis to interpret model behavior beyond simple accuracy metrics.
