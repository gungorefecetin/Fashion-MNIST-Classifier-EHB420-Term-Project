"use client"

import type { InferenceSession, Tensor as OrtTensor } from "onnxruntime-web"

export const FASHION_CLASS_NAMES = [
  "T-shirt/Top", "Trouser", "Pullover", "Dress", "Coat",
  "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot",
] as const

// Fashion-MNIST channel mean/std (used during training)
const MEAN = 0.2860
const STD = 0.3530

let sessionPromise: Promise<InferenceSession> | null = null

async function getSession(): Promise<InferenceSession> {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      const ort = await import("onnxruntime-web")
      // Serve WASM files we copied to public/ort/
      ort.env.wasm.wasmPaths = "/ort/"
      // Single-threaded keeps things simple — model is tiny, threads add overhead here
      ort.env.wasm.numThreads = 1
      return ort.InferenceSession.create("/model.onnx", {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      })
    })()
  }
  return sessionPromise
}

/**
 * Render image to 28x28 grayscale, auto-invert white backgrounds
 * (Fashion-MNIST is white object on black), and normalize.
 * Returns Float32Array of length 1*1*28*28 in NCHW order.
 */
async function imageToTensor(file: File): Promise<Float32Array> {
  const bitmap = await createImageBitmap(file)
  const canvas = new OffscreenCanvas(28, 28)
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!

  // Fit-contain on black background
  ctx.fillStyle = "black"
  ctx.fillRect(0, 0, 28, 28)
  const scale = Math.min(28 / bitmap.width, 28 / bitmap.height)
  const w = bitmap.width * scale
  const h = bitmap.height * scale
  ctx.drawImage(bitmap, (28 - w) / 2, (28 - h) / 2, w, h)

  const { data } = ctx.getImageData(0, 0, 28, 28)

  // Grayscale luminance
  const gray = new Float32Array(28 * 28)
  let sum = 0
  for (let i = 0; i < 28 * 28; i++) {
    const r = data[i * 4]
    const g = data[i * 4 + 1]
    const b = data[i * 4 + 2]
    const y = 0.299 * r + 0.587 * g + 0.114 * b
    gray[i] = y
    sum += y
  }

  // If average is bright, assume dark-on-light photo and invert to white-on-black
  const mean = sum / (28 * 28)
  const invert = mean > 127

  const out = new Float32Array(28 * 28)
  for (let i = 0; i < 28 * 28; i++) {
    const v = (invert ? 255 - gray[i] : gray[i]) / 255
    out[i] = (v - MEAN) / STD
  }
  return out
}

function softmax(logits: Float32Array): Float32Array {
  let max = -Infinity
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i]
  const exps = new Float32Array(logits.length)
  let sum = 0
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i] - max)
    exps[i] = e
    sum += e
  }
  for (let i = 0; i < logits.length; i++) exps[i] /= sum
  return exps
}

export interface PredictionResult {
  classId: number
  className: string
  confidence: number
  allProbabilities: { classId: number; className: string; probability: number }[]
}

export async function predict(file: File): Promise<PredictionResult> {
  const [session, input] = await Promise.all([getSession(), imageToTensor(file)])
  const ort = await import("onnxruntime-web")
  const tensor: OrtTensor = new ort.Tensor("float32", input, [1, 1, 28, 28])
  const outputs = await session.run({ input: tensor })
  const logits = outputs.logits.data as Float32Array
  const probs = softmax(logits)

  let topId = 0
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[topId]) topId = i

  return {
    classId: topId,
    className: FASHION_CLASS_NAMES[topId],
    confidence: probs[topId],
    allProbabilities: Array.from(probs).map((p, i) => ({
      classId: i,
      className: FASHION_CLASS_NAMES[i],
      probability: p,
    })),
  }
}

/** Optionally pre-warm the session in the background (e.g. on app mount). */
export function warmup(): void {
  void getSession()
}
