"use client"

import { useState, useCallback, useEffect, useRef } from "react"
import { Upload, X, Sparkles, ArrowRight } from "lucide-react"
import { cn } from "@/lib/utils"
import { predict, warmup } from "@/lib/ort-inference"

const FASHION_CLASSES = [
  { id: 0, name: "T-shirt/Top", icon: "👕" },
  { id: 1, name: "Trouser", icon: "👖" },
  { id: 2, name: "Pullover", icon: "🧥" },
  { id: 3, name: "Dress", icon: "👗" },
  { id: 4, name: "Coat", icon: "🧥" },
  { id: 5, name: "Sandal", icon: "👡" },
  { id: 6, name: "Shirt", icon: "👔" },
  { id: 7, name: "Sneaker", icon: "👟" },
  { id: 8, name: "Bag", icon: "👜" },
  { id: 9, name: "Ankle Boot", icon: "🥾" },
]

interface PredictionResult {
  classId: number
  className: string
  confidence: number
  allProbabilities: { classId: number; className: string; probability: number }[]
}

export function FashionClassifier() {
  const [image, setImage] = useState<string | null>(null)
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFile = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) return

    setImageFile(file)
    setError(null)
    const reader = new FileReader()
    reader.onload = (e) => {
      setImage(e.target?.result as string)
      setPrediction(null)
    }
    reader.readAsDataURL(file)
  }, [])

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setIsDragging(false)
      const file = e.dataTransfer.files[0]
      if (file) handleFile(file)
    },
    [handleFile]
  )

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (file) handleFile(file)
    },
    [handleFile]
  )

  const clearImage = useCallback(() => {
    setImage(null)
    setImageFile(null)
    setPrediction(null)
    setError(null)
    if (fileInputRef.current) fileInputRef.current.value = ""
  }, [])

  useEffect(() => {
    warmup()
  }, [])

  const analyzeImage = useCallback(async () => {
    if (!imageFile) return

    setIsAnalyzing(true)
    setError(null)

    try {
      const result = await predict(imageFile)
      const sorted = [...result.allProbabilities].sort(
        (a, b) => b.probability - a.probability
      )
      setPrediction({ ...result, allProbabilities: sorted })
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setIsAnalyzing(false)
    }
  }, [imageFile])

  return (
    <div className="w-full max-w-4xl mx-auto">
      <div className="grid md:grid-cols-2 gap-8 items-start">
        {/* Upload Area */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <p className="text-sm uppercase tracking-widest text-muted-foreground font-medium">
              Input
            </p>
            {image && (
              <button
                onClick={clearImage}
                className="text-sm text-muted-foreground hover:text-foreground transition-colors flex items-center gap-1"
              >
                <X className="w-3.5 h-3.5" />
                Clear
              </button>
            )}
          </div>

          <div
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onClick={() => !image && fileInputRef.current?.click()}
            className={cn(
              "relative aspect-square rounded-2xl border-2 border-dashed transition-all duration-300 overflow-hidden",
              isDragging
                ? "border-accent bg-accent/5 scale-[1.02]"
                : "border-border hover:border-muted-foreground/50",
              !image && "cursor-pointer",
              image && "border-solid border-border"
            )}
          >
            {image ? (
              <img
                src={image}
                alt="Uploaded fashion item"
                className="w-full h-full object-contain bg-secondary/30"
              />
            ) : (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 p-8">
                <div className="w-16 h-16 rounded-full bg-secondary flex items-center justify-center">
                  <Upload className="w-7 h-7 text-muted-foreground" />
                </div>
                <div className="text-center">
                  <p className="text-foreground font-medium">
                    Drop your image here
                  </p>
                  <p className="text-sm text-muted-foreground mt-1">
                    or click to browse
                  </p>
                </div>
                <p className="text-xs text-muted-foreground/70">
                  Supports JPG, PNG, WebP
                </p>
              </div>
            )}
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleInputChange}
            className="hidden"
          />

          {error && (
            <p className="text-sm text-red-500">{error}</p>
          )}

          {image && (
            <button
              onClick={analyzeImage}
              disabled={isAnalyzing}
              className={cn(
                "w-full py-4 px-6 rounded-xl font-medium text-sm tracking-wide transition-all duration-300 flex items-center justify-center gap-2",
                "bg-primary text-primary-foreground hover:opacity-90",
                isAnalyzing && "opacity-70 cursor-not-allowed"
              )}
            >
              {isAnalyzing ? (
                <>
                  <div className="w-4 h-4 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  Classify Image
                </>
              )}
            </button>
          )}
        </div>

        {/* Results Area */}
        <div className="space-y-4">
          <p className="text-sm uppercase tracking-widest text-muted-foreground font-medium">
            Prediction
          </p>

          <div
            className={cn(
              "aspect-square rounded-2xl border transition-all duration-500",
              prediction
                ? "bg-card border-border"
                : "bg-secondary/30 border-transparent"
            )}
          >
            {prediction ? (
              <div className="h-full flex flex-col p-6">
                {/* Main Prediction */}
                <div className="flex-1 flex flex-col items-center justify-center text-center">
                  <div className="w-20 h-20 rounded-2xl bg-accent/10 flex items-center justify-center mb-4">
                    <span className="text-4xl" role="img" aria-label={prediction.className}>
                      {FASHION_CLASSES[prediction.classId].icon}
                    </span>
                  </div>
                  <h3 className="text-2xl font-semibold text-foreground tracking-tight">
                    {prediction.className}
                  </h3>
                  <p className="text-accent font-medium mt-1">
                    {(prediction.confidence * 100).toFixed(1)}% confidence
                  </p>
                </div>

                {/* Top 3 Probabilities */}
                <div className="pt-6 border-t border-border space-y-3">
                  <p className="text-xs uppercase tracking-widest text-muted-foreground">
                    Top Predictions
                  </p>
                  {prediction.allProbabilities.slice(0, 3).map((prob, i) => (
                    <div key={prob.classId} className="flex items-center gap-3">
                      <span className="text-sm text-muted-foreground w-4">
                        {i + 1}.
                      </span>
                      <span className="text-sm font-medium flex-1 text-foreground">
                        {prob.className}
                      </span>
                      <div className="w-24 h-1.5 bg-secondary rounded-full overflow-hidden">
                        <div
                          className="h-full bg-accent/70 rounded-full transition-all duration-500"
                          style={{ width: `${prob.probability * 100}%` }}
                        />
                      </div>
                      <span className="text-xs text-muted-foreground w-12 text-right">
                        {(prob.probability * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-center p-8">
                <div className="w-16 h-16 rounded-full bg-secondary flex items-center justify-center mb-4">
                  <ArrowRight className="w-6 h-6 text-muted-foreground" />
                </div>
                <p className="text-muted-foreground">
                  Upload an image to see predictions
                </p>
                <p className="text-sm text-muted-foreground/70 mt-1">
                  The model will classify it into one of 10 fashion categories
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Categories Grid */}
      <div className="mt-16 pt-12 border-t border-border">
        <p className="text-sm uppercase tracking-widest text-muted-foreground font-medium mb-6 text-center">
          Supported Categories
        </p>
        <div className="grid grid-cols-5 md:grid-cols-10 gap-3">
          {FASHION_CLASSES.map((cls) => (
            <div
              key={cls.id}
              className={cn(
                "flex flex-col items-center gap-2 p-3 rounded-xl transition-all duration-300",
                prediction?.classId === cls.id
                  ? "bg-accent/10 ring-1 ring-accent"
                  : "bg-secondary/50 hover:bg-secondary"
              )}
            >
              <span className="text-2xl" role="img" aria-label={cls.name}>
                {cls.icon}
              </span>
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground text-center leading-tight">
                {cls.name}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
