import { FashionClassifier } from "@/components/fashion-classifier"

export default function Home() {
  return (
    <main className="min-h-screen">
      {/* Header */}
      <header className="py-6 px-6 md:px-12">
        <nav className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
              <span className="text-primary-foreground text-sm font-bold">F</span>
            </div>
            <span className="font-semibold text-foreground tracking-tight">
              Fashion Classifier
            </span>
          </div>
          <div className="flex items-center gap-6">
            <a
              href="https://github.com/zalandoresearch/fashion-mnist"
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
              Dataset
            </a>
            <a
              href="#"
              className="text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
              Documentation
            </a>
          </div>
        </nav>
      </header>

      {/* Hero */}
      <section className="pt-12 pb-8 md:pt-24 md:pb-16 px-6 md:px-12">
        <div className="max-w-4xl mx-auto text-center">
          <p className="text-sm uppercase tracking-[0.2em] text-accent font-medium mb-4">
            Neural Network Demo
          </p>
          <h1 className="text-4xl md:text-6xl lg:text-7xl font-semibold text-foreground leading-[1.1] tracking-tight text-balance">
            Fashion-MNIST
            <br />
            <span className="text-muted-foreground">Image Classifier</span>
          </h1>
          <p className="mt-6 text-lg text-muted-foreground max-w-2xl mx-auto leading-relaxed text-pretty">
            A convolutional neural network trained to classify clothing items into 10 categories. 
            Upload an image to test the model&apos;s predictions.
          </p>
        </div>
      </section>

      {/* Classifier */}
      <section className="py-12 px-6 md:px-12">
        <FashionClassifier />
      </section>

      {/* Model Info */}
      <section className="py-16 px-6 md:px-12 mt-8">
        <div className="max-w-4xl mx-auto">
          <div className="grid md:grid-cols-3 gap-8">
            <div className="text-center md:text-left">
              <p className="text-3xl font-semibold text-foreground">~92%</p>
              <p className="text-sm text-muted-foreground mt-1">Test Accuracy</p>
            </div>
            <div className="text-center md:text-left">
              <p className="text-3xl font-semibold text-foreground">CNN</p>
              <p className="text-sm text-muted-foreground mt-1">LeNet Architecture</p>
            </div>
            <div className="text-center md:text-left">
              <p className="text-3xl font-semibold text-foreground">421K</p>
              <p className="text-sm text-muted-foreground mt-1">Parameters</p>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 px-6 md:px-12 border-t border-border">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-sm text-muted-foreground">
            EHB420E — Artificial Neural Networks Term Project
          </p>
          <p className="text-sm text-muted-foreground">
            Efe Karagül · May 2026
          </p>
        </div>
      </footer>
    </main>
  )
}
