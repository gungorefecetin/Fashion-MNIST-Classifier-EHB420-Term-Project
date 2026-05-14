/** @type {import('next').NextConfig} */
const nextConfig = {
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  // onnxruntime-web is a hybrid CJS/ESM package; mark it external on the server
  // so Next/Webpack/Turbopack don't try to bundle its WASM glue.
  serverExternalPackages: ["onnxruntime-web"],
  async headers() {
    return [
      {
        source: "/model.onnx",
        headers: [
          { key: "Cache-Control", value: "public, max-age=31536000, immutable" },
        ],
      },
      {
        source: "/ort/:path*",
        headers: [
          { key: "Cache-Control", value: "public, max-age=31536000, immutable" },
        ],
      },
    ]
  },
}

export default nextConfig
