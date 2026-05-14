import { NextRequest, NextResponse } from "next/server"

export const runtime = "nodejs"

const BACKEND_URL = process.env.BACKEND_URL ?? "http://127.0.0.1:8000"

export async function POST(req: NextRequest) {
  try {
    const form = await req.formData()
    const file = form.get("file")
    if (!file || !(file instanceof Blob)) {
      return NextResponse.json({ error: "No file uploaded" }, { status: 400 })
    }

    const upstream = new FormData()
    upstream.append("file", file, (file as File).name ?? "upload.png")

    const res = await fetch(`${BACKEND_URL}/predict`, {
      method: "POST",
      body: upstream,
    })

    const data = await res.json()
    return NextResponse.json(data, { status: res.status })
  } catch (err) {
    return NextResponse.json(
      { error: `Backend unreachable: ${(err as Error).message}` },
      { status: 502 }
    )
  }
}
