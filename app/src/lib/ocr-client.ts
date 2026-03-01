export interface OcrRequest {
  imageBase64: string
  mimeType: string
  prompt?: string
}

export interface OcrResponse {
  text: string
  model: string
  usage?: {
    prompt_tokens: number
    completion_tokens: number
    total_tokens: number
  }
}

const DEFAULT_PROMPT = 'Convert the document to markdown.'

export async function runOcr(request: OcrRequest): Promise<OcrResponse> {
  const dataUrl = `data:${request.mimeType};base64,${request.imageBase64}`

  const body = {
    model: 'deepseek-ai/DeepSeek-OCR-2',
    messages: [
      {
        role: 'user',
        content: [
          { type: 'image_url', image_url: { url: dataUrl } },
          { type: 'text', text: request.prompt || DEFAULT_PROMPT },
        ],
      },
    ],
    max_tokens: 4096,
    temperature: 0.0,
    vllm_xargs: {
      ngram_size: 30,
      window_size: 300,
      whitelist_token_ids: [128821, 128822],
    },
  }

  const res = await fetch('/api/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })

  if (!res.ok) {
    const errorText = await res.text()
    throw new Error(`OCR request failed (${res.status}): ${errorText}`)
  }

  const data = await res.json()

  return {
    text: data.choices[0].message.content,
    model: data.model,
    usage: data.usage,
  }
}

export function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      const result = reader.result as string
      // Strip the data URL prefix to get raw base64
      const base64 = result.split(',')[1]
      resolve(base64)
    }
    reader.onerror = reject
    reader.readAsDataURL(file)
  })
}
