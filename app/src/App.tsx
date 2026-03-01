import { useState } from 'react'
import Header from './components/Header'
import ImageUpload from './components/ImageUpload'
import OcrResult from './components/OcrResult'
import LoadingSpinner from './components/LoadingSpinner'
import { runOcr, fileToBase64 } from './lib/ocr-client'
import type { OcrResponse } from './lib/ocr-client'

type Status = 'idle' | 'loading' | 'success' | 'error'

export default function App() {
  const [status, setStatus] = useState<Status>('idle')
  const [result, setResult] = useState<OcrResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [prompt, setPrompt] = useState('Convert the document to markdown.')

  const handleImageSelected = async (file: File) => {
    setStatus('loading')
    setError(null)
    setResult(null)

    try {
      const base64 = await fileToBase64(file)
      const response = await runOcr({
        imageBase64: base64,
        mimeType: file.type,
        prompt,
      })
      setResult(response)
      setStatus('success')
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error'
      setError(message)
      setStatus('error')
    }
  }

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <Header />

      <main className="mx-auto max-w-5xl px-6 py-8">
        <div className="grid gap-8 lg:grid-cols-2">
          {/* Left: Upload */}
          <div className="space-y-4">
            <h2 className="text-lg font-semibold">Upload Document</h2>

            <ImageUpload
              onImageSelected={handleImageSelected}
              disabled={status === 'loading'}
            />

            <div className="space-y-2">
              <label htmlFor="prompt" className="block text-sm text-gray-400">
                Prompt
              </label>
              <select
                id="prompt"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                className="w-full rounded border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-gray-200"
                disabled={status === 'loading'}
              >
                <option value="Convert the document to markdown.">
                  Document to Markdown
                </option>
                <option value="<|grounding|>Convert the document to markdown.">
                  Markdown + Bounding Boxes (grounding)
                </option>
                <option value="Convert the document to text.">
                  Plain text OCR
                </option>
                <option value="Extract all text from this image.">
                  Free text extraction
                </option>
              </select>
            </div>

            {status === 'error' && (
              <div className="rounded border border-red-800 bg-red-900/30 px-4 py-3">
                <p className="text-sm font-medium text-red-400">Error</p>
                <p className="mt-1 text-sm text-red-300">{error}</p>
                <p className="mt-2 text-xs text-red-400/70">
                  Make sure the vLLM server is running on port 8000.
                  See the <a href="http://localhost:3000/docs/deployment" className="underline">deployment docs</a>.
                </p>
              </div>
            )}
          </div>

          {/* Right: Result */}
          <div>
            {status === 'loading' && <LoadingSpinner />}
            {status === 'success' && result && <OcrResult result={result} />}
            {status === 'idle' && (
              <div className="flex h-full items-center justify-center rounded-lg border border-dashed border-gray-700 p-12">
                <p className="text-center text-gray-500">
                  Upload a document image to see the OCR result here.
                </p>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}
