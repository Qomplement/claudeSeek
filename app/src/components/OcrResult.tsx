import { useState } from 'react'
import Markdown from 'react-markdown'
import type { OcrResponse } from '../lib/ocr-client'

interface Props {
  result: OcrResponse
}

export default function OcrResult({ result }: Props) {
  const [view, setView] = useState<'rendered' | 'raw'>('rendered')
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    await navigator.clipboard.writeText(result.text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-white">OCR Result</h2>
        <div className="flex gap-2">
          <button
            onClick={() => setView(view === 'rendered' ? 'raw' : 'rendered')}
            className="rounded bg-gray-700 px-3 py-1 text-sm text-gray-300 hover:bg-gray-600"
          >
            {view === 'rendered' ? 'Raw Markdown' : 'Rendered'}
          </button>
          <button
            onClick={handleCopy}
            className="rounded bg-indigo-600 px-3 py-1 text-sm text-white hover:bg-indigo-500"
          >
            {copied ? 'Copied!' : 'Copy'}
          </button>
        </div>
      </div>

      <div className="max-h-[600px] overflow-auto rounded-lg border border-gray-700 bg-gray-800 p-4">
        {view === 'rendered' ? (
          <div className="prose prose-invert max-w-none prose-table:border-collapse prose-th:border prose-th:border-gray-600 prose-th:p-2 prose-td:border prose-td:border-gray-600 prose-td:p-2">
            <Markdown>{result.text}</Markdown>
          </div>
        ) : (
          <pre className="whitespace-pre-wrap text-sm text-gray-300">
            {result.text}
          </pre>
        )}
      </div>

      {result.usage && (
        <p className="text-xs text-gray-500">
          Tokens: {result.usage.prompt_tokens} prompt + {result.usage.completion_tokens} completion = {result.usage.total_tokens} total
        </p>
      )}
    </div>
  )
}
