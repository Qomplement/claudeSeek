export default function LoadingSpinner() {
  return (
    <div className="flex flex-col items-center gap-4 py-12">
      <div className="h-10 w-10 animate-spin rounded-full border-4 border-gray-600 border-t-indigo-500" />
      <p className="text-sm text-gray-400">
        Processing document with DeepSeek-OCR-2...
      </p>
    </div>
  )
}
