export default function Header() {
  return (
    <header className="border-b border-gray-700 bg-gray-900">
      <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-4">
        <div>
          <h1 className="text-xl font-bold text-white">
            DeepSeek-OCR-2
          </h1>
          <p className="text-sm text-gray-400">
            Document OCR powered by vLLM
          </p>
        </div>
        <nav className="flex gap-4">
          <a
            href="http://localhost:3000/docs/overview"
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm text-indigo-400 hover:text-indigo-300"
          >
            Docs
          </a>
          <a
            href="https://github.com/Qomplement/claudeSeek"
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm text-indigo-400 hover:text-indigo-300"
          >
            GitHub
          </a>
        </nav>
      </div>
    </header>
  )
}
