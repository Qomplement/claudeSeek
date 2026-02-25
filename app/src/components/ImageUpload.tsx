import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'

interface Props {
  onImageSelected: (file: File, preview: string) => void
  disabled?: boolean
}

const ACCEPTED_TYPES = {
  'image/png': ['.png'],
  'image/jpeg': ['.jpg', '.jpeg'],
  'image/webp': ['.webp'],
}

export default function ImageUpload({ onImageSelected, disabled }: Props) {
  const [preview, setPreview] = useState<string | null>(null)

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      const file = acceptedFiles[0]
      if (!file) return

      const url = URL.createObjectURL(file)
      setPreview(url)
      onImageSelected(file, url)
    },
    [onImageSelected],
  )

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED_TYPES,
    maxFiles: 1,
    disabled,
  })

  return (
    <div className="space-y-4">
      <div
        {...getRootProps()}
        className={`
          cursor-pointer rounded-lg border-2 border-dashed p-8 text-center transition-colors
          ${isDragActive ? 'border-indigo-500 bg-indigo-500/10' : 'border-gray-600 hover:border-gray-400'}
          ${disabled ? 'cursor-not-allowed opacity-50' : ''}
        `}
      >
        <input {...getInputProps()} />
        {preview ? (
          <img
            src={preview}
            alt="Uploaded document"
            className="mx-auto max-h-64 rounded object-contain"
          />
        ) : (
          <div className="space-y-2">
            <p className="text-lg text-gray-300">
              {isDragActive ? 'Drop the image here...' : 'Drop a document image here, or click to select'}
            </p>
            <p className="text-sm text-gray-500">
              PNG, JPG, or WebP
            </p>
          </div>
        )}
      </div>

      {preview && !disabled && (
        <button
          onClick={(e) => {
            e.stopPropagation()
            setPreview(null)
          }}
          className="text-sm text-gray-400 hover:text-white"
        >
          Clear image
        </button>
      )}
    </div>
  )
}
