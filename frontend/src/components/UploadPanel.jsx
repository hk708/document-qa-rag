import { useRef } from 'react'

const ALLOWED = ['.pdf', '.docx', '.txt']

export default function UploadPanel({ onUpload, uploadState }) {
  const inputRef = useRef(null)

  function handleChange(e) {
    const file = e.target.files[0]
    if (!file) return

    const ext = '.' + file.name.split('.').pop().toLowerCase()
    if (!ALLOWED.includes(ext)) {
      inputRef.current.value = ''
      alert(`Unsupported file type "${ext}". Allowed: ${ALLOWED.join(', ')}`)
      return
    }

    onUpload(file)
  }

  const isLoading = uploadState.status === 'loading'

  return (
    <section className="panel upload-panel">
      <h2>Upload Document</h2>
      <p className="hint">PDF, DOCX, or TXT · max 20 MB</p>

      <label className="upload-label">
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.docx,.txt"
          onChange={handleChange}
          disabled={isLoading}
        />
        <span className={`upload-btn ${isLoading ? 'disabled' : ''}`}>
          {isLoading ? (
            <>
              <span className="spinner" /> Indexing…
            </>
          ) : (
            'Choose File'
          )}
        </span>
      </label>

      {uploadState.message && (
        <p className={`status-msg ${uploadState.status}`}>
          {uploadState.status === 'success' ? '✅ ' : uploadState.status === 'warning' ? '⚠️ ' : '❌ '}
          {uploadState.message}
        </p>
      )}
    </section>
  )
}
