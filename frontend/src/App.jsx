import { useState } from 'react'
import UploadPanel from './components/UploadPanel'
import QuestionPanel from './components/QuestionPanel'

const API = 'http://localhost:8000/api'

export default function App() {
  const [indexed, setIndexed] = useState(false)
  const [uploadState, setUploadState] = useState({ status: 'idle', message: '' })
  const [askState, setAskState] = useState({ status: 'idle', answer: '', sources: [] })

  async function handleUpload(file) {
    setUploadState({ status: 'loading', message: '' })
    const form = new FormData()
    form.append('file', file)

    try {
      const res = await fetch(`${API}/upload`, { method: 'POST', body: form })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Upload failed')
      setIndexed(true)
      setUploadState({
        status: 'success',
        message: `${data.filename} — ${data.chunk_count} chunks indexed`,
      })
    } catch (err) {
      setUploadState({
        status: 'error',
        message: err.message.includes('fetch')
          ? 'Cannot reach backend. Is the server running at localhost:8000?'
          : err.message,
      })
    }
  }

  async function handleAsk(question) {
    if (!question.trim()) return
    setAskState({ status: 'loading', answer: '', sources: [] })

    try {
      const res = await fetch(`${API}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Request failed')
      setAskState({ status: 'success', answer: data.answer, sources: data.sources })
    } catch (err) {
      setAskState({
        status: 'error',
        answer: err.message.includes('fetch')
          ? 'Cannot reach backend. Is the server running at localhost:8000?'
          : err.message,
        sources: [],
      })
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>Document Q&amp;A</h1>
        <p className="subtitle">Upload a document, ask anything about it</p>
      </header>
      <main className="app-main">
        <UploadPanel onUpload={handleUpload} uploadState={uploadState} />
        <QuestionPanel onAsk={handleAsk} askState={askState} indexed={indexed} />
      </main>
    </div>
  )
}
