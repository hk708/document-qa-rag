import { useEffect, useState } from 'react'
import UploadPanel from './components/UploadPanel'
import QuestionPanel from './components/QuestionPanel'
import ConversationSidebar from './components/ConversationSidebar'

const API = 'http://localhost:8000/api'

export default function App() {
  const [indexed, setIndexed] = useState(false)
  const [uploadState, setUploadState] = useState({ status: 'idle', message: '' })
  const [askState, setAskState] = useState({ status: 'idle', answer: '', sources: [] })
  const [conversations, setConversations] = useState([])
  const [selectedConversationId, setSelectedConversationId] = useState(null)
  const [messages, setMessages] = useState([])

  useEffect(() => {
    loadConversations()
  }, [])

  async function loadConversations() {
    try {
      const res = await fetch(`${API}/conversations?user_id=local_user`)
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Failed to load conversations')
      setConversations(data)
    } catch {
      // Keep the app usable even if conversation list fetch fails.
    }
  }

  async function loadConversation(conversationId) {
    const res = await fetch(`${API}/conversations/${conversationId}`)
    const data = await res.json()
    if (!res.ok) throw new Error(data.detail || 'Failed to load conversation')
    setSelectedConversationId(conversationId)
    setMessages(data.messages || [])
  }

  function handleNewChat() {
    setSelectedConversationId(null)
    setMessages([])
    setAskState({ status: 'idle', answer: '', sources: [] })
  }

  async function handleUpload(file) {
    setUploadState({ status: 'loading', message: '' })
    const form = new FormData()
    form.append('file', file)

    try {
      const res = await fetch(`${API}/upload`, { method: 'POST', body: form })
      const data = await res.json()

      if (res.status === 409) {
        // Same filename stem, genuinely different content — hard conflict.
        setUploadState({ status: 'error', message: data.detail })
        return
      }

      if (!res.ok) throw new Error(data.detail || 'Upload failed')

      if (data.rename_detected) {
        // Same content uploaded under a different filename (rename/alias case).
        setIndexed(true)
        setUploadState({
          status: 'warning',
          message: `"${data.uploaded_filename}" has the same content as the already-indexed "${data.original_filename}". No new document was indexed.`,
        })
        return
      }

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

  async function handleAsk(question, answerMode) {
    if (!question.trim()) return
    setAskState({ status: 'loading', answer: '', sources: [] })

    try {
      const res = await fetch(`${API}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question,
          answer_mode: answerMode,
          conversation_id: selectedConversationId,
          user_id: 'local_user',
        }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Request failed')
      setAskState({ status: 'success', answer: data.answer, sources: data.sources })

      if (data.conversation_id) {
        await loadConversation(data.conversation_id)
      }
      await loadConversations()
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
      <main className="app-main app-main-chat">
        <ConversationSidebar
          conversations={conversations}
          selectedConversationId={selectedConversationId}
          onSelectConversation={loadConversation}
          onNewChat={handleNewChat}
        />

        <section className="chat-column">
          <UploadPanel onUpload={handleUpload} uploadState={uploadState} />
          <QuestionPanel
            onAsk={handleAsk}
            askState={askState}
            indexed={indexed}
            messages={messages}
            selectedConversationId={selectedConversationId}
          />
        </section>
      </main>
    </div>
  )
}
