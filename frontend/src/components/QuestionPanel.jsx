import { useState } from 'react'
import SourceChunks from './SourceChunks'

export default function QuestionPanel({
  onAsk,
  askState,
  indexed,
  messages,
  selectedConversationId,
}) {
  const [question, setQuestion] = useState('')
  const [answerMode, setAnswerMode] = useState('detailed')

  function handleSubmit(e) {
    e.preventDefault()
    if (!question.trim()) return
    onAsk(question, answerMode)
    setQuestion('')
  }

  const isLoading = askState.status === 'loading'
  const placeholder = indexed
    ? "What are the candidate's skills?"
    : 'Upload a document first…'

  return (
    <section className="panel question-panel">
      <h2>{selectedConversationId ? 'Conversation' : 'New Conversation'}</h2>

      <div className="chat-thread">
        {messages.length === 0 ? (
          <p className="hint">Ask your first question to start this conversation.</p>
        ) : (
          messages.map((m, i) => (
            <div key={i} className={`chat-msg ${m.role}`}>
              <div className="chat-role">{m.role === 'user' ? 'You' : 'Assistant'}</div>
              <p className="answer-text">{m.content}</p>
              {m.role === 'assistant' && m.sources && m.sources.length > 0 && (
                <SourceChunks sources={m.sources} />
              )}
            </div>
          ))
        )}
      </div>

      <form onSubmit={handleSubmit} className="ask-form">
        <select
          className="question-input"
          value={answerMode}
          onChange={e => setAnswerMode(e.target.value)}
          disabled={isLoading}
          aria-label="Answer mode"
        >
          <option value="concise">Concise</option>
          <option value="detailed">Detailed</option>
          <option value="bullet_summary">Bullet Summary</option>
        </select>
        <input
          type="text"
          className="question-input"
          placeholder={placeholder}
          value={question}
          onChange={e => setQuestion(e.target.value)}
          disabled={isLoading}
          autoComplete="off"
        />
        <button
          type="submit"
          className="ask-btn"
          disabled={isLoading || !question.trim()}
        >
          {isLoading ? (
            <>
              <span className="spinner" /> Thinking…
            </>
          ) : (
            'Ask'
          )}
        </button>
      </form>

      {isLoading && (
        <p className="loading-msg">Retrieving chunks and generating answer…</p>
      )}

      {askState.status === 'success' && (
        <div className="answer-block">
          <h3 className="section-label">Latest Answer</h3>
          <p className="answer-text">{askState.answer}</p>
          <SourceChunks sources={askState.sources} />
        </div>
      )}

      {askState.status === 'error' && (
        <p className="status-msg error">❌ {askState.answer}</p>
      )}
    </section>
  )
}
