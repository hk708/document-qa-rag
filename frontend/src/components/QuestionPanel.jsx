import { useState } from 'react'
import SourceChunks from './SourceChunks'

export default function QuestionPanel({ onAsk, askState, indexed }) {
  const [question, setQuestion] = useState('')

  function handleSubmit(e) {
    e.preventDefault()
    if (!question.trim()) return
    onAsk(question)
  }

  const isLoading = askState.status === 'loading'
  const placeholder = indexed
    ? "What are the candidate's skills?"
    : 'Upload a document first…'

  return (
    <section className="panel question-panel">
      <h2>Ask a Question</h2>

      <form onSubmit={handleSubmit} className="ask-form">
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
          <h3 className="section-label">Answer</h3>
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
