import { useState } from 'react'

export default function SourceChunks({ sources }) {
  const [expanded, setExpanded] = useState(null)

  if (!sources || sources.length === 0) return null

  function toggle(i) {
    setExpanded(expanded === i ? null : i)
  }

  return (
    <div className="sources">
      <h3 className="section-label">
        Sources <span className="source-count">({sources.length})</span>
      </h3>

      {sources.map((s, i) => (
        <div
          key={i}
          className={`source-card ${expanded === i ? 'expanded' : ''}`}
          onClick={() => toggle(i)}
        >
          <div className="source-header">
            <span className="source-rank">#{s.rank}</span>
            <span className="source-doc">{s.doc_id}</span>
            <span className="source-meta">
              chunk {s.chunk_index} · score {s.score.toFixed(4)}
            </span>
            <span className="source-toggle">{expanded === i ? '▲' : '▼'}</span>
          </div>

          {expanded === i && (
            <p className="source-text">{s.text}</p>
          )}
        </div>
      ))}
    </div>
  )
}
