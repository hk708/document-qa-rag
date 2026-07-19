export default function ConversationSidebar({
  conversations,
  selectedConversationId,
  onSelectConversation,
  onNewChat,
}) {
  return (
    <aside className="panel sidebar-panel">
      <div className="sidebar-header">
        <h2>Conversations</h2>
        <button className="ask-btn" type="button" onClick={onNewChat}>
          New Chat
        </button>
      </div>

      {conversations.length === 0 ? (
        <p className="hint">No conversations yet.</p>
      ) : (
        <div className="conversation-list">
          {conversations.map(conv => (
            <button
              key={conv.conversation_id}
              type="button"
              className={`conversation-item ${selectedConversationId === conv.conversation_id ? 'active' : ''}`}
              onClick={() => onSelectConversation(conv.conversation_id)}
            >
              <div className="conversation-title">{conv.title}</div>
              <div className="conversation-time">
                {new Date(conv.updated_at).toLocaleString()}
              </div>
            </button>
          ))}
        </div>
      )}
    </aside>
  )
}