/**
 * Session Management Logic for Personal AI Agent
 */

let sessionsList = [];

async function loadSessions() {
    try {
        const data = await API.sessions.list();
        sessionsList = data.sessions || [];
        renderSessionsList();
    } catch (e) {
        console.error('Failed to load sessions', e);
    }
}

function renderSessionsList() {
    const listEl = document.getElementById('sessions-list');
    listEl.innerHTML = '';

    if (sessionsList.length === 0) {
        listEl.innerHTML = `
            <div class="empty-state small">
                <p>Chưa có cuộc trò chuyện nào</p>
            </div>
        `;
        return;
    }

    sessionsList.forEach(session => {
        const div = document.createElement('div');
        div.className = `session-item ${session.session_key === window.currentSessionId ? 'active' : ''}`;
        
        // Use created date as title for now
        const title = `Hội thoại ${Utils.formatDate(session.created_at)}`;
        
        div.innerHTML = `
            <span class="session-title">${title}</span>
            <button class="btn-icon" onclick="deleteSession(event, '${session.session_key}')" title="Xóa">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/></svg>
            </button>
        `;
        
        div.onclick = (e) => {
            if (e.target.closest('.btn-icon')) return; // Ignore if clicked delete
            switchSession(session.session_key);
        };

        listEl.appendChild(div);
    });
}

async function switchSession(sessionKey) {
    if (sessionKey === window.currentSessionId) return;

    window.currentSessionId = sessionKey;
    sessionStorage.setItem('active_session', sessionKey);
    renderSessionsList();
    
    // In a real app, you would fetch chat history here.
    // For this implementation, since backend /chat doesn't return full history on load,
    // we just clear the screen to show it's a different session.
    // In future, you might want to add GET /api/chat/history endpoint.
    document.getElementById('messages-container').innerHTML = '';
    document.getElementById('welcome-state').style.display = 'none';
    document.getElementById('messages-container').style.display = 'flex';
    document.getElementById('chat-title').textContent = 'Cuộc trò chuyện';
    
    if (window.innerWidth <= 768) {
        toggleSidebar(); // Close sidebar on mobile after selecting
    }
}

function createNewSession() {
    window.currentSessionId = null;
    sessionStorage.removeItem('active_session');
    
    document.getElementById('messages-container').innerHTML = '';
    document.getElementById('welcome-state').style.display = 'flex';
    document.getElementById('messages-container').style.display = 'none';
    document.getElementById('chat-title').textContent = 'Cuộc trò chuyện mới';
    
    renderSessionsList(); // Remove active state
    
    if (window.innerWidth <= 768) {
        toggleSidebar(); // Close sidebar on mobile
    }
}

async function deleteSession(event, sessionKey) {
    event.stopPropagation();
    
    Utils.confirm('Bạn có chắc chắn muốn xóa cuộc trò chuyện này? Dữ liệu không thể khôi phục.', async () => {
        try {
            await API.sessions.delete(sessionKey);
            Utils.showToast('Đã xóa cuộc trò chuyện', 'success');
            
            if (window.currentSessionId === sessionKey) {
                createNewSession();
            }
            await loadSessions();
        } catch (e) {
            Utils.showToast('Xóa thất bại', 'error');
        }
    });
}
