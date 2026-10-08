/**
 * Session Management Logic for Personal AI Agent
 *
 * Session titles are stored in localStorage to display the first user question
 * as the session name, similar to Claude's sidebar behavior.
 */

let sessionsList = [];

// ─── Session Title Storage (localStorage) ───
function getSessionTitles() {
    try {
        return JSON.parse(localStorage.getItem('session_titles') || '{}');
    } catch {
        return {};
    }
}

function setSessionTitle(sessionKey, title) {
    const titles = getSessionTitles();
    titles[sessionKey] = title;
    localStorage.setItem('session_titles', JSON.stringify(titles));
}

function getSessionTitle(sessionKey) {
    return getSessionTitles()[sessionKey] || null;
}

function removeSessionTitle(sessionKey) {
    const titles = getSessionTitles();
    delete titles[sessionKey];
    localStorage.setItem('session_titles', JSON.stringify(titles));
}

// ─── Session API ───
async function loadSessions() {
    try {
        const data = await API.sessions.list();
        sessionsList = data.sessions || [];
        renderSessionsList();
    } catch (e) {
        console.error('Failed to load sessions', e);
    }
}

function getSessionDisplayTitle(session) {
    // Check localStorage for saved title (first user question)
    const saved = getSessionTitle(session.session_key);
    if (saved) {
        return saved.length > 55 ? saved.substring(0, 55) + '…' : saved;
    }
    // Fallback: show date
    return `Hội thoại ${Utils.formatDate(session.created_at)}`;
}

function getTimeGroupLabel(dateString) {
    if (!dateString) return 'Cũ hơn';
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now - date;
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffDays === 0) return 'Hôm nay';
    if (diffDays === 1) return 'Hôm qua';
    if (diffDays < 7) return '7 ngày trước';
    if (diffDays < 30) return '30 ngày trước';
    return 'Cũ hơn';
}

function renderSessionsList() {
    const listEl = document.getElementById('sessions-list');
    listEl.innerHTML = '';

    if (sessionsList.length === 0) {
        listEl.innerHTML = `
            <div class="empty-state-small">
                <p>Chưa có cuộc trò chuyện nào</p>
            </div>
        `;
        return;
    }

    // Group sessions by time
    const groups = {};
    sessionsList.forEach(session => {
        const label = getTimeGroupLabel(session.created_at);
        if (!groups[label]) groups[label] = [];
        groups[label].push(session);
    });

    // Render by group
    const groupOrder = ['Hôm nay', 'Hôm qua', '7 ngày trước', '30 ngày trước', 'Cũ hơn'];
    groupOrder.forEach(groupName => {
        if (!groups[groupName] || groups[groupName].length === 0) return;

        const groupLabel = document.createElement('div');
        groupLabel.className = 'session-time-group';
        groupLabel.textContent = groupName;
        listEl.appendChild(groupLabel);

        groups[groupName].forEach(session => {
            const div = document.createElement('div');
            div.className = `session-item ${session.session_key === window.currentSessionId ? 'active' : ''}`;

            const title = getSessionDisplayTitle(session);

            div.innerHTML = `
                <span class="session-title" title="${title}">${title}</span>
                <button class="btn-icon" onclick="deleteSession(event, '${session.session_key}')" title="Xóa">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/></svg>
                </button>
            `;

            div.onclick = (e) => {
                if (e.target.closest('.btn-icon')) return;
                switchSession(session.session_key);
            };

            listEl.appendChild(div);
        });
    });
}

async function switchSession(sessionKey) {
    if (sessionKey === window.currentSessionId) return;

    window.currentSessionId = sessionKey;
    sessionStorage.setItem('active_session', sessionKey);
    renderSessionsList();

    // Clear the screen for the new session
    document.getElementById('messages-container').innerHTML = '';
    document.getElementById('welcome-state').style.display = 'none';
    document.getElementById('messages-container').style.display = 'flex';

    // Find session and update chat title
    const session = sessionsList.find(s => s.session_key === sessionKey);
    const chatTitle = document.getElementById('chat-title');
    if (session && chatTitle) {
        chatTitle.textContent = getSessionDisplayTitle(session);
    } else if (chatTitle) {
        chatTitle.textContent = 'Cuộc trò chuyện';
    }

    // Load chat history from backend
    try {
        const data = await API.sessions.getHistory(sessionKey);
        if (data.messages && data.messages.length > 0) {
            renderChatHistory(data.messages);
        }
    } catch (e) {
        console.error('Failed to load chat history', e);
        Utils.showToast('Không thể tải lịch sử tin nhắn', 'error');
    }

    if (window.innerWidth <= 768) {
        toggleSidebar();
    }
}

function createNewSession() {
    window.currentSessionId = null;
    sessionStorage.removeItem('active_session');

    document.getElementById('messages-container').innerHTML = '';
    document.getElementById('welcome-state').style.display = 'flex';
    document.getElementById('messages-container').style.display = 'none';
    document.getElementById('chat-title').textContent = 'Cuộc trò chuyện mới';

    renderSessionsList();

    if (window.innerWidth <= 768) {
        toggleSidebar();
    }
}

async function deleteSession(event, sessionKey) {
    event.stopPropagation();

    Utils.confirm('Bạn có chắc chắn muốn xóa cuộc trò chuyện này? Dữ liệu không thể khôi phục.', async () => {
        try {
            await API.sessions.delete(sessionKey);
            removeSessionTitle(sessionKey); // Clean up stored title
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
