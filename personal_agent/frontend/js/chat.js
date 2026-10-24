/**
 * Chat Interface Logic for Personal AI Agent
 */

// State
let isWaitingForResponse = false;

// DOM Elements
const chatInput = document.getElementById('chat-input');
const chatForm = document.getElementById('chat-form');
const sendBtn = document.getElementById('send-btn');
const messagesContainer = document.getElementById('messages-container');
const welcomeState = document.getElementById('welcome-state');
const typingIndicator = document.getElementById('typing-indicator');

// Auto-resize textarea
if (chatInput) {
    chatInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        
        // Enable/disable send button
        sendBtn.disabled = this.value.trim() === '';
    });

    chatInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!sendBtn.disabled && !isWaitingForResponse) {
                chatForm.dispatchEvent(new Event('submit'));
            }
        }
    });
}

function scrollToBottom() {
    const chatArea = document.getElementById('chat-area');
    if (chatArea) {
        chatArea.scrollTop = chatArea.scrollHeight;
    }
}

function appendMessage(role, content, observations = []) {
    welcomeState.style.display = 'none';
    messagesContainer.style.display = 'flex';

    const msgEl = document.createElement('div');
    msgEl.className = `message ${role}`;

    const avatarLetter = role === 'user' 
        ? ((window.currentUser && window.currentUser.username) ? window.currentUser.username.charAt(0).toUpperCase() : 'U')
        : '';

    const avatar = role === 'user' 
        ? `<div class="message-avatar">${avatarLetter}</div>` 
        : `<div class="message-avatar agent-avatar">AI</div>`;
    
    let contentHtml = Utils.parseMarkdown(content);

    // If agent has observations, append them as collapsible block
    let observationsHtml = '';
    if (role === 'agent' && observations && observations.length > 0) {
        observationsHtml = `
            <div class="observation-block">
                <div class="obs-toggle" onclick="this.nextElementSibling.classList.toggle('show')">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"></polyline></svg>
                    <span>Đã dùng công cụ tìm kiếm (${observations.length} kết quả)</span>
                </div>
                <div class="obs-content">
                    ${observations.map(obs => `<div>${Utils.parseMarkdown(obs)}</div>`).join('<hr>')}
                </div>
            </div>
        `;
    }

    msgEl.innerHTML = `
        ${avatar}
        <div class="message-content">
            <div class="markdown-body">${contentHtml}</div>
            ${observationsHtml}
        </div>
    `;

    messagesContainer.appendChild(msgEl);
    scrollToBottom();
}

async function handleSendMessage(e) {
    if (e) e.preventDefault();
    
    const query = chatInput.value.trim();
    if (!query || isWaitingForResponse) return;

    // Track if this is a brand new session (first message)
    const isNewSession = !window.currentSessionId;

    // Reset input
    chatInput.value = '';
    chatInput.style.height = 'auto';
    sendBtn.disabled = true;

    // Append user message
    appendMessage('user', query);

    // Show typing
    isWaitingForResponse = true;
    typingIndicator.style.display = 'flex';
    scrollToBottom();

    try {
        const sessionId = window.currentSessionId || sessionStorage.getItem('active_session') || null;
        
        const response = await API.chat.send(query, sessionId);
        
        // Hide typing
        typingIndicator.style.display = 'none';
        isWaitingForResponse = false;

        // Save session if new
        if (response.session_id && response.session_id !== window.currentSessionId) {
            window.currentSessionId = response.session_id;
            sessionStorage.setItem('active_session', response.session_id);

            // Save the first question as the session title in localStorage
            if (typeof setSessionTitle === 'function') {
                setSessionTitle(response.session_id, query);
            }

            // Update the header title
            const chatTitle = document.getElementById('chat-title');
            if (chatTitle) {
                chatTitle.textContent = query.length > 50 ? query.substring(0, 50) + '…' : query;
            }

            // Refresh sessions list to show the new session with its title
            await loadSessions();
        }

        // If auth_required, open OAuth popup and prompt user
        if (response.status === 'auth_required') {
            window._pendingAuthQuery = query;
            const authUrl = response.authorize_url;
            appendMessage('agent', `${response.answer}\n\n👉 [**Bấm vào đây để cấp quyền kết nối Google**](${authUrl})`);
            
            if (authUrl) {
                const width = 500, height = 650;
                const left = (window.innerWidth - width) / 2 + window.screenX;
                const top = (window.innerHeight - height) / 2 + window.screenY;
                window.open(
                    authUrl,
                    'google_oauth_popup',
                    `width=${width},height=${height},top=${top},left=${left},scrollbars=yes,status=no`
                );
            }
            return;
        }

        // Append agent response
        appendMessage('agent', response.answer, response.tool_observations);

    } catch (error) {
        typingIndicator.style.display = 'none';
        isWaitingForResponse = false;
        
        // Show error message as agent
        appendMessage('agent', `**Lỗi:** ${error.message}`);
        Utils.showToast('Gửi tin nhắn thất bại', 'error');
    }
}

function sendSuggestion(text) {
    chatInput.value = text;
    chatInput.dispatchEvent(new Event('input'));
    handleSendMessage();
}

// Render entire chat history (if any)
function renderChatHistory(messages) {
    messagesContainer.innerHTML = '';
    
    if (!messages || messages.length === 0) {
        welcomeState.style.display = 'flex';
        messagesContainer.style.display = 'none';
        return;
    }

    welcomeState.style.display = 'none';
    messagesContainer.style.display = 'flex';

    // Group observations into the next assistant message
    let pendingObservations = [];
    
    messages.forEach(msg => {
        if (msg.role === 'user') {
            appendMessage('user', msg.content);
        } else if (msg.role === 'observation') {
            pendingObservations.push(msg.content);
        } else if (msg.role === 'assistant') {
            if (msg.content) {
                appendMessage('agent', msg.content, pendingObservations);
                pendingObservations = [];
            }
        }
    });
}
