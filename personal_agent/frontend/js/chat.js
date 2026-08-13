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

    const avatar = role === 'user' 
        ? `<div class="message-avatar">U</div>` 
        : `<div class="message-avatar"><svg width="18" height="18" viewBox="0 0 48 48" fill="none"><rect width="48" height="48" rx="12" fill="currentColor"/><circle cx="24" cy="24" r="3" fill="#fff"/></svg></div>`;
    
    let contentHtml = role === 'agent' ? Utils.parseMarkdown(content) : Utils.parseMarkdown(content);

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
            // Refresh sessions list
            await loadSessions();
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
            // Check if it's an ANSWER or just THOUGHT/ACTION
            // We usually only show ANSWER in UI, but if it has content we show it
            if (msg.content) {
                appendMessage('agent', msg.content, pendingObservations);
                pendingObservations = []; // reset
            }
        }
    });
}
