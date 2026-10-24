/**
 * Main Application Controller
 */

// ─── UI State Management ───
function switchPage(pageId) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    const target = document.getElementById(pageId);
    if (target) target.classList.add('active');
}

function openPanel(panelId) {
    const panel = document.getElementById(`${panelId}-panel`);
    if (!panel) return;

    const wasOpen = panel.classList.contains('open');

    // Close all panels first
    closeAllPanels();

    // If it wasn't open before, open it now (toggle behavior)
    if (!wasOpen) {
        panel.classList.add('open');

        // Only activate backdrop overlay on mobile devices
        if (window.innerWidth <= 768) {
            const backdrop = document.getElementById('panel-backdrop');
            if (backdrop) backdrop.classList.add('active');
        }

        // Refresh data if needed
        if (panelId === 'documents' && typeof loadDocuments === 'function') {
            loadDocuments();
        }
    }
}

function closePanel(panelId) {
    const panel = document.getElementById(`${panelId}-panel`);
    if (panel) panel.classList.remove('open');
    
    const anyOpen = document.querySelectorAll('.slide-panel.open').length > 0;
    if (!anyOpen) {
        const backdrop = document.getElementById('panel-backdrop');
        if (backdrop) backdrop.classList.remove('active');
    }
}

function closeAllPanels() {
    document.querySelectorAll('.slide-panel').forEach(p => p.classList.remove('open'));
    const backdrop = document.getElementById('panel-backdrop');
    if (backdrop) backdrop.classList.remove('active');
}

function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    if (!sidebar) return;
    sidebar.classList.toggle('open');
    
    const backdrop = document.getElementById('sidebar-backdrop');
    if (backdrop) {
        if (window.innerWidth <= 768 && sidebar.classList.contains('open')) {
            backdrop.classList.add('active');
        } else {
            backdrop.classList.remove('active');
        }
    }
}

function toggleUserMenu() {
    const menu = document.getElementById('user-menu');
    if (menu) {
        menu.style.display = menu.style.display === 'none' || !menu.style.display ? 'block' : 'none';
    }
}

function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'light' ? 'dark' : 'light';
    
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
    
    const label = document.getElementById('theme-label');
    if (label) {
        label.textContent = next === 'dark' ? 'Chế độ sáng' : 'Chế độ tối';
    }
    const menu = document.getElementById('user-menu');
    if (menu) menu.style.display = 'none'; // Close menu after click
}

// ─── Initialization ───
async function initApp() {
    try {
        const { access } = API.getTokens();
        
        // If no token, show login
        if (!access) {
            switchPage('auth-page');
            return;
        }
        
        // Fetch user profile
        const user = await API.auth.me();
        window.currentUser = user;
        
        // Update UI
        const nameEl = document.getElementById('user-name');
        const emailEl = document.getElementById('user-email');
        const avatarEl = document.getElementById('user-avatar');
        
        if (nameEl) nameEl.textContent = user.username || 'User';
        if (emailEl) emailEl.textContent = user.email || '';
        if (avatarEl) avatarEl.textContent = (user.username || 'U').charAt(0).toUpperCase();
        
        // Restore session if exists
        const savedSessionId = sessionStorage.getItem('active_session');
        const welcomeState = document.getElementById('welcome-state');
        const msgContainer = document.getElementById('messages-container');
        const chatTitle = document.getElementById('chat-title');
        
        if (savedSessionId) {
            window.currentSessionId = savedSessionId;
            if (welcomeState) welcomeState.style.display = 'none';
            if (msgContainer) msgContainer.style.display = 'flex';
            if (chatTitle) chatTitle.textContent = 'Cuộc trò chuyện';

            // Load chat history for the restored session
            try {
                const data = await API.sessions.getHistory(savedSessionId);
                if (data.messages && data.messages.length > 0 && typeof renderChatHistory === 'function') {
                    renderChatHistory(data.messages);
                    // Update title from session title storage
                    const savedTitle = typeof getSessionTitle === 'function' ? getSessionTitle(savedSessionId) : null;
                    if (savedTitle && chatTitle) {
                        chatTitle.textContent = savedTitle.length > 50 ? savedTitle.substring(0, 50) + '…' : savedTitle;
                    }
                }
            } catch (e) {
                console.error('Failed to restore chat history:', e);
            }
        } else {
            window.currentSessionId = null;
            if (welcomeState) welcomeState.style.display = 'flex';
            if (msgContainer) msgContainer.style.display = 'none';
            if (chatTitle) chatTitle.textContent = 'Cuộc trò chuyện mới';
        }
        
        // Load sidebar data
        if (typeof loadSessions === 'function') {
            await loadSessions();
        }

        // Check Google connection status
        await checkGoogleStatus();
        
        // Show main app
        switchPage('main-app');
        
    } catch (e) {
        console.error('App init failed:', e);
        API.clearTokens();
        switchPage('auth-page');
    }
}

// ─── Google OAuth Management (Popup Mechanism) ───
let isGoogleConnected = false;

async function checkGoogleStatus() {
    try {
        const data = await API.google.getStatus();
        updateGoogleStatusUI(Boolean(data && data.connected));
    } catch (e) {
        console.error('Failed to check Google connection status:', e);
        updateGoogleStatusUI(false);
    }
}

function updateGoogleStatusUI(connected) {
    isGoogleConnected = connected;

    // Header Button
    const headerBtn = document.getElementById('google-header-btn');
    const headerDot = document.getElementById('google-header-dot');
    const headerLabel = document.getElementById('google-header-label');

    if (headerBtn) {
        headerBtn.title = connected 
            ? 'Tài khoản Google: Đã kết nối (Nhấn để ngắt kết nối)' 
            : 'Tài khoản Google: Chưa kết nối (Nhấn để kết nối)';
        if (connected) {
            headerBtn.classList.add('connected');
        } else {
            headerBtn.classList.remove('connected');
        }
    }

    if (headerDot) {
        if (connected) {
            headerDot.classList.add('connected');
        } else {
            headerDot.classList.remove('connected');
        }
    }

    if (headerLabel) {
        headerLabel.textContent = connected ? 'Google (Đã nối)' : 'Google';
    }

    // Menu Item
    const menuLabel = document.getElementById('google-menu-label');
    const menuBtn = document.getElementById('google-menu-btn');
    if (menuLabel) {
        menuLabel.textContent = connected ? 'Ngắt kết nối Google' : 'Kết nối Google';
    }
    if (menuBtn) {
        if (connected) {
            menuBtn.classList.add('danger');
        } else {
            menuBtn.classList.remove('danger');
        }
    }
}

async function toggleGoogleAuth() {
    // Close user menu if open
    const menu = document.getElementById('user-menu');
    if (menu) menu.style.display = 'none';

    if (isGoogleConnected) {
        Utils.confirm(
            'Bạn có chắc chắn muốn ngắt kết nối tài khoản Google không? Sau khi ngắt kết nối, trợ lý sẽ không thể đọc hoặc gửi email Gmail.',
            async () => {
                try {
                    await API.google.disconnect();
                    updateGoogleStatusUI(false);
                    Utils.showToast('Đã ngắt kết nối tài khoản Google.', 'info');
                } catch (err) {
                    Utils.showToast(err.message || 'Lỗi khi ngắt kết nối', 'error');
                }
            }
        );
    } else {
        await connectGoogleWithPopup();
    }
}

async function connectGoogleWithPopup() {
    try {
        Utils.showToast('Đang tạo liên kết Google...', 'info', 1500);
        const data = await API.google.getAuthUrl();
        const authUrl = data.authorization_url || data.url;

        if (!authUrl) {
            throw new Error('Không nhận được URL xác thực từ hệ thống.');
        }

        // Kích thước và căn giữa popup
        const width = 500;
        const height = 650;
        const left = Math.max(0, Math.round(window.screenX + (window.outerWidth - width) / 2));
        const top = Math.max(0, Math.round(window.screenY + (window.outerHeight - height) / 2));

        const popup = window.open(
            authUrl,
            'GoogleAuthPopup',
            `width=${width},height=${height},left=${left},top=${top},status=no,menubar=no,toolbar=no,scrollbars=yes`
        );

        if (!popup || popup.closed || typeof popup.closed === 'undefined') {
            Utils.showToast('Trình duyệt đã chặn popup. Vui lòng bật quyền cho phép popup để kết nối Google.', 'warning', 5000);
            return;
        }

        popup.focus();
    } catch (err) {
        console.error('Lỗi khi mở popup Google OAuth:', err);
        Utils.showToast(err.message || 'Không thể mở cửa sổ kết nối Google', 'error');
    }
}

// Lắng nghe tín hiệu postMessage từ Popup Window khi hoàn tất OAuth
window.addEventListener('message', (event) => {
    if (!event.data || typeof event.data !== 'object') return;

    if (event.data.type === 'GOOGLE_AUTH_SUCCESS') {
        updateGoogleStatusUI(true);
        Utils.showToast('Kết nối Google thành công! Các công cụ Gmail đã sẵn sàng.', 'success', 4000);
        
        // Tự động gửi lại câu hỏi trước đó nếu có
        if (window._pendingAuthQuery) {
            const retryQuery = window._pendingAuthQuery;
            window._pendingAuthQuery = null;
            const chatInput = document.getElementById('chat-input');
            if (chatInput) {
                chatInput.value = retryQuery;
                chatInput.dispatchEvent(new Event('input'));
                handleSendMessage();
            }
        }
    } else if (event.data.type === 'GOOGLE_AUTH_ERROR') {
        const errorMsg = event.data.error || 'Xác thực không thành công';
        Utils.showToast(`Kết nối Google thất bại: ${errorMsg}`, 'error', 4000);
    }
});

// Global click to close dropdowns
document.addEventListener('click', (e) => {
    // Close user menu
    const userMenu = document.getElementById('user-menu');
    const userInfo = document.getElementById('user-info');
    if (userMenu && userMenu.style.display === 'block' && userInfo && !userInfo.contains(e.target)) {
        userMenu.style.display = 'none';
    }
    
    // Close sidebar on mobile when clicking outside
    const sidebar = document.getElementById('sidebar');
    const toggleBtn = document.getElementById('sidebar-toggle-open');
    if (window.innerWidth <= 768 && sidebar && sidebar.classList.contains('open')) {
        if (!sidebar.contains(e.target) && toggleBtn && !toggleBtn.contains(e.target)) {
            sidebar.classList.remove('open');
            const backdrop = document.getElementById('sidebar-backdrop');
            if (backdrop) backdrop.classList.remove('active');
        }
    }
});

// Boot up
document.addEventListener('DOMContentLoaded', () => {
    // Apply saved theme or default to dark for futuristic experience
    const savedTheme = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
    const label = document.getElementById('theme-label');
    if (label) {
        label.textContent = savedTheme === 'dark' ? 'Chế độ sáng' : 'Chế độ tối';
    }
    
    initApp();
});
