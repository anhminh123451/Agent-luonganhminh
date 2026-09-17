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
        
        // Show main app
        switchPage('main-app');
        
    } catch (e) {
        console.error('App init failed:', e);
        API.clearTokens();
        switchPage('auth-page');
    }
}

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
