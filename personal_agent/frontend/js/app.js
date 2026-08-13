/**
 * Main Application Controller
 */

// ─── UI State Management ───
function switchPage(pageId) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.getElementById(pageId).classList.add('active');
}

function openPanel(panelId) {
    // Close other panels
    document.querySelectorAll('.slide-panel').forEach(p => p.classList.remove('open'));
    
    // Open target panel
    const panel = document.getElementById(`${panelId}-panel`);
    if (panel) {
        panel.classList.add('open');
        
        // Refresh data if needed
        if (panelId === 'documents') {
            loadDocuments();
        }
    }
}

function closePanel(panelId) {
    const panel = document.getElementById(`${panelId}-panel`);
    if (panel) panel.classList.remove('open');
}

function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    sidebar.classList.toggle('open');
}

function toggleUserMenu() {
    const menu = document.getElementById('user-menu');
    menu.style.display = menu.style.display === 'none' ? 'block' : 'none';
}

function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
    
    const label = document.getElementById('theme-label');
    if (label) {
        label.textContent = next === 'dark' ? 'Chế độ sáng' : 'Chế độ tối';
    }
    toggleUserMenu(); // Close menu after click
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
        document.getElementById('user-name').textContent = user.username || 'User';
        document.getElementById('user-email').textContent = user.email || '';
        document.getElementById('user-avatar').textContent = (user.username || 'U').charAt(0).toUpperCase();
        
        // Restore session if exists
        const savedSessionId = sessionStorage.getItem('active_session');
        if (savedSessionId) {
            window.currentSessionId = savedSessionId;
            document.getElementById('welcome-state').style.display = 'none';
            document.getElementById('messages-container').style.display = 'flex';
            document.getElementById('chat-title').textContent = 'Cuộc trò chuyện';
        } else {
            window.currentSessionId = null;
            document.getElementById('welcome-state').style.display = 'flex';
            document.getElementById('messages-container').style.display = 'none';
            document.getElementById('chat-title').textContent = 'Cuộc trò chuyện mới';
        }
        
        // Load sidebar data
        await loadSessions();
        
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
    if (userMenu && userMenu.style.display === 'block' && !userInfo.contains(e.target)) {
        userMenu.style.display = 'none';
    }
    
    // Close sidebar on mobile when clicking outside
    const sidebar = document.getElementById('sidebar');
    const toggleBtn = document.getElementById('sidebar-toggle-open');
    if (window.innerWidth <= 768 && sidebar && sidebar.classList.contains('open')) {
        if (!sidebar.contains(e.target) && !toggleBtn.contains(e.target)) {
            sidebar.classList.remove('open');
        }
    }
});

// Boot up
document.addEventListener('DOMContentLoaded', () => {
    // Apply saved theme
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        document.documentElement.setAttribute('data-theme', savedTheme);
        const label = document.getElementById('theme-label');
        if (label) {
            label.textContent = savedTheme === 'dark' ? 'Chế độ sáng' : 'Chế độ tối';
        }
    }
    
    initApp();
});
