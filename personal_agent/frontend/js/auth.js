/**
 * Authentication Logic for Personal AI Agent
 */

// Toggle between Login and Register tabs
function switchAuthTab(tab) {
    const loginBtn = document.getElementById('tab-login');
    const registerBtn = document.getElementById('tab-register');
    const loginForm = document.getElementById('login-form');
    const registerForm = document.getElementById('register-form');

    if (tab === 'login') {
        loginBtn.classList.add('active');
        registerBtn.classList.remove('active');
        loginForm.classList.add('active');
        registerForm.classList.remove('active');
    } else {
        loginBtn.classList.remove('active');
        registerBtn.classList.add('active');
        loginForm.classList.remove('active');
        registerForm.classList.add('active');
    }
}

// Button loading state
function setButtonLoading(btnId, isLoading) {
    const btn = document.getElementById(btnId);
    if (!btn) return;
    
    const text = btn.querySelector('.btn-text');
    const loader = btn.querySelector('.btn-loader');
    
    if (isLoading) {
        btn.disabled = true;
        if (text) text.style.display = 'none';
        if (loader) loader.style.display = 'block';
    } else {
        btn.disabled = false;
        if (text) text.style.display = 'inline';
        if (loader) loader.style.display = 'none';
    }
}

// Handle Login Form Submit
async function handleLogin(e) {
    e.preventDefault();
    
    const email = document.getElementById('login-email').value;
    const password = document.getElementById('login-password').value;
    
    setButtonLoading('login-btn', true);
    
    try {
        const res = await API.auth.login(email, password);
        API.setTokens(res.access_token, res.refresh_token);
        
        // Setup user context and transition to app
        await initApp();
        Utils.showToast('Đăng nhập thành công', 'success');
        
    } catch (error) {
        Utils.showToast(error.message, 'error');
    } finally {
        setButtonLoading('login-btn', false);
    }
}

// Handle Register Form Submit
async function handleRegister(e) {
    e.preventDefault();
    
    const name = document.getElementById('register-name').value;
    const email = document.getElementById('register-email').value;
    const password = document.getElementById('register-password').value;
    
    setButtonLoading('register-btn', true);
    
    try {
        await API.auth.register(name, email, password);
        Utils.showToast('Đăng ký thành công! Vui lòng đăng nhập.', 'success');
        
        // Switch to login tab and fill email
        switchAuthTab('login');
        document.getElementById('login-email').value = email;
        document.getElementById('login-password').value = '';
        document.getElementById('login-password').focus();
        
    } catch (error) {
        Utils.showToast(error.message, 'error');
    } finally {
        setButtonLoading('register-btn', false);
    }
}

// Handle Logout
async function handleLogout() {
    try {
        await API.auth.logout();
    } catch (e) {
        console.error(e);
    }
    
    // Clear state
    API.clearTokens();
    window.currentUser = null;
    sessionStorage.removeItem('active_session');
    
    // UI Reset
    document.getElementById('messages-container').innerHTML = '';
    document.getElementById('sessions-list').innerHTML = '';
    
    // Switch page
    switchPage('auth-page');
    Utils.showToast('Đã đăng xuất', 'info');
}

// Listen for token expiry
window.addEventListener('auth:expired', () => {
    window.currentUser = null;
    switchPage('auth-page');
    Utils.showToast('Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.', 'error', 5000);
});
