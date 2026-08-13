/**
 * API Client for Personal AI Agent
 */

const API_BASE_URL = 'http://localhost:8000/api';

const API = {
    // ─── Token Management ───
    getTokens() {
        return {
            access: localStorage.getItem('access_token'),
            refresh: localStorage.getItem('refresh_token')
        };
    },

    setTokens(access, refresh) {
        if (access) localStorage.setItem('access_token', access);
        if (refresh) localStorage.setItem('refresh_token', refresh);
    },

    clearTokens() {
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
    },

    // ─── Base Fetch Wrapper ───
    async fetchWithAuth(endpoint, options = {}) {
        let { access } = this.getTokens();
        
        const headers = new Headers(options.headers || {});
        if (access) {
            headers.set('Authorization', `Bearer ${access}`);
        }

        const config = {
            ...options,
            headers
        };

        let response = await fetch(`${API_BASE_URL}${endpoint}`, config);

        // Handle Token Expiry (401)
        if (response.status === 401 && access) {
            const refreshed = await this.refreshToken();
            if (refreshed) {
                // Retry original request with new token
                const newAccess = this.getTokens().access;
                headers.set('Authorization', `Bearer ${newAccess}`);
                response = await fetch(`${API_BASE_URL}${endpoint}`, { ...config, headers });
            } else {
                // Refresh failed, force logout
                this.clearTokens();
                window.dispatchEvent(new Event('auth:expired'));
                throw new Error('Session expired. Please login again.');
            }
        }

        return response;
    },

    async refreshToken() {
        const { refresh } = this.getTokens();
        if (!refresh) return false;

        try {
            const response = await fetch(`${API_BASE_URL}/auth/refresh`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ refresh_token: refresh })
            });

            if (response.ok) {
                const data = await response.json();
                this.setTokens(data.access_token, data.refresh_token);
                return true;
            }
        } catch (e) {
            console.error('Failed to refresh token:', e);
        }
        return false;
    },

    // ─── API Endpoints ───
    
    auth: {
        async login(email, password) {
            // OAuth2 Form format
            const formData = new URLSearchParams();
            formData.append('username', email);
            formData.append('password', password);

            const res = await fetch(`${API_BASE_URL}/auth/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                body: formData
            });
            
            if (!res.ok) {
                const error = await res.json();
                throw new Error(error.detail || 'Login failed');
            }
            return res.json();
        },

        async register(name, email, password) {
            const res = await fetch(`${API_BASE_URL}/auth/register`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username: name, email, password })
            });
            
            if (!res.ok) {
                const error = await res.json();
                throw new Error(error.detail || 'Registration failed');
            }
            return res.json();
        },

        async me() {
            const res = await API.fetchWithAuth('/auth/me');
            if (!res.ok) throw new Error('Failed to get user profile');
            return res.json();
        },
        
        async logout() {
            const { refresh } = API.getTokens();
            if (refresh) {
                await API.fetchWithAuth('/auth/logout', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ refresh_token: refresh })
                }).catch(() => {}); // ignore errors on logout
            }
            API.clearTokens();
        }
    },

    chat: {
        async send(query, sessionId) {
            const payload = {
                query,
                agent_profile: "personal_agent"
            };
            if (sessionId) payload.session_id = sessionId;

            const res = await API.fetchWithAuth('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            if (!res.ok) {
                const error = await res.json();
                throw new Error(error.detail?.error || error.detail || 'Chat request failed');
            }
            return res.json();
        }
    },

    sessions: {
        async list() {
            const res = await API.fetchWithAuth('/sessions');
            if (!res.ok) throw new Error('Failed to load sessions');
            return res.json();
        },

        async delete(sessionKey) {
            const res = await API.fetchWithAuth(`/sessions/${sessionKey}`, { method: 'DELETE' });
            if (!res.ok) throw new Error('Failed to delete session');
            return res.json();
        }
    },

    documents: {
        async upload(file, onProgress) {
            const formData = new FormData();
            formData.append('file', file);

            // XMLHttpRequest used for progress tracking
            return new Promise((resolve, reject) => {
                const xhr = new XMLHttpRequest();
                const { access } = API.getTokens();

                xhr.upload.addEventListener('progress', (e) => {
                    if (e.lengthComputable && onProgress) {
                        const percentComplete = (e.loaded / e.total) * 100;
                        onProgress(percentComplete);
                    }
                });

                xhr.addEventListener('load', () => {
                    if (xhr.status >= 200 && xhr.status < 300) {
                        resolve(JSON.parse(xhr.responseText));
                    } else {
                        try {
                            const err = JSON.parse(xhr.responseText);
                            reject(new Error(err.detail || 'Upload failed'));
                        } catch {
                            reject(new Error(`Upload failed with status ${xhr.status}`));
                        }
                    }
                });

                xhr.addEventListener('error', () => reject(new Error('Network error during upload')));
                
                xhr.open('POST', `${API_BASE_URL}/documents/upload`);
                if (access) {
                    xhr.setRequestHeader('Authorization', `Bearer ${access}`);
                }
                xhr.send(formData);
            });
        },

        async list() {
            const res = await API.fetchWithAuth('/documents');
            if (!res.ok) throw new Error('Failed to load documents');
            return res.json();
        },

        async delete(docId) {
            const res = await API.fetchWithAuth(`/documents/${docId}`, { method: 'DELETE' });
            if (!res.ok) throw new Error('Failed to delete document');
            return res.json();
        }
    }
};
