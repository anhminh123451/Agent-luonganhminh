/**
 * Utilities & Helpers for Personal AI Agent
 */

const Utils = {
    // ─── Toasts ───
    showToast(message, type = 'info', duration = 3000) {
        const container = document.getElementById('toast-container');
        if (!container) return;

        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        let iconSvg = '';
        if (type === 'success') {
            iconSvg = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><path d="M22 4L12 14.01l-3-3"/></svg>';
        } else if (type === 'error') {
            iconSvg = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M15 9l-6 6M9 9l6 6"/></svg>';
        } else {
            iconSvg = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/></svg>';
        }

        toast.innerHTML = `
            <div class="toast-icon">${iconSvg}</div>
            <div class="toast-message">${message}</div>
        `;

        container.appendChild(toast);
        
        // Trigger reflow for animation
        toast.offsetHeight;
        toast.classList.add('show');

        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300); // Wait for transition
        }, duration);
    },

    // ─── Date Formatting ───
    formatDate(dateString) {
        if (!dateString) return '';
        const date = new Date(dateString);
        const now = new Date();
        const diffMs = now - date;
        const diffMins = Math.round(diffMs / 60000);
        const diffHours = Math.round(diffMs / 3600000);
        const diffDays = Math.round(diffMs / 86400000);

        if (diffMins < 1) return 'Vừa xong';
        if (diffMins < 60) return `${diffMins} phút trước`;
        if (diffHours < 24) return `${diffHours} giờ trước`;
        if (diffDays === 1) return 'Hôm qua';
        if (diffDays < 7) return `${diffDays} ngày trước`;
        
        return date.toLocaleDateString('vi-VN', {
            year: 'numeric', month: 'short', day: 'numeric'
        });
    },

    // ─── Simple Markdown Parser ───
    parseMarkdown(text) {
        if (!text) return '';
        let html = text;

        // Escape HTML tags to prevent XSS
        html = html.replace(/</g, '&lt;').replace(/>/g, '&gt;');

        // Bold: **text**
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Italic: *text* or _text_
        html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
        
        // Code blocks: ```language code ```
        html = html.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
        
        // Inline code: `code`
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
        
        // Lists: * item or - item
        html = html.replace(/^[\*-]\s+(.+)$/gm, '<li>$1</li>');
        html = html.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');
        
        // Paragraphs: double newline
        html = html.split(/\n\n+/).map(p => {
            if (p.startsWith('<ul') || p.startsWith('<pre')) return p;
            return `<p>${p}</p>`;
        }).join('');
        
        // Single newlines to <br> (only outside of pre/ul)
        html = html.replace(/(?<!<\/pre>)(?<!<\/ul>)\n/g, '<br>');

        return html;
    },

    // ─── Debounce ───
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    // ─── Confirm Dialog ───
    confirm(message, onConfirm) {
        const dialog = document.getElementById('confirm-dialog');
        const msgEl = document.getElementById('confirm-message');
        const okBtn = document.getElementById('confirm-ok');
        
        msgEl.textContent = message;
        dialog.style.display = 'flex';
        
        // Cleanup old listeners
        const newOkBtn = okBtn.cloneNode(true);
        okBtn.parentNode.replaceChild(newOkBtn, okBtn);
        
        newOkBtn.addEventListener('click', () => {
            dialog.style.display = 'none';
            if (onConfirm) onConfirm();
        });
    }
};

// Global confirm close
function closeConfirmDialog() {
    document.getElementById('confirm-dialog').style.display = 'none';
}
