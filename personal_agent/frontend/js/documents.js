/**
 * Document Management Logic for Personal AI Agent
 */

let documentsList = [];

// Drag & Drop
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    document.getElementById('upload-zone').classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    document.getElementById('upload-zone').classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    document.getElementById('upload-zone').classList.remove('dragover');
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        processUpload(e.dataTransfer.files[0]);
    }
}

function handleFileSelect(e) {
    if (e.target.files && e.target.files.length > 0) {
        processUpload(e.target.files[0]);
    }
}

// Upload Process
async function processUpload(file) {
    // Validate extension
    const allowed = ['.pdf', '.docx', '.csv', '.md'];
    const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
    
    if (!allowed.includes(ext)) {
        Utils.showToast('Định dạng file không được hỗ trợ', 'error');
        return;
    }
    
    // UI Setup
    const progressDiv = document.getElementById('upload-progress');
    const fill = document.getElementById('progress-fill');
    const status = document.getElementById('upload-status');
    const filename = document.getElementById('upload-filename');
    
    filename.textContent = file.name;
    status.textContent = 'Đang tải lên...';
    fill.style.width = '0%';
    progressDiv.style.display = 'block';
    
    try {
        const result = await API.documents.upload(file, (percent) => {
            fill.style.width = `${percent}%`;
            if (percent === 100) {
                status.textContent = 'Đang xử lý (chunking & embedding)...';
            }
        });
        
        status.textContent = 'Hoàn tất!';
        fill.style.backgroundColor = 'var(--color-success)';
        Utils.showToast(`Upload thành công: ${result.chunks_indexed} chunks`, 'success');
        
        // Reload list
        await loadDocuments();
        
    } catch (e) {
        status.textContent = 'Lỗi!';
        fill.style.backgroundColor = 'var(--color-danger)';
        Utils.showToast(e.message, 'error');
    } finally {
        setTimeout(() => {
            progressDiv.style.display = 'none';
            fill.style.width = '0%';
            fill.style.backgroundColor = 'var(--color-primary)';
            document.getElementById('file-input').value = '';
        }, 3000);
    }
}

// Load and Render
async function loadDocuments() {
    try {
        const data = await API.documents.list();
        documentsList = data.documents || [];
        renderDocumentsList();
    } catch (e) {
        console.error('Failed to load documents', e);
    }
}

function renderDocumentsList() {
    const listEl = document.getElementById('documents-list');
    listEl.innerHTML = '';

    if (documentsList.length === 0) {
        listEl.innerHTML = `
            <div class="empty-state">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--color-text-quaternary)" stroke-width="1"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                <p>Chưa có tài liệu nào</p>
                <p class="empty-state-hint">Upload tài liệu để AI Agent có thể phân tích</p>
            </div>
        `;
        return;
    }

    documentsList.forEach(doc => {
        const ext = doc.title.substring(doc.title.lastIndexOf('.')).substring(1).toLowerCase();
        
        const div = document.createElement('div');
        div.className = 'doc-card';
        div.innerHTML = `
            <div class="doc-icon ${ext}">${ext.toUpperCase()}</div>
            <div class="doc-info">
                <div class="doc-title" title="${doc.title}">${doc.title}</div>
                <div class="doc-meta">
                    ${Utils.formatDate(doc.created_at)}
                </div>
            </div>
            <button class="btn-icon" onclick="deleteDocument('${doc.doc_id}')" title="Xóa tài liệu">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/></svg>
            </button>
        `;
        listEl.appendChild(div);
    });
}

function deleteDocument(docId) {
    Utils.confirm('Bạn có chắc chắn muốn xóa tài liệu này khỏi hệ thống? Agent sẽ không thể trả lời dựa trên tài liệu này nữa.', async () => {
        try {
            await API.documents.delete(docId);
            Utils.showToast('Đã xóa tài liệu', 'success');
            await loadDocuments();
        } catch (e) {
            Utils.showToast('Xóa tài liệu thất bại', 'error');
        }
    });
}
