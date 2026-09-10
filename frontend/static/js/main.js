﻿// DOM elements
const el = (id) => document.getElementById(id);
const uploadArea = el('uploadArea');
const imageInput = el('imageInput');
const uploadPlaceholder = el('uploadPlaceholder');
const uploadPreview = el('uploadPreview');
const previewImage = el('previewImage');
const removeBtn = el('removeBtn');
const fileInfo = el('fileInfo');
const fileName = el('fileName');
const fileSize = el('fileSize');
const processBtn = el('processBtn');
const scaleButtons = document.querySelectorAll('.scale-btn');
const originalImage = el('originalImage');
const originalPlaceholder = el('originalPlaceholder');
const originalInfo = el('originalInfo');
const enhancedImage = el('enhancedImage');
const enhancedPlaceholder = el('enhancedPlaceholder');
const enhancedInfo = el('enhancedInfo');
const loadingOverlay = el('loadingOverlay');
const loadingStatus = el('loadingStatus');
const progressFill = el('progressFill');
const downloadBtn = el('downloadBtn');
const terminalContent = el('terminalContent');
const clearTerminal = el('clearTerminal');
const statusBadge = el('statusBadge');
const statusDot = el('statusBadge').querySelector('.status-dot');
const statusText = el('statusText');
const deviceInfo = el('deviceInfo');
const subtitle = el('subtitle');
const toast = el('toast');

let selectedScale = 4;
let currentFile = null;
let serverModel = 'edsr';
let serverDevice = 'cpu';

// ---- Terminal ----
function log(text, type) {
    const line = document.createElement('div');
    line.className = 'log-line ' + (type || '');
    line.textContent = new Date().toLocaleTimeString() + '  ' + text;
    terminalContent.appendChild(line);
    terminalContent.scrollTop = terminalContent.scrollHeight;
    if (terminalContent.children.length > 50) terminalContent.firstChild.remove();
}

function toastMsg(msg) {
    toast.textContent = msg;
    toast.hidden = false;
    setTimeout(() => toast.hidden = true, 2500);
}

// ---- Server connection ----
function connectToServer() {
    fetch('/api/status')
        .then(r => r.json())
        .then(data => {
            statusDot.className = 'status-dot online';
            statusText.textContent = 'GPU 就绪';
            serverModel = data.model_type || 'edsr';
            serverDevice = (data.device || 'cpu').toUpperCase();
            deviceInfo.textContent = serverDevice + ' | ' + serverModel.toUpperCase();

            const modelName = serverModel === 'realesrgan' ? 'Real-ESRGAN + EDSR' : 'EDSR';
            subtitle.textContent = modelName;

            // Update scale labels
            document.querySelector('[data-scale="4"] .scale-label').textContent =
                serverModel === 'realesrgan' ? 'Real-ESRGAN' : 'EDSR';
            document.querySelector('[data-scale="8"] .scale-label').textContent =
                serverModel === 'realesrgan' ? 'ESRGAN+EDSR' : 'EDSR级联';
        })
        .catch(() => {
            statusDot.className = 'status-dot offline';
            statusText.textContent = '离线';
            deviceInfo.textContent = '';
        });
}

// ---- Upload ----
uploadArea.addEventListener('click', () => imageInput.click());

uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadArea.classList.add('dragover'); });
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
uploadArea.addEventListener('drop', e => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

imageInput.addEventListener('change', e => {
    if (e.target.files.length) handleFile(e.target.files[0]);
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        toastMsg('请上传图像文件');
        return;
    }
    currentFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
        const url = e.target.result;
        previewImage.src = url;
        originalImage.src = url;
        originalImage.hidden = false;
        originalPlaceholder.hidden = true;

        uploadPlaceholder.hidden = true;
        uploadPreview.hidden = false;
        fileInfo.hidden = false;

        const img = new Image();
        img.onload = () => {
            fileName.textContent = file.name;
            fileSize.textContent = img.width + ' × ' + img.height;
            originalInfo.hidden = false;
            originalInfo.textContent = img.width + ' × ' + img.height;
            log('加载: ' + file.name + ' (' + img.width + '×' + img.height + ')');
            processBtn.disabled = false;
        };
        img.src = url;

        // Reset output
        enhancedImage.hidden = true;
        enhancedPlaceholder.hidden = false;
        enhancedInfo.hidden = true;
        downloadBtn.hidden = true;
    };
    reader.readAsDataURL(file);
}

removeBtn.addEventListener('click', () => {
    currentFile = null;
    uploadPreview.hidden = true;
    uploadPlaceholder.hidden = false;
    fileInfo.hidden = true;
    originalImage.hidden = true;
    originalPlaceholder.hidden = false;
    originalInfo.hidden = true;
    processBtn.disabled = true;
    imageInput.value = '';
    log('已移除图像');
});

// ---- Scale buttons ----
scaleButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        scaleButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        selectedScale = +btn.dataset.scale;
    });
});

// ---- Process ----
processBtn.addEventListener('click', async () => {
    if (!currentFile) return;

    // Show loading
    loadingOverlay.hidden = false;
    enhancedPlaceholder.hidden = true;
    progressFill.style.width = '0%';
    loadingStatus.textContent = '处理中...';
    processBtn.disabled = true;
    log('开始 ' + selectedScale + 'x 超分...');

    const formData = new FormData();
    formData.append('image', currentFile);
    formData.append('scale', selectedScale);
    formData.append('method', 'edsr');

    const startTime = Date.now();

    try {
        const response = await fetch('/api/enhance', {
            method: 'POST',
            body: formData
        });

        // Simulate progress since we don't have streaming for this endpoint
        let progress = 0;
        const progressTimer = setInterval(() => {
            progress = Math.min(progress + Math.random() * 15, 85);
            progressFill.style.width = progress + '%';
        }, 500);

        const data = await response.json();
        clearInterval(progressTimer);
        progressFill.style.width = '100%';

        if (data.success) {
            const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
            loadingStatus.textContent = '完成! ' + elapsed + 's';
            setTimeout(() => { loadingOverlay.hidden = true; }, 500);

            enhancedImage.src = data.enhanced.path + '?t=' + Date.now();
            enhancedImage.hidden = false;
            enhancedInfo.hidden = false;
            enhancedInfo.textContent = data.enhanced.width + ' × ' + data.enhanced.height;

            downloadBtn.href = data.enhanced.path;
            downloadBtn.download = 'sr_' + selectedScale + 'x_' + currentFile.name;
            downloadBtn.hidden = false;

            log('✓ 完成: ' + data.original.width + '×' + data.original.height +
                ' → ' + data.enhanced.width + '×' + data.enhanced.height +
                ' (' + data.metrics.process_time_ms + 'ms, ' + data.metrics.device.toUpperCase() + ')', 'success');
            toastMsg('超分完成! ' + elapsed + 's');
        } else {
            loadingOverlay.hidden = true;
            log('✗ 错误: ' + data.error, 'error');
            toastMsg('处理失败: ' + data.error);
        }
    } catch (error) {
        loadingOverlay.hidden = true;
        log('✗ 请求失败: ' + error.message, 'error');
        toastMsg('网络错误');
    } finally {
        processBtn.disabled = false;
    }
});

// ---- Clear terminal ----
clearTerminal.addEventListener('click', () => {
    terminalContent.innerHTML = '';
    log('终端已清空');
});

// ---- Init ----
document.addEventListener('DOMContentLoaded', () => {
    connectToServer();
    setInterval(connectToServer, 60000); // Check every 60s, no spam
});