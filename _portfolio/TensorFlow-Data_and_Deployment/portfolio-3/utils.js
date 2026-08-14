// utils.js - Utility functions for input validation and formatting
function clampInputValue(id, min, max) {
    const input = document.getElementById(id);
    let value = parseFloat(input.value);
    if (Number.isNaN(value)) {
        return;
    }
    value = Math.min(Math.max(value, min), max);

    // Format based on input type
    if (id === 'learningRate') {
        input.value = value.toFixed(3);
    } else if (id === 'epochs' || id === 'batchSize') {
        input.value = Math.round(value);
    } else {
        input.value = value.toFixed(1);
    }
}

function applyControlConfig() {
    Object.entries(controlConfig).forEach(([id, attrs]) => {
        const input = document.getElementById(id);
        if (!input) return;
        Object.entries(attrs).forEach(([attr, value]) => {
            if (value === undefined || value === null) return;
            if (attr === 'value') {
                input.value = value;
            } else {
                input.setAttribute(attr, value);
            }
        });
    });
}

function setupInputClamping() {
    Object.entries(controlConfig).forEach(([id, attrs]) => {
        if (attrs.min === undefined || attrs.max === undefined) return;
        const input = document.getElementById(id);
        if (!input) return;
        input.addEventListener('change', () => clampInputValue(id, attrs.min, attrs.max));
        input.addEventListener('blur', () => clampInputValue(id, attrs.min, attrs.max));
    });
}

function addLog(message) {
    const entry = document.createElement('div');
    entry.className = 'text-dark';
    entry.textContent = message;
    logPanel.appendChild(entry);
    logPanel.scrollTop = logPanel.scrollHeight;
}

function updateStatus(status) {
    const badgeClasses = {
        idle: 'bg-dark',
        training: 'bg-warning',
        ready: 'bg-success'
    };
    statusBadge.className = `badge ${badgeClasses[status]}`;
    const statusText = status.charAt(0).toUpperCase() + status.slice(1);
    statusBadge.textContent = statusText;
}

// Initialize canvas with black background
function initializeCanvas() {
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, 280, 280);
}

// Resize canvas for different DPI displays
function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    canvas.width = 280;
    canvas.height = 280;
    ctx.scale(canvas.width / rect.width, canvas.height / rect.height);
    initializeCanvas();
}
