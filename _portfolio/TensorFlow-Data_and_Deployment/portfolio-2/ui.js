// ui.js - UI interactions and status management
// Add log entry
function addLog(message) {
    const entry = document.createElement('div');
    entry.className = 'text-dark';
    entry.textContent = message;
    logPanel.appendChild(entry);
    logPanel.scrollTop = logPanel.scrollHeight;
}

// Update status badge
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