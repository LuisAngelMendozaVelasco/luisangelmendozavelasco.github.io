// utils.js - Utility functions

/**
 * Add a message to the log panel
 */
function addLog(message) {
    const timestamp = new Date().toLocaleTimeString();
    const logEntry = `[${timestamp}] ${message}`;
    
    logPanel.innerHTML += logEntry + '<br>';
    logPanel.scrollTop = logPanel.scrollHeight;
    
    // Keep only last maxLogs entries
    const logs = logPanel.innerHTML.split('<br>').filter(log => log.trim());
    if (logs.length > LOGGING.maxLogs) {
        logPanel.innerHTML = logs.slice(-LOGGING.maxLogs).join('<br>');
    }
}

/**
 * Update status badge and text
 */
function updateStatus(status) {
    const statusMap = {
        'loading': { text: 'Loading Model...', class: 'bg-warning', showSpinner: true },
        'ready': { text: 'Ready!', class: 'bg-success', showSpinner: false },
        'processing': { text: 'Processing...', class: 'bg-info', showSpinner: true },
        'error': { text: 'Error!', class: 'bg-danger', showSpinner: false }
    };
    
    const config = statusMap[status] || { text: 'Unknown', class: 'bg-secondary', showSpinner: false };
    
    statusBadge.className = `badge ${config.class}`;
    statusText.textContent = config.text;
    loadingSpinner.style.display = config.showSpinner ? 'inline-block' : 'none';
}

/**
 * Format probability as percentage
 */
function formatProbability(probability) {
    return (probability * 100).toFixed(2);
}

/**
 * Format class name (capitalize and replace underscores)
 */
function formatClassName(className) {
    return className
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}
