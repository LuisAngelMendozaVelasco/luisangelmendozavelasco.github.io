// ui.js - UI event handlers and interactions
// Test case buttons
document.querySelectorAll('.test-case-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
        document.querySelectorAll('.test-case-btn').forEach(b => {
            b.classList.remove('btn-primary');
            b.classList.add('btn-outline-primary');
        });
        e.target.closest('.test-case-btn').classList.remove('btn-outline-primary');
        e.target.closest('.test-case-btn').classList.add('btn-primary');

        const caseKey = e.target.closest('.test-case-btn').dataset.case;
        const values = testCases[caseKey];
        document.getElementById('sepalLength').value = values[0];
        document.getElementById('sepalWidth').value = values[1];
        document.getElementById('petalLength').value = values[2];
        document.getElementById('petalWidth').value = values[3];
    });
});

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