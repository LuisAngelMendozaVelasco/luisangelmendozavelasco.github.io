// chart.js - Chart initialization and management
let lossChart = null;
let accuracyChart = null;

// Initialize charts
function initCharts() {
    // Loss Chart
    const lossCtx = lossChartCanvas.getContext('2d');
    lossChart = new Chart(lossCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Training Loss',
                data: [],
                borderColor: '#ff0000',
                backgroundColor: 'rgba(255, 0, 0, 0.25)',
                tension: 0.4,
                fill: true,
                pointRadius: 0,
            },
            {
                label: 'Validation Loss',
                data: [],
                borderColor: '#0000ff',
                backgroundColor: 'rgba(0, 0, 255, 0.25)',
                tension: 0.4,
                fill: true,
                pointRadius: 0,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                },
                x: {}
            }
        }
    });

    // Accuracy Chart
    const accuracyCtx = accuracyChartCanvas.getContext('2d');
    accuracyChart = new Chart(accuracyCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Training Accuracy',
                data: [],
                borderColor: '#00ff00',
                backgroundColor: 'rgba(0, 255, 0, 0.25)',
                tension: 0.4,
                fill: true,
                pointRadius: 0,
            },
            {
                label: 'Validation Accuracy',
                data: [],
                borderColor: '#ff9800',
                backgroundColor: 'rgba(255, 152, 0, 0.25)',
                tension: 0.4,
                fill: true,
                pointRadius: 0,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                }
            },
            scales: {
                // y: {
                //     beginAtZero: true,
                //     max: 1,
                // },
                y: {
                    type: 'linear',
                    display: true,
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Accuracy (%)'
                    }
                },
                x: {}
            }
        }
    });
}