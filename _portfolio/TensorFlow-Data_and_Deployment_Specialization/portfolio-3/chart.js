// chart.js - Chart initialization and management
let lossChart = null;
let accuracyChart = null;

// Initialize loss chart
function initLossChart() {
    const ctx = lossChartCanvas.getContext('2d');
    lossChart = new Chart(ctx, {
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
                pointRadius: 0
            },
            {
                label: 'Validation Loss',
                data: [],
                borderColor: '#0000ff',
                backgroundColor: 'rgba(0, 0, 255, 0.25)',
                tension: 0.4,
                fill: true,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                }
            },
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Loss'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Epoch'
                    }
                }
            }
        }
    });
}

// Initialize accuracy chart
function initAccuracyChart() {
    const ctx = accuracyChartCanvas.getContext('2d');
    accuracyChart = new Chart(ctx, {
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
                pointRadius: 0
            },
            {
                label: 'Validation Accuracy',
                data: [],
                borderColor: '#ff9800',
                backgroundColor: 'rgba(255, 152, 0, 0.25)',
                tension: 0.4,
                fill: true,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                }
            },
            scales: {
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
                x: {
                    title: {
                        display: true,
                        text: 'Epoch'
                    }
                }
            }
        }
    });
}

// Initialize both charts
function initCharts() {
    initLossChart();
    initAccuracyChart();
}

function resetCharts() {
    if (lossChart) {
        lossChart.data.labels = [];
        lossChart.data.datasets[0].data = [];
        lossChart.data.datasets[1].data = [];
        lossChart.update();
    }
    if (accuracyChart) {
        accuracyChart.data.labels = [];
        accuracyChart.data.datasets[0].data = [];
        accuracyChart.data.datasets[1].data = [];
        accuracyChart.update();
    }
}
