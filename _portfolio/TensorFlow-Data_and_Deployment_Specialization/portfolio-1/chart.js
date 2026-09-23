// chart.js - Chart initialization and management
let chart = null;

// Initialize chart
function initChart() {
    const ctx = lossChartCanvas.getContext('2d');
    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Loss',
                data: [],
                borderColor: '#ff0000',
                backgroundColor: 'rgba(255, 0, 0, 0.25)',
                tension: 0.4,
                fill: true,
                pointRadius: 0,
                yAxisID: 'y'
            },
            {
                label: 'Accuracy (%)',
                data: [],
                borderColor: '#00aa00',
                backgroundColor: 'rgba(0, 170, 0, 0.25)',
                tension: 0.4,
                fill: true,
                pointRadius: 0,
                yAxisID: 'y1'
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
                    type: 'linear',
                    display: true,
                    position: 'left',
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Loss'
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Accuracy (%)'
                    },
                    grid: {
                        drawOnChartArea: false,
                    }
                },
                x: {}
            }
        }
    });
}