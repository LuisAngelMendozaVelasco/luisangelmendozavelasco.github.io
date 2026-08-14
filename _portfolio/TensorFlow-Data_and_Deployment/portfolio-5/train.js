import { data } from './data.js';
import { state } from './state.js';
import { loadMobilenet, createClassificationModel } from './model.js';
import { showTrainingMetrics, resetLossChart, updateTrainingMetrics, setTrainingStatus, updateButtonStates } from './ui.js';
import { validateSamples } from './samples.js';

export async function train() {
    if (!validateSamples()) return;

    state.isTraining = true;
    setTrainingStatus('training');
    updateButtonStates();

    showTrainingMetrics();
    data.lossHistory.length = 0;
    resetLossChart();

    if (!data.mobilenet) {
        data.mobilenet = await loadMobilenet();
    }

    data.dataset.ys = null;
    data.dataset.encodeLabels(3);

    data.model = createClassificationModel();
    data.model.compile({ optimizer: tf.train.adam(0.0001), loss: 'categoricalCrossentropy' });

    const epochs = 25;

    await data.model.fit(data.dataset.xs, data.dataset.ys, {
        epochs,
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                const loss = logs.loss.toFixed(5);
                data.lossHistory.push(loss);
                updateTrainingMetrics(epoch + 1, loss);
                updateLossChart(epoch + 1, loss);
            }
        }
    });

    state.isTraining = false;
    state.isModelTrained = true;
    setTrainingStatus('complete');
    updateButtonStates();
}

function updateLossChart(epoch, loss) {
    if (!data.lossChart) {
        const ctx = document.getElementById('lossChart').getContext('2d');
        data.lossChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Training Loss',
                    data: [],
                    borderColor: '#0d6efd',
                    backgroundColor: 'rgba(13, 110, 253, 0.1)',
                    borderWidth: 2,
                    tension: 0.1,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }

    data.lossChart.data.labels.push(`Epoch ${epoch}`);
    data.lossChart.data.datasets[0].data.push(loss);
    data.lossChart.update();
}
