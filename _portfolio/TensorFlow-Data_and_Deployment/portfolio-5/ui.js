import { state } from './state.js';
import { data } from './data.js';

export function updateButtonStates() {
    document.getElementById('train').disabled = state.isPredicting;
    document.getElementById('startPredicting').disabled = !state.isModelTrained || state.isPredicting;
    document.getElementById('stopPredicting').disabled = !state.isPredicting;
    document.getElementById('0').disabled = state.isPredicting;
    document.getElementById('1').disabled = state.isPredicting;
    document.getElementById('2').disabled = state.isPredicting;
}

export function setTrainingStatus(status) {
    const badge = document.getElementById('trainingStatus');
    if (status === 'training') {
        badge.className = 'badge bg-warning';
        badge.innerText = 'Training model...';
    } else if (status === 'complete') {
        badge.className = 'badge bg-success';
        badge.innerText = 'Trained model';
    } else {
        badge.className = 'badge bg-secondary';
        badge.innerText = 'Untrained model';
    }
}

export function showTrainingMetrics() {
    const metrics = document.getElementById('trainingMetrics');
    metrics.style.display = 'block';
    metrics.scrollIntoView({ behavior: 'smooth', block: 'start' });
    metrics.focus();
}

export function resetLossChart() {
    if (data.lossChart) {
        data.lossChart.data.labels = [];
        data.lossChart.data.datasets[0].data = [];
        data.lossChart.update();
    }
}

export function updateTrainingMetrics(epoch, loss) {
    document.getElementById('currentEpoch').innerText = epoch;
    document.getElementById('currentLoss').innerText = loss;
    document.getElementById('progressFill').style.width = `${(epoch / 10) * 100}%`;
}

export function setPredictionText(text) {
    document.getElementById('prediction').innerText = text;
}

export function setSampleCount(type, value) {
    document.getElementById(type).innerText = value;
}
