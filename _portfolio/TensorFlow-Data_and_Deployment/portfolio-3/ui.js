// ui.js - UI event handlers and interactions
function setupUIEventListeners() {
    // Training button listeners
    startBtn.addEventListener('click', handleStartTraining);
    stopBtn.addEventListener('click', handleStopTraining);
    resetBtn.addEventListener('click', handleReset);

    // Drawing and prediction button listeners
    classifyBtn.addEventListener('click', classify);
    clearBtn.addEventListener('click', clearCanvas);
    loadSamplesBtn.addEventListener('click', loadRandomSamples);
}

function handleStartTraining() {
    train();
}

function handleStopTraining() {
    trainingCancelled = true;
    stopBtn.disabled = true;
}

function handleReset() {
    // Dispose of the current model
    if (model) {
        model.dispose();
        model = null;
    }

    // Reset training state
    resetTrainingState();

    // Reset UI elements
    resetUI();

    // Reset charts
    resetCharts();
}

function resetTrainingState() {
    lossHistory = [];
    valLossHistory = [];
    accuracyHistory = [];
    valAccuracyHistory = [];
    predictionHistory = [];
}

function resetUI() {
    currentEpoch.textContent = '-';
    currentLoss.textContent = '-';
    currentAccuracy.textContent = '-';
    currentValLoss.textContent = '-';
    currentValAccuracy.textContent = '-';
    progressFill.style.width = '0%';
    logPanel.innerHTML = '';
    predictionContainer.style.display = 'none';
    historyContainer.innerHTML = '';

    updateStatus('idle');

    startBtn.disabled = false;
    stopBtn.disabled = true;
    classifyBtn.disabled = true;
}
