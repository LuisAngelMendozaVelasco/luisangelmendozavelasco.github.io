// app.js - Main application initialization and event listeners

// ==========================================
// EVENT HANDLERS
// ==========================================

/**
 * Handles the start training button click
 */
function handleStartTraining() {
    train();
}

/**
 * Handles the stop training button click
 */
function handleStopTraining() {
    trainingCancelled = true;
    stopBtn.disabled = true;
}

/**
 * Handles the reset button click - resets the entire application state
 */
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

    // Reset chart
    resetChart();
}

/**
 * Handles the predict button click
 */
function handlePredict() {
    predict();
}

// ==========================================
// STATE MANAGEMENT
// ==========================================

/**
 * Resets all training-related state variables
 */
function resetTrainingState() {
    lossHistory = [];
    predictionHistory = [];
}

/**
 * Resets all UI elements to their initial state
 */
function resetUI() {
    currentEpoch.textContent = '-';
    currentLoss.textContent = '-';
    progressFill.style.width = '0%';
    logPanel.innerHTML = '';
    resultContainer.style.display = 'none';
    historyContainer.innerHTML = '';

    updateStatus('idle');

    startBtn.disabled = false;
    stopBtn.disabled = true;
    predictBtn.disabled = true;
}

/**
 * Resets the training chart
 */
function resetChart() {
    if (chart) {
        chart.data.labels = [];
        chart.data.datasets[0].data = [];
        chart.update();
    }
}

// ==========================================
// INITIALIZATION
// ==========================================

/**
 * Initializes the application when the page loads
 */
function initializeApp() {
    applyControlConfig();
    setupInputClamping();
    initChart();
    loadTrainingData();
    initializeTooltips();
}

/**
 * Initializes Bootstrap tooltips for elements with title attributes
 */
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[title]'));
    const tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// ==========================================
// EVENT LISTENERS SETUP
// ==========================================

// Set up event listeners
startBtn.addEventListener('click', handleStartTraining);
stopBtn.addEventListener('click', handleStopTraining);
resetBtn.addEventListener('click', handleReset);
predictBtn.addEventListener('click', handlePredict);

// Initialize the application
window.addEventListener('load', initializeApp);