// app.js - Main application initialization

async function initializeApp() {
    addLog('Initializing Object Detection App...');
    
    // Initialize UI
    initializeUI();
    
    // Setup event listeners
    setupUIEventListeners();
    
    // Load model asynchronously
    await loadModel();
    
    // Only log ready when model is successfully loaded
    if (model) {
        addLog('Application ready!');
    }
}

// Initialize app when DOM is loaded
window.addEventListener('load', initializeApp);
