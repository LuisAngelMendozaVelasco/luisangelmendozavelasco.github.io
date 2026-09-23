// app.js - Main application initialization and event listeners
import {MnistData} from './data.js';

function initializeApp() {
    // Apply control configurations
    applyControlConfig();
    setupInputClamping();
    
    // Initialize charts
    initCharts();
    
    // Initialize canvas
    initializeCanvas();
    
    // Setup drawing listeners
    setupDrawingListeners();
    
    // Setup UI event listeners
    setupUIEventListeners();
    
    // Load training data asynchronously
    mnistData = new MnistData();
    mnistData.load().catch(error => {
        addLog('Error loading MNIST dataset: ' + error.message);
    });
    
    // Initialize tooltips
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

// Initialize the application when DOM is loaded
window.addEventListener('load', initializeApp);
