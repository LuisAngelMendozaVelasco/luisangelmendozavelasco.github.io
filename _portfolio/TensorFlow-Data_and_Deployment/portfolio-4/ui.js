// ui.js - User interface event handlers and state management

/**
 * Setup UI event listeners
 */
function setupUIEventListeners() {
    // Image selection
    imageSelect.addEventListener('change', handleImageSelect);
    
    // Detect button
    detectBtn.addEventListener('click', detectObjects);
}

/**
 * Populate image select dropdown
 */
function populateImageSelect() {
    IMAGE_SAMPLES.forEach(sample => {
        const option = document.createElement('option');
        option.value = sample.path;
        option.textContent = sample.name;
        imageSelect.appendChild(option);
    });
    
    addLog(`Loaded ${IMAGE_SAMPLES.length} image samples`);
}

/**
 * Initialize UI state
 */
function initializeUI() {
    // Disable detect button until model loads and image is selected
    detectBtn.disabled = true;
    imageSelect.disabled = true;
    
    // Populate image dropdown
    populateImageSelect();
    
    addLog('UI initialized');
}
