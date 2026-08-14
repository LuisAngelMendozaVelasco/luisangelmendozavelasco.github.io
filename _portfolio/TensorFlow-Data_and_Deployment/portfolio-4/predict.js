// predict.js - Object detection and prediction functionality

/**
 * Detect objects in the selected image
 */
async function detectObjects() {
    if (!model) {
        alert('Model not loaded. Please wait for the model to load!');
        return;
    }

    // Check if an image is actually loaded and displayed
    if (selectedImage.style.display === 'none' || imageSelect.selectedIndex === 0) {
        alert('Please select an image first!');
        return;
    }
    
    try {
        updateStatus('processing');
        detectBtn.disabled = true;
        addLog('Starting object detection...');
        
        // Perform classification
        const predictions = await model.classify(selectedImage, MODEL_CONFIG.topK);
        
        // Display topK results
        displayPredictions(predictions);
        
        addLog(`✓ Detection complete!`);
        updateStatus('ready');
        detectBtn.disabled = false;
        
    } catch (error) {
        addLog(`✗ Error during detection: ${error.message}`);
        updateStatus('error');
        detectBtn.disabled = false;
        console.error('Detection error:', error);
    }
}

/**
 * Display predictions in the UI
 */
function displayPredictions(predictions) {
    resultsContainer.style.display = 'block';
    predictionsList.innerHTML = '';
    
    if (!predictions || predictions.length === 0) {
        predictionsList.innerHTML = '<p class="text-muted">No predictions found</p>';
        return;
    }
    
    predictions.forEach((prediction, index) => {
        const className = formatClassName(prediction.className);
        const probability = formatProbability(prediction.probability);
        
        const predictionHTML = `
            <div class="prediction-item p-3 mb-3 bg-light rounded">
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <span class="fs-6 fw-bold text-dark">${index + 1}. ${className}</span>
                    <span class="badge bg-primary">${probability}%</span>
                </div>
                <div class="probability-bar rounded">
                    <div class="probability-fill" style="width: ${probability}%"></div>
                </div>
            </div>
        `;
        
        predictionsList.innerHTML += predictionHTML;
    });
}

/**
 * Handle image selection
 */
function handleImageSelect(event) {
    const selectedIndex = imageSelect.selectedIndex;
    
    if (selectedIndex === 0) {
        // No selection - clear any previous image
        selectedImage.onload = null;  // Remove event handlers to prevent onerror trigger
        selectedImage.onerror = null;
        selectedImage.src = '';
        selectedImage.style.display = 'none';
        imagePlaceholder.innerHTML = '<div id="imagePlaceholder" class="text-muted text-center py-5">Select an image from the dropdown above</div>';
        imagePlaceholder.style.display = 'block';
        resultsContainer.style.display = 'none';
        detectBtn.disabled = true;
        return;
    }
    
    const selectedSample = IMAGE_SAMPLES[selectedIndex - 1];
    
    addLog(`Selected image: "${selectedSample.name}"`);
    
    // Setup event handlers before loading image
    selectedImage.onload = () => {
        imagePlaceholder.style.display = 'none';
        selectedImage.style.display = 'block';
        detectBtn.disabled = false;
        addLog(`✓ Image loaded: "${selectedSample.name}"`);
    };
    
    selectedImage.onerror = () => {
        addLog(`✗ Error loading image: "${selectedSample.name}"`);
        imagePlaceholder.innerHTML = '<div class="text-danger text-center py-5">Failed to load image</div>';
        imagePlaceholder.style.display = 'block';
        selectedImage.style.display = 'none';
        detectBtn.disabled = true;
    };
    
    // Load image
    selectedImage.src = selectedSample.path;
    resultsContainer.style.display = 'none';
}
