// model.js - MobileNet model management
let model = null;

/**
 * Load the pre-trained MobileNet model
 */
async function loadModel() {
    updateStatus('loading');
    addLog('Starting to load MobileNet model...');
    
    try {
        model = await mobilenet.load({
            version: MODEL_CONFIG.modelVersion,
            alpha: 1.0
        });
        
        addLog('✓ MobileNet model loaded successfully!');
        updateStatus('ready');
        imageSelect.disabled = false;
        
    } catch (error) {
        addLog(`✗ Error loading model: ${error.message}`);
        updateStatus('error');
        console.error('Model loading error:', error);
    }
}

