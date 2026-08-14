// predict.js - Prediction and drawing functionality
let predictionHistory = [];
let pos = { x: 0, y: 0 };
let isDrawing = false;

// Mouse position tracking
function setPosition(e) {
    const rect = canvas.getBoundingClientRect();
    pos.x = e.clientX - rect.left;
    pos.y = e.clientY - rect.top;
}

// Drawing function
function draw(e) {
    if (!isDrawing || e.buttons !== 1) return;
    
    ctx.beginPath();
    ctx.lineWidth = 15;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = 'white';
    ctx.moveTo(pos.x, pos.y);
    
    setPosition(e);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
}

// Clear canvas
function clearCanvas() {
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, 280, 280);
    predictionContainer.style.display = 'none';
}

// Classify drawn digit
function classify() {
    if (!model) {
        alert('Please train the model first!');
        return;
    }

    try {
        // Get pixels from canvas and convert to tensor
        let raw = tf.browser.fromPixels(canvas, 1);
        
        // Resize to 28x28
        let resized = tf.image.resizeBilinear(raw, [28, 28]);
        
        // Normalize to 0-1 range
        let normalized = resized.div(255.0);
        
        // Reshape for model input
        let input = normalized.expandDims(0);
        
        // Make prediction
        const prediction = model.predict(input);
        const predictionData = prediction.dataSync();
        const predictedIndex = Array.from(predictionData).indexOf(Math.max(...predictionData));
        const confidence = predictionData[predictedIndex];

        // Display results
        let resultHTML = `<div class="fs-5 fw-bold text-primary my-3">Predicted Digit: ${predictedIndex}</div>`;
        resultHTML += `<div class="mb-3"><strong>Confidence: ${(confidence * 100).toFixed(2)}%</strong></div>`;
        resultHTML += '<div><strong>Confidence Scores:</strong></div>';

        for (let i = 0; i < 10; i++) {
            const conf = (predictionData[i] * 100).toFixed(2);
            resultHTML += `
                <div class="d-flex align-items-center my-2 gap-2">
                    <div class="fw-bold" style="min-width: 30px;">${i}</div>
                    <div class="flex-grow-1" style="height: 25px; background-color: #e9ecef; border-radius: 4px; overflow: hidden;">
                        <div class="confidence-fill" style="width: ${conf}%">${conf}%</div>
                    </div>
                </div>
            `;
        }

        predictionResult.innerHTML = resultHTML;
        predictionContainer.style.display = 'block';

        // Add to history
        const historyEntry = `<li class="list-group-item"><strong>${predictedIndex}</strong> (${(confidence * 100).toFixed(2)}%)</li>`;
        historyContainer.insertAdjacentHTML('afterbegin', historyEntry);
        if (historyContainer.children.length > 10) {
            historyContainer.removeChild(historyContainer.lastChild);
        }

        // Cleanup
        raw.dispose();
        resized.dispose();
        normalized.dispose();
        input.dispose();
        prediction.dispose();

    } catch (error) {
        alert('Error classifying image: ' + error.message);
        console.error(error);
    }
}

// Helper function to draw MNIST image data on canvas
function drawMnistImage(imageData, sampleCanvas) {
    sampleCanvas.width = 28;
    sampleCanvas.height = 28;
    const ctx = sampleCanvas.getContext('2d');
    
    const canvasImageData = ctx.createImageData(28, 28);
    const data = canvasImageData.data;
    
    // Convert grayscale image (0-1 range) to RGBA
    for (let i = 0; i < imageData.length; i++) {
        const pixelValue = Math.round(imageData[i] * 255);
        const j = i * 4;
        data[j + 0] = pixelValue;      // R
        data[j + 1] = pixelValue;      // G
        data[j + 2] = pixelValue;      // B
        data[j + 3] = 255;             // A
    }
    
    ctx.putImageData(canvasImageData, 0, 0);
}

// Helper function to decode one-hot encoded labels
function decodeLabel(labelIndex) {
    if (!mnistData || !mnistData.datasetLabels) {
        return 0;
    }
    
    // Labels are one-hot encoded: 10 bytes per image
    const labelStart = labelIndex * NUM_CLASSES;
    let maxValue = 0;
    let maxIndex = 0;
    
    for (let i = 0; i < NUM_CLASSES; i++) {
        const value = mnistData.datasetLabels[labelStart + i];
        if (value > maxValue) {
            maxValue = value;
            maxIndex = i;
        }
    }
    
    return maxIndex;
}

// Helper function to get random samples without modifying data.js
function getRandomMnistSample() {
    if (!mnistData || !mnistData.datasetImages || !mnistData.datasetLabels) {
        return null;
    }
    
    const randomIndex = Math.floor(Math.random() * NUM_DATASET_ELEMENTS);
    const image = mnistData.datasetImages.slice(randomIndex * IMAGE_SIZE, (randomIndex + 1) * IMAGE_SIZE);
    const label = decodeLabel(randomIndex);
    return { image, label };
}

// Load and display random samples from MNIST
async function loadRandomSamples() {
    if (!mnistData || !mnistData.datasetImages) {
        alert('Dataset not loaded yet!');
        return;
    }

    try {
        loadSamplesBtn.disabled = true;
        samplesContainer.innerHTML = '';

        const numSamples = 10;

        for (let i = 0; i < numSamples; i++) {
            const sample = getRandomMnistSample();
            if (!sample) continue;
            
            const { image, label } = sample;
            
            // Create container for sample
            const container = document.createElement('div');
            container.className = 'd-inline-block text-center';
            container.style.margin = '5px';
            
            // Create 28x28 canvas for the image
            const sampleCanvas = document.createElement('canvas');
            sampleCanvas.className = 'digit-example';
            drawMnistImage(image, sampleCanvas);
            
            // Create scaled version for display (100x100)
            const scaledCanvas = document.createElement('canvas');
            scaledCanvas.className = 'digit-example';
            scaledCanvas.width = 100;
            scaledCanvas.height = 100;
            scaledCanvas.title = `Label: ${label}`;
            scaledCanvas.style.cursor = 'pointer';
            
            const scaledCtx = scaledCanvas.getContext('2d');
            scaledCtx.imageSmoothingEnabled = false;
            scaledCtx.drawImage(sampleCanvas, 0, 0, 28, 28, 0, 0, 100, 100);
            
            // Add label text
            const labelDiv = document.createElement('div');
            labelDiv.className = 'mt-1 small fw-bold';
            labelDiv.textContent = `${label}`;
            
            container.appendChild(scaledCanvas);
            container.appendChild(labelDiv);
            samplesContainer.appendChild(container);
        }

        loadSamplesBtn.disabled = false;

    } catch (error) {
        samplesContainer.innerHTML = '<p class="text-danger">Error loading samples: ' + error.message + '</p>';
        loadSamplesBtn.disabled = false;
        console.error(error);
    }
}

// Setup canvas drawing event listeners
function setupDrawingListeners() {
    canvas.addEventListener('mousedown', (e) => {
        isDrawing = true;
        setPosition(e);
    });

    canvas.addEventListener('mousemove', draw);

    canvas.addEventListener('mouseup', () => {
        isDrawing = false;
    });

    canvas.addEventListener('mouseleave', () => {
        isDrawing = false;
    });

    canvas.addEventListener('touchstart', (e) => {
        e.preventDefault();
        isDrawing = true;
        const touch = e.touches[0];
        const mouseEvent = new MouseEvent('mousemove', {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        setPosition(mouseEvent);
    });

    canvas.addEventListener('touchmove', (e) => {
        e.preventDefault();
        const touch = e.touches[0];
        const mouseEvent = new MouseEvent('mousemove', {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        draw(mouseEvent);
    });

    canvas.addEventListener('touchend', (e) => {
        e.preventDefault();
        isDrawing = false;
    });
}
