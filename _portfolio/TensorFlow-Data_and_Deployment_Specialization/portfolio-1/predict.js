// predict.js - Prediction functions
let predictionHistory = [];

// Prediction function
function predict() {
    if (!model) {
        alert('Please train the model first!');
        return;
    }

    const sepalLength = parseFloat(document.getElementById('sepalLength').value);
    const sepalWidth = parseFloat(document.getElementById('sepalWidth').value);
    const petalLength = parseFloat(document.getElementById('petalLength').value);
    const petalWidth = parseFloat(document.getElementById('petalWidth').value);

    if (isNaN(sepalLength) || isNaN(sepalWidth) || isNaN(petalLength) || isNaN(petalWidth)) {
        alert('Please enter valid numbers for all fields');
        return;
    }

    const testInput = tf.tensor2d([[sepalLength, sepalWidth, petalLength, petalWidth]]);
    const prediction = model.predict(testInput);
    const predictionData = prediction.dataSync();
    const predictedIndex = tf.argMax(prediction, axis = 1).dataSync()[0];
    const predictedClass = classNames[predictedIndex];
    const confidences = Array.from(predictionData);

    // Display results
    let resultHTML = `<div class="class-prediction"><strong style="font-size: 1.3em; color: #ff0000;">${predictedClass}</strong></div>`;
    resultHTML += '<div class="result-label" style="margin-top: 15px;"><strong>Confidence Scores</strong></div>';

    classNames.forEach((className, index) => {
        const confidence = (confidences[index] * 100).toFixed(2);
        resultHTML += `
            <div class="class-prediction">
                <div class="class-name">${className}</div>
                <div class="class-confidence">
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${confidence}%"></div>
                    </div>
                    <div class="confidence-value">${confidence}%</div>
                </div>
            </div>
        `;
    });

    resultContent.innerHTML = resultHTML;
    resultContainer.style.display = 'block';

    // Add to history
    const historyEntry = `<strong>${predictedClass}</strong> (${(confidences[predictedIndex] * 100).toFixed(2)}%) - Input: [${sepalLength}, ${sepalWidth}, ${petalLength}, ${petalWidth}]`;
    predictionHistory.unshift(historyEntry);
    if (predictionHistory.length > 10) {
        predictionHistory.pop();
    }

    updatePredictionHistory();

    // Cleanup
    testInput.dispose();
    prediction.dispose();
}

function updatePredictionHistory() {
    historyContainer.innerHTML = predictionHistory.map((entry, index) =>
        `<div class="history-item">${index + 1}. ${entry}</div>`
    ).join('');
}