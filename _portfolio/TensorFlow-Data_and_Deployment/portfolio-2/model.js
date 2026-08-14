// model.js - Model building and training functions
let model = null;
let isTraining = false;
let trainingCancelled = false;
let lossHistory = [];
let accuracyHistory = [];
let valLossHistory = [];
let valAccuracyHistory = [];

// Build model
function buildModel(learningRate, numOfFeatures) {
    if (model) {
        model.dispose();
    }

    model = tf.sequential({
        layers: [
            tf.layers.dense({
                inputShape: [numOfFeatures],
                activation: "relu",
                units: 32
            }),
            tf.layers.dense({
                activation: "relu",
                units: 64
            }),
            tf.layers.dense({
                activation: "relu",
                units: 32
            }),
            tf.layers.dense({
                activation: "sigmoid",
                units: 1
            })
        ]
    });

    model.compile({
        loss: "binaryCrossentropy",
        optimizer: tf.train.rmsprop(learningRate),
        metrics: ['accuracy']
    });
}

// Training function
async function train() {
    try {
        isTraining = true;
        trainingCancelled = false;
        startBtn.disabled = true;
        stopBtn.disabled = false;
        updateStatus('training');
        lossHistory = [];
        accuracyHistory = [];
        valLossHistory = [];
        valAccuracyHistory = [];
        logPanel.innerHTML = '';

        const epochs = parseInt(document.getElementById('epochs').value);
        const learningRate = parseFloat(document.getElementById('learningRate').value);
        const batchSize = parseInt(document.getElementById('batchSize').value);

        addLog(`Epochs: ${epochs}, Learning Rate: ${learningRate}, Batch Size: ${batchSize}`);

        if (!trainingData || !testingData) {
            await loadTrainingData();
        }

        const numOfFeatures = (await trainingData.columnNames()).length - 1;
        buildModel(learningRate, numOfFeatures);

        const convertedTrainingData = trainingData.map(({ xs, ys }) => {
            return { xs: Object.values(xs), ys: Object.values(ys) };
        }).batch(batchSize);

        const convertedTestingData = testingData.map(({ xs, ys }) => {
            return { xs: Object.values(xs), ys: Object.values(ys) };
        }).batch(batchSize);

        await model.fitDataset(convertedTrainingData, {
            epochs: epochs,
            validationData: convertedTestingData,
            callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    if (trainingCancelled) {
                        throw new Error("Training cancelled!");
                    }
                    
                    const loss = logs.loss.toFixed(4);
                    const accuracy = (logs.acc * 100).toFixed(2);
                    const valLoss = logs.val_loss.toFixed(4);
                    const valAccuracy = (logs.val_acc * 100).toFixed(2);
                    
                    lossHistory.push(loss);
                    accuracyHistory.push(accuracy);
                    valLossHistory.push(valLoss);
                    valAccuracyHistory.push(valAccuracy);

                    currentEpoch.textContent = epoch + 1;
                    currentLoss.textContent = loss;
                    currentAccuracy.textContent = accuracy + '%';
                    currentValLoss.textContent = valLoss;
                    currentValAccuracy.textContent = valAccuracy + '%';

                    progressFill.style.width = ((epoch + 1) / epochs * 100) + '%';

                    addLog(`Epoch ${epoch + 1}/${epochs} - Loss: ${loss}, Accuracy: ${accuracy}% | Validation Loss: ${valLoss}, Validation Accuracy: ${valAccuracy}%`);

                    // Update charts
                    if (lossChart) {
                        lossChart.data.labels = lossHistory.map((_, i) => i + 1);
                        lossChart.data.datasets[0].data = lossHistory;
                        lossChart.data.datasets[1].data = valLossHistory;
                        lossChart.update('none');
                    }
                    if (accuracyChart) {
                        accuracyChart.data.labels = accuracyHistory.map((_, i) => i + 1);
                        accuracyChart.data.datasets[0].data = accuracyHistory;
                        accuracyChart.data.datasets[1].data = valAccuracyHistory;
                        accuracyChart.update('none');
                    }

                    // Prevent UI blocking
                    await new Promise(resolve => setTimeout(resolve, 10));
                }
            }
        });

        addLog('Training completed successfully!');
        updateStatus('ready');
    } catch (error) {
        if (error.message !== "Training cancelled!") {
            addLog(`Error: ${error.message}`);
        } else {
            addLog('Training cancelled by user!');
        }
        updateStatus('idle');
    } finally {
        isTraining = false;
        startBtn.disabled = false;
        stopBtn.disabled = true;
    }
}