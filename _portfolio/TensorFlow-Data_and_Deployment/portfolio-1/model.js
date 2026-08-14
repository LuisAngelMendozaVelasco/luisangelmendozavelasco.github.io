// model.js - Model building and training functions
let model = null;
let isTraining = false;
let trainingCancelled = false;
let lossHistory = [];
let accuracyHistory = [];

// Build model
function buildModel(learningRate, numOfFeatures) {
    if (model) {
        model.dispose();
    }

    model = tf.sequential({
        layers: [
            tf.layers.dense({
                inputShape: [numOfFeatures],
                activation: "sigmoid",
                units: 5
            }),
            tf.layers.dense({
                activation: "softmax",
                units: 3
            })
        ]
    });

    model.compile({
        loss: "categoricalCrossentropy",
        optimizer: tf.train.adam(learningRate),
        metrics: ['categoricalAccuracy']
    });
}

// Training function
async function train() {
    try {
        isTraining = true;
        trainingCancelled = false;
        startBtn.disabled = true;
        stopBtn.disabled = false;
        predictBtn.disabled = true;
        updateStatus('training');
        lossHistory = [];
        accuracyHistory = [];
        logPanel.innerHTML = '';

        const epochs = parseInt(document.getElementById('epochs').value);
        const learningRate = parseFloat(document.getElementById('learningRate').value);
        const batchSize = parseInt(document.getElementById('batchSize').value);
        
        addLog(`Epochs: ${epochs}, Learning Rate: ${learningRate}, Batch Size: ${batchSize}`);

        if (!trainingData) {
            await loadTrainingData();
        }

        const numOfFeatures = (await trainingData.columnNames()).length - 1;
        buildModel(learningRate, numOfFeatures);

        const convertedData = trainingData.map(({ xs, ys }) => {
            const labels = [
                ys.species == "setosa" ? 1 : 0,
                ys.species == "virginica" ? 1 : 0,
                ys.species == "versicolor" ? 1 : 0
            ];
            return { xs: Object.values(xs), ys: Object.values(labels) };
        }).batch(batchSize);

        await model.fitDataset(convertedData, {
            epochs: epochs,
            callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    if (trainingCancelled) {
                        throw new Error("Training cancelled!");
                    }

                    console.log(logs);

                    const loss = logs.loss.toFixed(4);
                    const accuracy = (logs.categoricalAccuracy * 100).toFixed(2);
                    lossHistory.push(loss);
                    accuracyHistory.push(accuracy);

                    currentEpoch.textContent = epoch + 1;
                    currentLoss.textContent = loss;
                    currentAccuracy.textContent = accuracy + '%';
                    progressFill.style.width = ((epoch + 1) / epochs * 100) + '%';

                    addLog(`Epoch ${epoch + 1}/${epochs} - Loss: ${loss}, Accuracy: ${accuracy}%`);

                    // Update chart
                    if (chart) {
                        chart.data.labels = lossHistory.map((_, i) => i + 1);
                        chart.data.datasets[0].data = lossHistory;
                        chart.data.datasets[1].data = accuracyHistory;
                        chart.update('none');
                    }

                    // Prevent UI blocking
                    await new Promise(resolve => setTimeout(resolve, 10));
                }
            }
        });

        addLog('Training completed successfully!');
        updateStatus('ready');
        predictBtn.disabled = false;
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