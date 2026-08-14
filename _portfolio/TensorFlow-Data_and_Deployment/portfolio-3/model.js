// model.js - Model building and training functions

let model = null;
let isTraining = false;
let trainingCancelled = false;
let lossHistory = [];
let valLossHistory = [];
let accuracyHistory = [];
let valAccuracyHistory = [];

// Build model
function buildModel(learningRate) {
    if (model) {
        model.dispose();
    }

    model = tf.sequential({
        layers: [
            tf.layers.conv2d({
                inputShape: [IMAGE_H, IMAGE_W, 1],
                kernelSize: 3,
                filters: 8,
                activation: 'relu'
            }),
            tf.layers.maxPooling2d({ poolSize: [2, 2] }),
            tf.layers.conv2d({
                kernelSize: 3,
                filters: 16,
                activation: 'relu'
            }),
            tf.layers.maxPooling2d({ poolSize: [2, 2] }),
            tf.layers.flatten(),
            tf.layers.dense({
                units: 128,
                activation: 'relu'
            }),
            tf.layers.dense({
                units: NUM_CLASSES,
                activation: 'softmax'
            })
        ]
    });

    model.compile({
        optimizer: tf.train.adam(learningRate),
        loss: 'categoricalCrossentropy',
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
        classifyBtn.disabled = true;
        updateStatus('training');
        lossHistory = [];
        valLossHistory = [];
        accuracyHistory = [];
        valAccuracyHistory = [];
        logPanel.innerHTML = '';

        const epochs = parseInt(document.getElementById('epochs').value);
        const learningRate = parseFloat(document.getElementById('learningRate').value);
        const batchSize = parseInt(document.getElementById('batchSize').value);
        
        addLog(`Epochs: ${epochs}, Learning Rate: ${learningRate}, Batch Size: ${batchSize}`);

        buildModel(learningRate);

        // Prepare training data
        const [trainXs, trainYs] = tf.tidy(() => {
            const d = mnistData.nextTrainBatch(TRAIN_DATA_SIZE);
            return [
                d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
                d.labels
            ];
        });

        // Prepare validation data
        const [valXs, valYs] = tf.tidy(() => {
            const d = mnistData.nextTestBatch(TEST_DATA_SIZE);
            return [
                d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
                d.labels
            ];
        });

        await model.fit(trainXs, trainYs, {
            batchSize: batchSize,
            validationData: [valXs, valYs],
            epochs: epochs,
            shuffle: true,
            callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    if (trainingCancelled) {
                        throw new Error("Training cancelled!");
                    }

                    const loss = logs.loss.toFixed(4);
                    const accuracy = (logs.acc * 100).toFixed(2);
                    const valLoss = logs.val_loss.toFixed(4);
                    const valAccuracy = (logs.val_acc * 100).toFixed(2);

                    lossHistory.push(parseFloat(loss));
                    accuracyHistory.push(parseFloat(accuracy));
                    valLossHistory.push(parseFloat(valLoss));
                    valAccuracyHistory.push(parseFloat(valAccuracy));

                    currentEpoch.textContent = epoch + 1;
                    currentLoss.textContent = loss;
                    currentAccuracy.textContent = accuracy + '%';
                    currentValLoss.textContent = valLoss;
                    currentValAccuracy.textContent = valAccuracy + '%';
                    progressFill.style.width = ((epoch + 1) / epochs * 100) + '%';

                    addLog(`Epoch ${epoch + 1}/${epochs} - Loss: ${loss}, Accuracy: ${accuracy}%, Val Loss: ${valLoss}, Val Accuracy: ${valAccuracy}%`);

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
        classifyBtn.disabled = false;

        // Cleanup
        trainXs.dispose();
        trainYs.dispose();
        valXs.dispose();
        valYs.dispose();

    } catch (error) {
        if (error.message !== "Training cancelled!") {
            addLog(`Error: ${error.message}`);
            console.error(error);
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
