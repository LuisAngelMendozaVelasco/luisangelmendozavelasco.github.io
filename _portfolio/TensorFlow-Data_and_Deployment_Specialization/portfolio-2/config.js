// config.js - Configuration constants and control settings
const classNames = ["Benign", "Malignant"];

const controlConfig = {
    epochs: { min: 10, max: 100, step: 10, value: 50, title: 'Range: 10-100 epochs' },
    learningRate: { min: 0.001, max: 0.1, step: 0.001, value: 0.01, title: 'Range: 0.001-0.1' },
    batchSize: { min: 32, max: 64, step: 2, value: 32, title: 'Range: 32-64 samples' }
};