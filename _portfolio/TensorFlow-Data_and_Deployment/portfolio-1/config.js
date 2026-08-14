// config.js - Configuration constants and control settings
const classNames = ["Setosa", "Virginica", "Versicolor"];

const testCases = {
    // sepalLength, sepalWidth, petalLength, petalWidth
    setosa: [4.4, 2.9, 1.4, 0.2],
    versicolor: [6.4, 3.2, 4.5, 1.5],
    virginica: [5.8, 2.7, 5.1, 1.9]
};

const controlConfig = {
    epochs: { min: 50, max: 500, step: 50, value: 100, title: 'Range: 50-500 epochs' },
    learningRate: { min: 0.001, max: 0.1, step: 0.001, value: 0.1, title: 'Range: 0.001-0.1' },
    batchSize: { min: 32, max: 64, step: 2, value: 32, title: 'Range: 32-64 samples' },
    sepalLength: { min: 4.0, max: 8.0, step: 0.1, value: 4.4, title: 'Range: 4.0-8.0 cm' },
    sepalWidth: { min: 2.0, max: 5.0, step: 0.1, value: 2.9, title: 'Range: 2.0-5.0 cm' },
    petalLength: { min: 1.0, max: 7.0, step: 0.1, value: 1.4, title: 'Range: 1.0-7.0 cm' },
    petalWidth: { min: 0.1, max: 3.0, step: 0.1, value: 0.2, title: 'Range: 0.1-3.0 cm' }
};