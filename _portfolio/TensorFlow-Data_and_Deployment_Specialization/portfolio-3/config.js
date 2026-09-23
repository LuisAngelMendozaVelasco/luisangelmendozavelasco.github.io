// config.js - Configuration constants and control settings
const classNames = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

const controlConfig = {
    epochs: { min: 5, max: 20, step: 5, value: 10, title: 'Range: 5-20 epochs' },
    learningRate: { min: 0.001, max: 0.1, step: 0.001, value: 0.01, title: 'Range: 0.001-0.1' },
    batchSize: { min: 128, max: 1024, step: 128, value: 128, title: 'Range: 128-1024 samples' }
};

const TRAIN_DATA_SIZE = 5500;
const TEST_DATA_SIZE = 1000;
const IMAGE_H = 28;
const IMAGE_W = 28;
const NUM_CLASSES = 10;
const IMAGE_SIZE = IMAGE_H * IMAGE_W;
const NUM_DATASET_ELEMENTS = TRAIN_DATA_SIZE + TEST_DATA_SIZE;

let mnistData = null;
