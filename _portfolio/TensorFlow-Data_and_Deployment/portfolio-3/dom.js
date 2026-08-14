// dom.js - DOM element references and basic DOM utilities
// Training section
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const resetBtn = document.getElementById('resetBtn');
const statusBadge = document.getElementById('statusBadge');
const currentEpoch = document.getElementById('currentEpoch');
const currentLoss = document.getElementById('currentLoss');
const currentAccuracy = document.getElementById('currentAccuracy');
const currentValAccuracy = document.getElementById('currentValAccuracy');
const progressFill = document.getElementById('progressFill');
const logPanel = document.getElementById('logPanel');
const lossChartCanvas = document.getElementById('lossChartCanvas');
const accuracyChartCanvas = document.getElementById('accuracyChartCanvas');

// Testing section - Drawing canvas
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const classifyBtn = document.getElementById('classifyBtn');
const clearBtn = document.getElementById('clearBtn');

// Testing section - Prediction results
const predictionContainer = document.getElementById('predictionContainer');
const predictionResult = document.getElementById('predictionResult');
const historyContainer = document.getElementById('historyContainer');

// Testing section - Samples
const loadSamplesBtn = document.getElementById('loadSamplesBtn');
const samplesContainer = document.getElementById('samplesContainer');
