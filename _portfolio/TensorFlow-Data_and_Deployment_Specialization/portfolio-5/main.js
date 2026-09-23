import { data } from './data.js';
import { state } from './state.js';
import { updateButtonStates, setPredictionText } from './ui.js';
import { train } from './train.js';
import { startPredicting, stopPredicting } from './predict.js';
import { incrementSample } from './samples.js';
import { loadMobilenet } from './model.js';

async function init() {
    updateButtonStates();
    data.webcam = new Webcam(document.getElementById('wc'));
    await data.webcam.setup();
    data.mobilenet = await loadMobilenet();
    tf.tidy(() => data.mobilenet.predict(data.webcam.capture()));
    setPredictionText('Waiting for predictions...');

    // Attach UI event listeners for sample buttons
    const sampleLabels = ['rock', 'paper', 'scissors'];
    sampleLabels.forEach((label, i) => {
        document.getElementById(String(i)).addEventListener('click', () => {
            incrementSample(label);
            const img = data.webcam.capture();
            data.dataset.addExample(data.mobilenet.predict(img), i);
            updateButtonStates();
        });
    });

    document.getElementById('train').addEventListener('click', () => train());
    document.getElementById('startPredicting').addEventListener('click', () => startPredicting());
    document.getElementById('stopPredicting').addEventListener('click', () => stopPredicting());
}

init();
