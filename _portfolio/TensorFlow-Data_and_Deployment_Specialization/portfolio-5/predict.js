import { data } from './data.js';
import { state } from './state.js';
import { setPredictionText, updateButtonStates } from './ui.js';

export async function predict() {
    while (state.isPredicting) {
        const predictedClass = tf.tidy(() => {
            const img = data.webcam.capture();
            const activation = data.mobilenet.predict(img);
            const predictions = data.model.predict(activation);
            return predictions.as1D().argMax();
        });

        const classId = (await predictedClass.data())[0];

        if (state.isPredicting) {
            let predictionText = '';
            switch (classId) {
                case 0:
                    predictionText = 'I see Rock';
                    break;
                case 1:
                    predictionText = 'I see Paper';
                    break;
                case 2:
                    predictionText = 'I see Scissors';
                    break;
            }
            setPredictionText(predictionText);
        }

        predictedClass.dispose();
        await tf.nextFrame();
    }
}

export function startPredicting() {
    state.isPredicting = true;
    updateButtonStates();
    predict();
}

export function stopPredicting() {
    state.isPredicting = false;
    updateButtonStates();
    setPredictionText('Waiting for predictions...');
}
