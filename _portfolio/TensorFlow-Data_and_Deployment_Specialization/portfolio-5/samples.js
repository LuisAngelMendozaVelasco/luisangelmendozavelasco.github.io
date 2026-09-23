import { data } from './data.js';
import { setSampleCount } from './ui.js';

export function validateSamples() {
    if (data.rockSamples < 10 || data.paperSamples < 10 || data.scissorsSamples < 10) {
        alert(`You need at least 10 samples of each hand shape!\n\nCurrent samples:\nRock: ${data.rockSamples}\nPaper: ${data.paperSamples}\nScissors: ${data.scissorsSamples}`);
        return false;
    }
    return true;
}

export function incrementSample(type) {
    if (type === 'rock') {
        data.rockSamples += 1;
        setSampleCount('rocksamples', data.rockSamples);
    } else if (type === 'paper') {
        data.paperSamples += 1;
        setSampleCount('papersamples', data.paperSamples);
    } else if (type === 'scissors') {
        data.scissorsSamples += 1;
        setSampleCount('scissorssamples', data.scissorsSamples);
    }
}
