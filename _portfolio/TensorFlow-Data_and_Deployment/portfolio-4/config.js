// config.js - Configuration and image samples
const IMAGE_SAMPLES = [
    {
        name: 'Coffee Cup',
        path: './image/cup.jpg'
    },
    {
        name: 'Car',
        path: './image/car.jpg'
    },
    {
        name: 'Television',
        path: './image/tv.jpg'
    },
    {
        name: 'Cat',
        path: './image/cat.jpg'
    },
    {
        name: 'Pig',
        path: './image/pig.jpg'
    },
    {
        name: 'Dog',
        path: './image/dog.jpg'
    }
];

// MobileNet model settings
const MODEL_CONFIG = {
    modelVersion: 2,
    topK: 3  // Number of top predictions to show
};

// Logging settings
const LOGGING = {
    maxLogs: 25
};
