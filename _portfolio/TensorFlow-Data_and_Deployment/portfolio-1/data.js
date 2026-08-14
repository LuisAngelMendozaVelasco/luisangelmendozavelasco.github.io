// data.js - Data loading functions
let trainingData = null;

// Load training data
async function loadTrainingData() {
    const csvUrl = 'https://raw.githubusercontent.com/LuisAngelMendozaVelasco/TensorFlow-Data_and_Deployment_Specialization/refs/heads/main/Browser-based_Models_with_TensorFlow_js/Week1/Labs/data/iris.csv';
    trainingData = tf.data.csv(csvUrl, {
        columnConfigs: {
            species: {
                isLabel: true
            }
        }
    });
}