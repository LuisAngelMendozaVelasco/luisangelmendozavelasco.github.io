// data.js - Data loading functions
let trainingData = null;
let testingData = null;

// Load training and testing data
async function loadTrainingData() {
    const trainingUrl = 'https://raw.githubusercontent.com/LuisAngelMendozaVelasco/TensorFlow-Data_and_Deployment_Specialization/refs/heads/main/Browser-based_Models_with_TensorFlow_js/Week1/Labs/data/wdbc-train.csv';
    const testingUrl = 'https://raw.githubusercontent.com/LuisAngelMendozaVelasco/TensorFlow-Data_and_Deployment_Specialization/refs/heads/main/Browser-based_Models_with_TensorFlow_js/Week1/Labs/data/wdbc-test.csv';

    trainingData = tf.data.csv(trainingUrl, {
        columnConfigs: {
            diagnosis: {
                isLabel: true
            }
        }
    });

    testingData = tf.data.csv(testingUrl, {
        columnConfigs: {
            diagnosis: {
                isLabel: true
            }
        }
    });
}