package com.algorithms;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class SingleNeuronBackpropagationIntegrationTest {

    @Test
    void testBackpropagation() {
        SingleNeuronBackpropagation singleNeuronBackpropagation = new SingleNeuronBackpropagation();
        double[] input = {-3, -2, -1, 0, 1, 2, 3};
        double[] expectedOutput = {-7, -5, -3, -1, 1, 3, 5};
        double learningRate = 0.01;
        int epochs = 1000;
        double[] finalCoefficients = singleNeuronBackpropagation.trainLinearModel(input, expectedOutput, epochs, learningRate);
        double a = finalCoefficients[0];
        double b = finalCoefficients[1];
        double[] calculatedOutput = singleNeuronBackpropagation.forwardPropagate(a, input, b);

        for (int i = 0; i < calculatedOutput.length; i++) {
            calculatedOutput[i] = Math.round(calculatedOutput[i]);
            Assertions.assertEquals(expectedOutput[i], calculatedOutput[i]);
        }
    }
}