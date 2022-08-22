package com.algorithms;

import org.junit.jupiter.api.Test;

import static java.lang.Math.*;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DoubleNeuronBackpropagationIntegrationTest {

    @Test
    void testBackpropagation() {
        DoubleNeuronBackpropagation doubleNeuronBackpropagation = new DoubleNeuronBackpropagation();
        double[] input = {-3, -2, -1, 0, 1, 2, 3};
        double[] expectedOutput = {-0.5, 0, 0.5, 1, 1.5, 2, 2.5};
        double learningRate = 0.001;
        int epochs = 10000;

        double w1 = random() + 1;
        double w2 = random() + 1;
        double b1 = random() + 1;
        double b2 = random() + 1;

        double[][] parameters = {{w1, b1}, {w2, b2}};

        parameters = doubleNeuronBackpropagation.trainModel(input, expectedOutput, parameters, epochs, learningRate);


        double[][] layer_outputs = new double[3][input.length];
        double[] calculatedAnswers = doubleNeuronBackpropagation.forwardPropagate(input, parameters, layer_outputs);

        for (int i = 0; i < calculatedAnswers.length; i++) {
            assertTrue(abs(calculatedAnswers[i] - expectedOutput[i]) < 0.0001);
        }
    }
}