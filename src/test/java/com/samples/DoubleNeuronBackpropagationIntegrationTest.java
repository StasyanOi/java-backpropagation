package com.samples;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class DoubleNeuronBackpropagationIntegrationTest {

    DoubleNeuronBackpropagation doubleNeuronBackpropagation = new DoubleNeuronBackpropagation();

    @Test
    void testBackpropagation() {
        double[] input = {-3, -2, -1, 0, 1, 2, 3};
        double[] expected_output = {-7, -5, -3, -1, 1, 3, 5};
        double learning_rate = 0.01;
        int epochs = 1000;
        double[][] weights = doubleNeuronBackpropagation.train_model(input, expected_output, epochs, learning_rate);

        double w1 = weights[0][0];
        Assertions.assertEquals(w1, 0);
        double b1 = weights[0][1];
        Assertions.assertEquals(b1, 0);
        double w2 = weights[1][0];
        Assertions.assertEquals(w2, 0);
        double b2 = weights[1][1];
        Assertions.assertEquals(b2, 0);
    }
}