package com.samples;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

class DoubleNeuronBackpropagationIntegrationTest {

    DoubleNeuronBackpropagation doubleNeuronBackpropagation = new DoubleNeuronBackpropagation();

    @Test
    void testBackpropagation() {
        double[] input = {-3, -2, -1, 0, 1, 2, 3};
        double[] expected_output = {-0.5, 0, 0.5, 1, 1.5, 2, 2.5};
        double learning_rate = 0.001;
        int epochs = 10000;

        double w1 = Math.random() + 1;
        double w2 = Math.random() + 1;
        double b1 = Math.random() + 1;
        double b2 = Math.random() + 1;

        double[][] parameters = {{w1, b1}, {w2, b2}};

        parameters = doubleNeuronBackpropagation.train_model(input, expected_output, parameters, epochs, learning_rate);


        double[][] layer_outputs = new double[3][input.length];
        double[] calculated_answers = doubleNeuronBackpropagation.forward_propagate(input, parameters, layer_outputs);

        for (int i = 0; i < calculated_answers.length; i++) {
            assertTrue(Math.abs(calculated_answers[i] - expected_output[i]) < 0.0001);
        }
    }
}