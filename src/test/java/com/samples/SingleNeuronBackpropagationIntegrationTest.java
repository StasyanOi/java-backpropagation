package com.samples;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class SingleNeuronBackpropagationIntegrationTest {

    SingleNeuronBackpropagation singleNeuronBackpropagation = new SingleNeuronBackpropagation();

    @Test
    void testBackpropagation() {
        double[] input = {-3, -2, -1, 0, 1, 2, 3};
        double[] expected_output = {-7, -5, -3, -1, 1, 3, 5};
        double learning_rate = 0.01;
        int epochs = 1000;
        double[] final_coefficients = singleNeuronBackpropagation.train_linear_model(input, expected_output, epochs, learning_rate);
        double a = final_coefficients[0];
        double b = final_coefficients[1];
        double[] calculated_output = singleNeuronBackpropagation.forward_propagate(a, input, b);

        for (int i = 0; i < calculated_output.length; i++) {
            calculated_output[i] = Math.round(calculated_output[i]);
            Assertions.assertEquals(expected_output[i], calculated_output[i]);
        }
    }
}