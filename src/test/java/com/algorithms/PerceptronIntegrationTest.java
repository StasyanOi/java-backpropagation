package com.algorithms;

import org.apache.commons.math3.linear.RealMatrix;
import org.junit.jupiter.api.Test;

import static com.utils.DataUtils.getAndInputNew;
import static com.utils.DataUtils.getAndOutputNew;
import static org.junit.jupiter.api.Assertions.assertEquals;

class PerceptronIntegrationTest {

    @Test
    void testBackpropagation() {
        Perceptron perceptron = new Perceptron(2, new int[]{2, 3, 3, 3, 2});

        RealMatrix andInput = getAndInputNew();
        RealMatrix andOutput = getAndOutputNew();

        for (int i = 0; i < 1000; i++) {
            perceptron.trainOnBatchBsgd(andInput, andOutput);
        }
        for (int i = 0; i < 4; i++) {
            RealMatrix calculatedResult = perceptron.feedForwardVector(andInput.getColumnMatrix(i));
            RealMatrix expectedResult = andOutput.getColumnMatrix(i);

            assertEquals(Math.round(calculatedResult.getEntry(0, 0)), expectedResult.getEntry(0, 0));
            assertEquals(Math.round(calculatedResult.getEntry(1, 0)), expectedResult.getEntry(1, 0));
        }
    }
}