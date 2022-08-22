package com.algorithms;

import static com.utils.ErrorsUtils.mse;
import static com.utils.MathUtils.average;

/**
 * This is a sample with plain backpropagation implementation in a line function (a*x + b = y).
 * <p>
 * In backpropagation we try to find the coefficient with which to nudge the "weights" (in this case a and b)
 * in order to minimize the error function (mean squared error in this case).
 */
public class SingleNeuronBackpropagation {

    /**
     * Function calculates an array of outputs based on an array of inputs using a line function (a*x + b = y)
     *
     * @param a - angle coefficient
     * @param x - input
     * @param b - up-down coefficient
     * @return an array of outputs calculated using the line function
     */
    public double[] forwardPropagate(double a, double[] x, double b) {
        double[] y = new double[x.length];
        for (int i = 0; i < y.length; i++) {
            y[i] = a * x[i] + b;
        }
        return y;
    }

    private double update_weight(double initial_coefficient, double coefficient_partial_derivative, double learning_rate) {
        return initial_coefficient - coefficient_partial_derivative * learning_rate;
    }

    private double[] partial_w(double[] input, double[] y_real, double[] y_calc) {
        double[] partial_w = new double[input.length];
        for (int i = 0; i < partial_w.length; i++) {
            // this is the partial derivative of w from mse difference
            partial_w[i] = (-2 * y_real[i] + 2 * y_calc[i]) * input[i];
        }
        return partial_w;
    }

    private double[] partial_b(double[] y_real, double[] y_calc) {
        double[] partial_b = new double[y_real.length];
        for (int i = 0; i < partial_b.length; i++) {
            // this is the partial derivative of b from mse difference
            partial_b[i] = (-2 * y_real[i] + 2 * y_calc[i]);
        }
        return partial_b;
    }

    public double[] trainLinearModel(double[] input, double[] expected_output, int epochs, double learning_rate) {
        double w = Math.random();
        double b = Math.random();

        for (int i = 0; i < epochs; i++) {
            double[] calculated_output = forwardPropagate(w, input, b);
            double mse = mse(expected_output, calculated_output);
            System.out.println("mse: " + mse);
            System.out.println("w: " + w);
            System.out.println("b: " + b);
            double partial_derivative_w = average(partial_w(input, expected_output, calculated_output));
            double partial_derivative_b = average(partial_b(expected_output, calculated_output));
            w = update_weight(w, partial_derivative_w, learning_rate);
            b = update_weight(b, partial_derivative_b, learning_rate);
        }
        return new double[]{w, b};
    }
}
