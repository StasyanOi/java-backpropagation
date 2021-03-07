import java.util.Arrays;

public class DoubleNeuronBackpropagation {

    private static double[] line(double w, double[] x, double b) {
        double[] y = new double[x.length];
        for (int i = 0; i < y.length; i++) {
            y[i] = w * x[i] + b;
        }
        return y;
    }

    private static double mse(double[] y_real, double[] y_calc) {
        double error = 0;
        int n = y_real.length;
        for (int i = 0; i < n; i++) {
            error += Math.pow(y_real[i] - y_calc[i], 2);
        }
        return error / n;
    }

    private static double update_weight(double initial, double partial_derivative, double learning_rate) {
        return initial - partial_derivative * learning_rate;
    }

    private static double[] partial_w(double[] input, double[] y_real, double[] y_calc) {
        double[] partial_w = new double[input.length];
        for (int i = 0; i < partial_w.length; i++) {
            partial_w[i] = (-2 * y_real[i] + 2 * y_calc[i]) * input[i];
        }
        return partial_w;
    }

    private static double[] partial_b(double[] y_real, double[] y_calc) {
        double[] partial_b = new double[y_real.length];
        for (int i = 0; i < partial_b.length; i++) {
            partial_b[i] = (-2 * y_real[i] + 2 * y_calc[i]);
        }
        return partial_b;
    }

    private static double aggregate(double[] array) {
        double sum = 0;
        for (int i = 0; i < array.length; i++) {
            sum += array[i];
        }
        return sum;
    }

    private static double average(double[] array) {
        return aggregate(array) / array.length;
    }

    private static double[] sum(double[] array1, double[] array2) {
        double[] array3 = new double[array1.length];
        for (int i = 0; i < array1.length; i++) {
            array3[i] = array1[i] + array2[i];
        }
        return array3;
    }

    private static double[] minus(double[] array1, double[] array2) {
        double[] array3 = new double[array1.length];
        for (int i = 0; i < array1.length; i++) {
            array3[i] = array1[i] - array2[i];
        }
        return array3;
    }

    private static double[] divide(double[] array1, double[] array2) {
        double[] array3 = new double[array1.length];
        for (int i = 0; i < array1.length; i++) {
            array3[i] = array1[i] / array2[i];
        }
        return array3;
    }

    private static double[] delta_initial(double[] y_real, double[] y_calc) {
        double[] delta = new double[y_real.length];
        for (int i = 0; i < delta.length; i++) {
            delta[i] = (-2 * y_real[i] + 2 * y_calc[i]);
        }
        return delta;
    }

    private static double[] multiply(double[] array1, double[] array2) {
        double[] array3 = new double[array1.length];
        for (int i = 0; i < array1.length; i++) {
            array3[i] = array1[i] * array2[i];
        }
        return array3;
    }

    private static double[] multiply(double[] array1, double num) {
        double[] array3 = new double[array1.length];
        for (int i = 0; i < array1.length; i++) {
            array3[i] = array1[i] * num;
        }
        return array3;
    }


    private static void print(double[][] array) {
        for (int i = 0; i < array.length; i++) {
            System.out.println(Arrays.toString(array[i]));
        }
    }

    public static void main(String[] args) {
        double[] input = {-3, -2, -1, 0, 1, 2, 3};
        double[] expected_output = {-7, -5, -3, -1, 1, 3, 5};
        double learning_rate = 0.01;
        int epochs = 100;
        train_model(input, expected_output, epochs, learning_rate);
    }

    private static void train_model(double[] input, double[] expected_output, int epochs, double learning_rate) {
        double w1 = Math.random();
        double w2 = Math.random();
        double b1 = Math.random();
        double b2 = Math.random();

        double[][] parameters = {{w1, b1}, {w2, b2}};
        double[][] a = new double[3][7];
        a[0] = input;

        for (int i = 0; i < epochs; i++) {

            double[] calculated_output = input;
            for (int j = 0; j < parameters.length; j++) {
                double[] output = line(parameters[j][0], calculated_output, parameters[j][1]);
                a[j + 1] = output;
                calculated_output = output;
            }

            double mse = mse(expected_output, calculated_output);

            System.out.println("mse: " + mse);

            //update w2
            double[] delta_w2 = delta_initial(expected_output, calculated_output);
            double[] partial_derivative_w2 = multiply(delta_w2, a[1]);
            parameters[1][0] = update_weight(parameters[1][0], average(partial_derivative_w2), learning_rate);

            //update b2
            double[] delta_b2 = delta_initial(expected_output, calculated_output);
            double[] partial_derivative_b2 = multiply(delta_b2, 1);
            parameters[1][1] = update_weight(parameters[1][1], average(partial_derivative_b2), learning_rate);

            //update w1
            double[] delta_w1 = multiply(delta_w2, parameters[1][0]);
            double[] partial_derivative_w1 = multiply(delta_w1, a[0]);
            parameters[0][0] = update_weight(parameters[0][0], average(partial_derivative_w1), learning_rate);

            //update b1
            double[] delta_b1 = multiply(delta_b2, parameters[1][0]);
            double[] partial_derivative_b1 = multiply(delta_b1, 1);
            parameters[0][1] = update_weight(parameters[0][1], average(partial_derivative_b1), learning_rate);


        }

    }
}
