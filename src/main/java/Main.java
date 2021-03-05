public class Main {

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

    private static double sum(double[] array) {
        double sum = 0;
        for (int i = 0; i < array.length; i++) {
            sum += array[i];
        }
        return sum;
    }

    private static double average(double[] array) {
        return sum(array) / array.length;
    }

    private static double[] train_linear_model(double[] input, double[] expected_output, int epochs, double learning_rate) {
        double w = Math.random();
        double b = Math.random();

        for (int i = 0; i < epochs; i++) {
            double[] calculated_output = line(w, input, b);
            double mse = mse(expected_output, calculated_output);
            System.out.println("mse: " + mse);
            System.out.println("w: " + w);
            System.out.println("b: " + b);
            double partial_derivative_w = average(partial_w(input, expected_output, calculated_output));
            double partial_derivative_b = average(partial_b(expected_output, calculated_output));
            System.out.println("partial_derivative_b: " + partial_derivative_b);
            System.out.println("partial_derivative_w: " + partial_derivative_w);
            w = update_weight(w, partial_derivative_w, learning_rate);
            b = update_weight(b, partial_derivative_b, learning_rate);
        }

        return new double[]{w, b};
    }

    public static void main(String[] args) {
        double[] input = {-3, -2, -1, 0, 1, 2, 3};
        double[] expected_output = {-7, -5, -3, -1, 1, 3, 5};
        double learning_rate = 0.01;
        int epochs = 20;
        train_linear_model(input, expected_output, epochs, learning_rate);
    }
}
