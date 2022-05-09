package com.utils;


public final class ErrorsUtils {

    private ErrorsUtils() {}

    public static double mse(double[] y_real, double[] y_calc) {
        double error = 0;
        int n = y_real.length;
        for (int i = 0; i < n; i++) {
            error += Math.pow(y_real[i] - y_calc[i], 2);
        }
        return error / n;
    }
}
