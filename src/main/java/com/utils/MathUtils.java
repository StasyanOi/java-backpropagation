package com.utils;

public final class MathUtils {

    private MathUtils() {}

    public static double sum(double[] numbers) {
        double sum = 0;
        for (double number : numbers) {
            sum += number;
        }
        return sum;
    }

    public static double average(double[] array) {
        return sum(array) / array.length;
    }
}
