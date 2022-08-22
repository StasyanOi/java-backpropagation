package com.utils;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

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

    public static RealMatrix random(int numRows, int numCols) {
        RealMatrix matrix64F = MatrixUtils.createRealMatrix(numRows, numCols);
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                matrix64F.setEntry(i, j, Math.random());
            }
        }
        return matrix64F;
    }
}
