package com.utils;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

public final class DataUtils {
    private DataUtils() {
    }


    public static RealMatrix getAndOutputNew() {
        RealMatrix output = MatrixUtils.createRealMatrix(2, 4);
        output.setEntry(0, 0, 0);
        output.setEntry(1, 0, 0);
        output.setEntry(0, 1, 0);
        output.setEntry(1, 1, 0);
        output.setEntry(0, 2, 0);
        output.setEntry(1, 2, 0);
        output.setEntry(0, 3, 1);
        output.setEntry(1, 3, 1);
        return output;
    }

    public static RealMatrix getAndInputNew() {
        RealMatrix input = MatrixUtils.createRealMatrix(2, 4);
        input.setEntry(0, 0, 0);
        input.setEntry(1, 0, 0);
        input.setEntry(0, 1, 0);
        input.setEntry(1, 1, 1);
        input.setEntry(0, 2, 1);
        input.setEntry(1, 2, 0);
        input.setEntry(0, 3, 1);
        input.setEntry(1, 3, 1);
        return input;
    }
}
