package com.algorithms;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import static com.utils.MathUtils.random;

public class Perceptron {

        private final RealMatrix[] layer_outputs;
        private final RealMatrix[] weights;
        private final RealMatrix[] biases;
        private final RealMatrix[] deltas;
        private final int[] architecture;

        public Perceptron(int input_size, int[] layers) {
            deltas = new RealMatrix[layers.length];
            layer_outputs = new RealMatrix[layers.length];
            weights = new RealMatrix[layers.length];

            int layer_input = input_size;
            for (int i = 0; i < weights.length; i++) {
                weights[i] = random(layers[i], layer_input);
                layer_input = layers[i];
            }

            biases = new RealMatrix[layers.length];

            for (int i = 0; i < biases.length; i++) {
                biases[i] = random(layers[i], 1);
            }

            architecture = layers;
        }

        public void trainOnBatchSgd(RealMatrix train_inputs, RealMatrix train_outputs) {
            for (int i = 0; i < train_inputs.getColumnDimension(); i++) {
                train_on_vector(train_inputs.getColumnMatrix(i), train_outputs.getColumnMatrix(i));
            }
        }

        public void trainOnBatchBsgd(RealMatrix train_inputs, RealMatrix train_outputs) {
            RealMatrix[] grads_w = new RealMatrix[architecture.length];
            RealMatrix[] grads_b = new RealMatrix[architecture.length];
            double learning_rate = 0.01;
            int elements = train_inputs.getColumnDimension();
            for (int m = 0; m < elements; m++) {
                RealMatrix train_input = train_inputs.getColumnMatrix(m);
                RealMatrix train_output = train_outputs.getColumnMatrix(m);
                RealMatrix calculated_output = feedForwardVector(train_input);

                System.out.println("mse: " + mse(train_output, calculated_output));


                for (int i = architecture.length - 1; i >= 0; i--) {
                    if (i == architecture.length - 1) {
                        deltas[i] = train_output.subtract(calculated_output).scalarMultiply(-2);
                    } else {
                        deltas[i] = weights[i + 1].transpose().multiply(deltas[i + 1]);
                    }
                }

                for (int i = 0; i < deltas.length; i++) {
                    if (grads_w[i] == null) {
                        if (i == 0) {
                            grads_w[i] = deltas[i].multiply(train_input.transpose());
                        } else {
                            grads_w[i] = deltas[i].multiply(layer_outputs[i - 1].transpose());
                        }
                    } else {
                        if (i == 0) {
                            grads_w[i] = grads_w[i].add(deltas[i].multiply(train_input.transpose()));
                        } else {
                            grads_w[i] = grads_w[i].add(deltas[i].multiply(layer_outputs[i - 1].transpose()));
                        }
                    }
                }

                for (int i = 0; i < deltas.length; i++) {
                    if (grads_b[i] == null) {
                        grads_b[i] = deltas[i];
                    } else {
                        grads_b[i] = grads_b[i].add(deltas[i]);
                    }
                }


            }

            for (int i = 0; i < weights.length; i++) {
                weights[i] = weights[i].subtract(grads_w[i].scalarMultiply(learning_rate / elements));
            }

            for (int i = 0; i < biases.length; i++) {
                biases[i] = biases[i].subtract(grads_b[i].scalarMultiply(learning_rate / elements));
            }
        }

        public void train_on_vector(RealMatrix train_input, RealMatrix train_output) {
            RealMatrix calculated_output = feedForwardVector(train_input);
            System.out.println("mse: " + mse(train_output, calculated_output));
            double learning_rate = 0.01;

            for (int i = architecture.length - 1; i >= 0; i--) {
                if (i == architecture.length - 1) {
                    deltas[i] = train_output.subtract(calculated_output).scalarMultiply(-2);
                } else {
                    deltas[i] = weights[i + 1].transpose().multiply(deltas[i + 1]);
                }
            }

            RealMatrix[] grads_w = new RealMatrix[architecture.length];
            for (int i = 0; i < deltas.length; i++) {
                if (i == 0) {
                    grads_w[i] = deltas[i].multiply(train_input.transpose());
                } else {
                    grads_w[i] = deltas[i].multiply(layer_outputs[i - 1].transpose());
                }
            }

            RealMatrix[] grads_b = new RealMatrix[architecture.length];
            for (int i = 0; i < deltas.length; i++) {
                grads_b[i] = deltas[i];
            }

            for (int i = 0; i < weights.length; i++) {
                weights[i] = weights[i].subtract(grads_w[i].scalarMultiply(learning_rate));
            }

            for (int i = 0; i < biases.length; i++) {
                biases[i] = biases[i].subtract(grads_b[i].scalarMultiply(learning_rate));
            }
        }


        public RealMatrix feedForwardVector(RealMatrix input_vector) {
            RealMatrix output_vector = MatrixUtils.createRealMatrix(architecture[architecture.length - 1], 1);
            RealMatrix network_output = input_vector;
            for (int i = 0; i < architecture.length; i++) {
                network_output = network_output.preMultiply(weights[i]).add(biases[i]);
                layer_outputs[i] = network_output;
            }
            output_vector.setColumnMatrix(0, network_output);
            return output_vector;
        }

        public double mse(RealMatrix real, RealMatrix calc) {
            RealMatrix subtracted = real.subtract(calc);
            int rowDimension = subtracted.getRowDimension();
            int columnDimension = subtracted.getColumnDimension();
            for (int i = 0; i < rowDimension; i++) {
                for (int j = 0; j < columnDimension; j++) {
                    subtracted.setEntry(i, j, Math.pow(subtracted.getEntry(i, j), 2));
                }
            }

            double total = 0;
            for (int i = 0; i < rowDimension; i++) {
                for (int j = 0; j < columnDimension; j++) {
                    total += subtracted.getEntry(i, j);
                }
            }
            return total / (rowDimension * columnDimension);
        }
    }