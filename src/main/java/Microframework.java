import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

public class Microframework {


    private static class NN {

        private final RealMatrix[] layer_outputs;
        private final RealMatrix[] weights;
        private final RealMatrix[] biases;
        private final RealMatrix[] deltas;
        private final int[] architecture;

        public NN(int input_size, int[] layers) {
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

        public void train_on_batch_new(RealMatrix train_inputs, RealMatrix train_outputs) {
            for (int i = 0; i < train_inputs.getColumnDimension(); i++) {
                train_new(train_inputs.getColumnMatrix(i), train_outputs.getColumnMatrix(i));
            }
        }

        public void train_new(RealMatrix train_input, RealMatrix train_output) {
            RealMatrix calculated_output = feed_forward_vector_new(train_input);
            System.out.println("mse: " + mse_new(train_output, calculated_output));
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

            for (int i = 0; i < grads_w.length; i++) {
                grads_w[i] = grads_w[i].scalarMultiply(learning_rate);
            }

            RealMatrix[] grads_b = new RealMatrix[architecture.length];
            for (int i = 0; i < deltas.length; i++) {
                grads_b[i] = deltas[i];
            }

            for (int i = 0; i < grads_b.length; i++) {
                grads_b[i] = grads_b[i].scalarMultiply(learning_rate);
            }

            for (int i = 0; i < weights.length; i++) {
                weights[i] = weights[i].subtract(grads_w[i]);
            }

            for (int i = 0; i < biases.length; i++) {
                biases[i] = biases[i].subtract(grads_b[i]);
            }

        }


        public RealMatrix feed_forward_vector_new(RealMatrix input_vector) {
            RealMatrix output_vector = MatrixUtils.createRealMatrix(architecture[architecture.length - 1], 1);
            RealMatrix network_output = input_vector;
            for (int i = 0; i < architecture.length; i++) {
                network_output = network_output.preMultiply(weights[i]).add(biases[i]);
                layer_outputs[i] = network_output;
            }
            output_vector.setColumnMatrix(0, network_output);
            return output_vector;
        }

        public double mse_new(RealMatrix real, RealMatrix calc) {
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


    public static RealMatrix random(int numRows, int numCols) {
        RealMatrix matrix64F = MatrixUtils.createRealMatrix(numRows, numCols);
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                matrix64F.setEntry(i, j, Math.random());
            }
        }
        return matrix64F;
    }


    public static void main(String[] args) {
        NN nn = new NN(2, new int[]{2, 3, 2, 2});

        RealMatrix andInput = getAndInputNew();
        RealMatrix andOutput = getAndOutputNew();

        for (int i = 0; i < 1000; i++) {
            nn.train_on_batch_new(andInput, andOutput);
        }
        for (int i = 0; i < 4; i++) {
            System.out.println(nn.feed_forward_vector_new(andInput.getColumnMatrix(i)));
            System.out.println(andOutput.getColumnMatrix(i));
        }
    }

    private static RealMatrix getAndOutputNew() {
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

    private static RealMatrix getAndInputNew() {
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
