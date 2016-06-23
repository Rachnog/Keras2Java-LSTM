/**
 * Created by Alex on 09.06.2016.
 */

import org.jblas.DoubleMatrix;

import java.io.*;

public class Main {

    public static void main(String[] args) throws IOException {

        double[][] x = new double[][]{{0.0,3.0,0.0,0.0,0.4849019607843267,0.14588235294117696,0.0147,0.0,0.5,0.52,2.0},
                {1.0,3.0,0.0,1.0,0.34590000000000004,0.0855,0.0147,0.0,0.0,0.467,0.0},
                {1.0,3.0,0.0,1.0,0.34590000000000004,0.0855,0.0147,0.0,0.0,0.467,0.0}};

        DoubleMatrix X = new DoubleMatrix(x);

        SimpleLSTMPropagator propagator = new SimpleLSTMPropagator("C:\\Users\\Alex\\IdeaProjects\\LSTM\\src\\situation0\\", 2);
        DoubleMatrix prediction = propagator.forward_propagate_full(X);
        System.out.println(prediction);

    }
}
