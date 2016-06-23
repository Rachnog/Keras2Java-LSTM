/**
 * Created by Alex on 09.06.2016.
 */

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;


public class ActivationFunction {

    public static DoubleMatrix tanh(DoubleMatrix X) {
        return MatrixFunctions.tanh(X);
    }

    public static DoubleMatrix softmax(DoubleMatrix X) {
        X = X.transpose();
        DoubleMatrix expM = MatrixFunctions.exp(X);
        for (int i = 0; i < X.rows; i++) {
            DoubleMatrix expMi = expM.getRow(i);
            expM.putRow(i, expMi.div(expMi.sum()));
        }
        return expM;
    }

    public static DoubleMatrix hardSigmoid(DoubleMatrix X) {
        double slope = 0.2;
        double shift = 0.5;
        X = X.mul(slope).add(shift);
        DoubleMatrix clippedX = new DoubleMatrix(X.rows, X.columns);
        for (int i = 0; i < X.length; i++) {
            if (X.get(i) > 1) {
                clippedX.put(i, 1);
            } else if (X.get(i) < 0) {
                clippedX.put(i, 0);
            } else {
                clippedX.put(i, X.get(i));
            }
        }
        return clippedX;
    }
}

