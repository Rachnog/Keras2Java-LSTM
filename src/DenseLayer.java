import org.jblas.DoubleMatrix;

import java.io.IOException;

/**
 * Created by Alex on 23.06.2016.
 */
public class DenseLayer implements AbstractLayer {

    private DoubleMatrix W_dense;
    private DoubleMatrix b_dense;

    private int realSize;

    public DenseLayer(String path) throws IOException {

        this.W_dense = Utils.loadMatrixFromFile(path + "2_param_0.txt");
        this.b_dense = Utils.loadMatrixFromFile(path + "2_param_1.txt");
    }

    public DoubleMatrix forwardStep(DoubleMatrix X) {
        return ActivationFunction.softmax(this.W_dense.transpose().mmul(X).addColumnVector(this.b_dense));
    }

    @Override
    public void setRealSize(int realSize) {

    }

}
