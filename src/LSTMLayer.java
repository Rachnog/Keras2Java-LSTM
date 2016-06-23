import org.jblas.DoubleMatrix;

import java.io.IOException;
import java.util.ArrayList;

/**
 * Created by Alex on 23.06.2016.
 * Basic LSTM layer with forward propagation routine
 */
public class LSTMLayer implements AbstractLayer {

    private DoubleMatrix W_i;
    private DoubleMatrix U_i;
    private DoubleMatrix b_i;
    private DoubleMatrix W_c;
    private DoubleMatrix U_c;
    private DoubleMatrix b_c;
    private DoubleMatrix W_f;
    private DoubleMatrix U_f;
    private DoubleMatrix b_f;
    private DoubleMatrix W_o;
    private DoubleMatrix U_o;
    private DoubleMatrix b_o;

    private int realSize;
    private int layerNum;
    private boolean returnSequence;

    public LSTMLayer(String path, int layerNum, boolean returnSequence) throws IOException {

        this.layerNum = layerNum;
        this.returnSequence = returnSequence;

        // Just loading initial weights trained by Keras
        this.W_i = Utils.loadMatrixFromFile(path + this.layerNum + "_param_0.txt");
        this.U_i = Utils.loadMatrixFromFile(path + this.layerNum + "_param_1.txt");
        this.b_i = Utils.loadMatrixFromFile(path + this.layerNum + "_param_2.txt");
        this.W_c = Utils.loadMatrixFromFile(path + this.layerNum + "_param_3.txt");
        this.U_c = Utils.loadMatrixFromFile(path + this.layerNum + "_param_4.txt");
        this.b_c = Utils.loadMatrixFromFile(path + this.layerNum + "_param_5.txt");
        this.W_f = Utils.loadMatrixFromFile(path + this.layerNum + "_param_6.txt");
        this.U_f = Utils.loadMatrixFromFile(path + this.layerNum + "_param_7.txt");
        this.b_f = Utils.loadMatrixFromFile(path + this.layerNum + "_param_8.txt");
        this.W_o = Utils.loadMatrixFromFile(path + this.layerNum + "_param_9.txt");
        this.U_o = Utils.loadMatrixFromFile(path + this.layerNum + "_param_10.txt");
        this.b_o = Utils.loadMatrixFromFile(path + this.layerNum + "_param_11.txt");
    }

    public void setRealSize(int realSize) {
        this.realSize = realSize;
    }

    public DoubleMatrix forwardStep(DoubleMatrix X) {

        // If our input is shorter then defined by architecture (15),
        // let's add some zero vectors to it to allow adequate matrix multiplication
        if (this.layerNum == 0) {
            X = inputFix(X);
        }

        ArrayList<DoubleMatrix> outputs = new ArrayList();

        // Let's define previous cell output and hidden state
        DoubleMatrix h_t_1 = DoubleMatrix.zeros(this.W_i.columns, 1);
        DoubleMatrix C_t_1 = DoubleMatrix.zeros(this.W_i.columns, 1);

        for (int i = 0; i < X.columns; i++) {

            // Weights update for every cell step-by-step.
            // For more details check out: http://deeplearning.net/tutorial/lstm.html
            DoubleMatrix x_t = X.getColumn(i);
            DoubleMatrix W_i_mul_x = this.W_i.transpose().mmul(x_t);
            DoubleMatrix U_i_mul_h_1 = this.U_i.transpose().mmul(h_t_1);
            DoubleMatrix i_t = ActivationFunction.hardSigmoid(W_i_mul_x.addColumnVector(U_i_mul_h_1).addColumnVector(this.b_i));

            DoubleMatrix W_c_mul_x = this.W_c.transpose().mmul(x_t);
            DoubleMatrix U_c_mul_h_1 = this.U_c.transpose().mmul(h_t_1);
            DoubleMatrix C_tilda = ActivationFunction.tanh(W_c_mul_x.addColumnVector(U_c_mul_h_1).addColumnVector(this.b_c));

            DoubleMatrix W_f_mul_x = this.W_f.transpose().mmul(x_t);
            DoubleMatrix U_f_mul_h_1 = this.U_f.transpose().mmul(h_t_1);
            DoubleMatrix f_i = ActivationFunction.hardSigmoid(W_f_mul_x.addColumnVector(U_f_mul_h_1).addColumnVector(this.b_f));

            DoubleMatrix C_t = (i_t.mul(C_tilda)).add(f_i.mul(C_t_1));

            DoubleMatrix W_o_mul_x = this.W_o.transpose().mmul(x_t);
            DoubleMatrix U_o_mul_h_1 = this.U_o.transpose().mmul(h_t_1);

            DoubleMatrix o_t = ActivationFunction.hardSigmoid(W_o_mul_x.addColumnVector(U_o_mul_h_1).addColumnVector(this.b_o));
            DoubleMatrix h_t = o_t.mul(ActivationFunction.tanh(C_t));

            outputs.add(h_t);
            h_t_1 = h_t;
            C_t_1 = C_t;

        }

        if (this.returnSequence) {

            // We return out sequence corresponding to our input,
            // which has length of this.realSize.
            // We will restore it in next layer again using fixInput()
            int rows = outputs.get(0).rows;
            DoubleMatrix result = DoubleMatrix.zeros(rows, this.realSize);
            for (int i = 0; i < outputs.size(); i++) {
                for (int j = 0; j < this.realSize; j++) {
                    result.put(i, j, outputs.get(j).get(i));
                }
            }
            return result;

        } else {
            // If we don't want to return sequence of outputs from every cell,
            // but only for the last one (for the last LSTM layer), use this.
            return outputs.get(outputs.size() - 1);
        }

    }

    public DoubleMatrix inputFix(DoubleMatrix X) {
        DoubleMatrix res = DoubleMatrix.zeros(W_i.rows, W_i.columns);
        for (int i = 0; i < X.rows; i++) {
            res.putColumn(i, X.getRow(i));
        }
        return res;
    }


}
