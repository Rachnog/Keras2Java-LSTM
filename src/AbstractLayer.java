import org.jblas.DoubleMatrix;

/**
 * Created by Alex on 23.06.2016.
 */
public interface AbstractLayer {

    public DoubleMatrix forwardStep(DoubleMatrix X);

    public void setRealSize(int realSize);



}
