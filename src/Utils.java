import org.jblas.DoubleMatrix;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

/**
 * Created by Alex on 23.06.2016.
 */
public class Utils {

    public static DoubleMatrix loadMatrixFromFile(String filePath) throws IOException {
        FileInputStream fstream = new FileInputStream(filePath);
        BufferedReader br = new BufferedReader(new InputStreamReader(fstream));
        ArrayList<ArrayList> loadedArrays = new ArrayList();

        String strLine;
        while ((strLine = br.readLine()) != null) {
            ArrayList<Double> row = new ArrayList();
            String[] a = strLine.split(" ");
            for (String s : a) {
                row.add(Double.parseDouble(s));
            }
            loadedArrays.add(row);
        }
        br.close();

        int columns = loadedArrays.get(0).size();
        int rows = loadedArrays.size();
        double[][] target = new double[rows][columns];

        for (int i = 0; i < loadedArrays.size(); i++) {
            for (int j = 0; j < target[i].length; j++) {
                target[i][j] = (Double) loadedArrays.get(i).get(j);
            }
        }
        return new DoubleMatrix(target);
    }


}
