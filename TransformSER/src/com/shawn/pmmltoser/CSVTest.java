package com.shawn.pmmltoser;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

/**
 * Created by ShawnG on 2017/3/23.
 */
public class CSVTest {
    public static double[][] normalize(double[][] values){
        double[] mean = new double[12];
        double[] sum = new double[12];
        double[] std = new double[12];
        double[][] result = new double[values.length][12];
        for (int i = 0; i < values.length; i++) {
            for (int j = 0; j < sum.length; j++) {
                sum[j] += values[i][j];
            }
        }
        for (int i = 0; i < sum.length; i++) {
            mean[i] = sum[i] / values.length;
        }
        for (int i = 0; i < values.length; i++) {
            for (int j = 0; j < sum.length; j++) {
                std[j] += Math.pow(values[i][j] - mean[j], 2);
            }
        }
        for (int i = 0; i < std.length; i++) {
            std[i] = Math.pow(std[i]/(values.length),0.5);
        }
        for (int i = 0; i < values.length; i++) {
            for (int j = 0; j < mean.length; j++) {
                result[i][j] = (values[i][j] - mean[j])/std[j];
            }
        }
        return result;
    }

    public static double[][] readList(){
        BufferedReader reader = null;
        double[][] values = new double[500][12];
        int i = 0;
        try {
            reader = new BufferedReader(new FileReader("F:\\PythonEXP\\SensorDataSetAnalysis\\data_set\\raw_data.csv"));
            reader.readLine();
            String line = null;
            while (i<500 && (line=reader.readLine())!=null){
                String item[] = line.split(",");
                for (int j = 0; j< item.length;j++) {
                    double value = Double.parseDouble(item[j]);
                    values[i][j] = value;
                }
                i++;
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return values;
    }
    public static void main(String[] args) {
        double[][] values = readList();
        double[][] list = normalize(values);
        for (double v : list[0]) {
            System.out.println(v);
        }
    }
}
