package com.shawn.pmmltoser;

import org.dmg.pmml.FieldName;
import org.dmg.pmml.PMML;
import org.dmg.pmml.neural_network.NeuralNetwork;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.evaluator.*;
import org.jpmml.evaluator.neural_network.NeuralNetworkEvaluator;
import org.jpmml.evaluator.tree.TreeModelEvaluator;
import org.xml.sax.SAXException;

import javax.xml.bind.JAXBException;
import java.io.*;
import java.util.*;

/**
 * Created by ShawnG on 2017/3/20.
 */
public class PmmlNewTest {
    public static double[] normalize(double[][] values){
        double[] mean = new double[9];
        double[] std = new double[9];
        double[][] result = new double[values.length][9];
        double[] res_data = new double[18];

        for (int column = 0; column < 9; column++){
            double sum = 0;
            for (int row = 0; row < values.length; row++){
                sum = sum + values[row][column];
            }
            mean[column] = sum / values.length;
            double sum_1 = 0;
            for (int row_1 = 0; row_1 < values.length; row_1++){
                sum_1 = sum_1 + (values[row_1][column] - mean[column])*(values[row_1][column] - mean[column]);
            }
            std[column] = Math.sqrt((sum_1/values.length));
        }

        for (int i = 0; i < values.length; i++) {
            for (int j = 0; j < 9; j++) {
                result[i][j] = (values[i][j] - mean[j])/std[j];
            }
        }

        int index = 0;
        for (int i = 0; i < 9; i++) {
            double max = result[i][0];
            double min = result[i][0];
            for (int j = 0; j < values.length; j++) {
                if (values[j][i]>max){
                    max = values[j][i];
                }
                if (values[j][i]<min){
                    min = values[j][i];
                }
            }
            res_data[index] = max;
            index++;
            res_data[index] = min;
            index++;
        }
        return res_data;
    }

    public static double[][] readList(){
        BufferedReader reader = null;
        double[][] values = new double[40][12];
        double[][] result = new double[40][9];
        int i = 0;
        try {
            reader = new BufferedReader(new FileReader("F:\\PythonEXP\\SensorDataSetAnalysis\\data_set\\raw_data.csv"));
            reader.readLine();
            String line = null;
            while (i<40 && (line=reader.readLine())!=null){
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
        for (int j = 0; j < values.length; j++) {
            result[j][0] = values[j][1];
            result[j][1] = values[j][2];
            result[j][2] = values[j][3];
            result[j][3] = values[j][4];
            result[j][4] = values[j][5];
            result[j][5] = values[j][6];
            result[j][6] = values[j][9];
            result[j][7] = values[j][10];
            result[j][8] = values[j][11];
        }
        return result;
    }

    public static Map<String, Double> analysisData(Map<String, Double> data, double[] list) {
        data.put("AX_max", list[0]);
        data.put("AX_min", list[1]);
        data.put("AY_max", list[2]);
        data.put("AY_min", list[3]);
        data.put("AZ_max", list[4]);
        data.put("AZ_min", list[5]);

        data.put("GX_max", list[6]);
        data.put("GX_min", list[7]);
        data.put("GY_max", list[8]);
        data.put("GY_min", list[9]);
        data.put("GZ_max", list[10]);
        data.put("GZ_min", list[11]);

        data.put("LX_max", list[12]);
        data.put("LX_min", list[13]);
        data.put("LY_max", list[14]);
        data.put("LY_min", list[15]);
        data.put("LZ_max", list[16]);
        data.put("LZ_min", list[17]);
        return data;
    }

    public static String readModel(double[] list, PMML pmml){
        ModelEvaluatorFactory modelEvaluatorFactory = ModelEvaluatorFactory.newInstance();
        ModelEvaluator<?> modelEvaluator = modelEvaluatorFactory.newModelEvaluator(pmml);
        Evaluator evaluator = modelEvaluator;
        Map<String, Double> data = new HashMap<>();
        analysisData(data, list);
        Map<FieldName, FieldValue> arguments = new LinkedHashMap<>();
        List<InputField> inputFields = evaluator.getInputFields();
        FieldValue inputFieldValue = null;
        for (InputField inputField : inputFields) {
            FieldName inputFieldName = inputField.getName();
            Object str = null;
            if (inputFieldName.toString().equals("AX_max")) {
                str = data.get("AX_max");
            } else if (inputFieldName.toString().equals("AX_min")) {
                str = data.get("AX_min");
            } else if (inputFieldName.toString().equals("AY_max")) {
                str = data.get("AY_max");
            } else if (inputFieldName.toString().equals("AY_min")) {
                str = data.get("AY_min");
            } else if (inputFieldName.toString().equals("AZ_max")) {
                str = data.get("AZ_max");
            } else if (inputFieldName.toString().equals("AZ_min")) {
                str = data.get("AZ_min");
            } else if (inputFieldName.toString().equals("GX_max")) {
                str = data.get("GX_max");
            } else if (inputFieldName.toString().equals("GX_min")) {
                str = data.get("GX_min");
            } else if (inputFieldName.toString().equals("GY_max")) {
                str = data.get("GY_max");
            } else if (inputFieldName.toString().equals("GY_min")) {
                str = data.get("GY_min");
            } else if (inputFieldName.toString().equals("GZ_max")) {
                str = data.get("GZ_max");
            } else if (inputFieldName.toString().equals("GZ_min")) {
                str = data.get("GZ_min");
            } else if (inputFieldName.toString().equals("LX_max")) {
                str = data.get("LX_max");
            } else if (inputFieldName.toString().equals("LX_min")) {
                str = data.get("LX_min");
            } else if (inputFieldName.toString().equals("LY_max")) {
                str = data.get("LY_max");
            } else if (inputFieldName.toString().equals("LY_min")) {
                str = data.get("LY_min");
            } else if (inputFieldName.toString().equals("LZ_max")) {
                str = data.get("LZ_max");
            } else if (inputFieldName.toString().equals("LZ_min")) {
                str = data.get("LZ_min");
            }
//            System.out.println(str);
            try {
                inputFieldValue = inputField.prepare(str);
            }catch (Exception e){
            }
            arguments.put(inputFieldName, inputFieldValue);
        }
        Map<FieldName, ?> results = evaluator.evaluate(arguments);
        List<TargetField> targetFields = evaluator.getTargetFields();
        Object targetFieldValue = null;
        for (TargetField targetField : targetFields) {
            FieldName targetFieldName = targetField.getName();
            targetFieldValue = results.get(targetFieldName);
        }
        Object unboxedTargetFieldValue = null;
        if (targetFieldValue instanceof Computable) {
            Computable computable = (Computable) targetFieldValue;

            unboxedTargetFieldValue = computable.getResult();
        }
        return unboxedTargetFieldValue.toString();
    }

    public static void main(String[] args) throws FileNotFoundException, JAXBException, SAXException {
        InputStream is = new FileInputStream("F:\\PythonEXP\\SensorDataSetAnalysis\\models\\MLPClassifier_new.pmml");
        PMML pmml = org.jpmml.model.PMMLUtil.unmarshal(is);
        double[][] values = readList();
        double[] list = normalize(values);
        String code = readModel(list, pmml);
        System.out.print(code);
    }
}
