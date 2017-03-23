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
    public static double[][] normalize(double[][] values){
        double[] mean = {
                0.172987568246,-0.59553202046,4.46550770779,
                -0.149157725964,-0.62031950318,4.43278505239,
                0.32214529421,0.0247874827203,0.0327226553929
        };
        double[] std = {
                2.465133,3.433202,7.775168,
                2.830752,3.278206,7.572936,
                1.884442,1.235964,1.079798
        };
        double[][] result = new double[values.length][9];
        for (int i = 0; i < values.length; i++) {
            for (int j = 0; j < mean.length; j++) {
                result[i][j] = (values[i][j] - mean[j])/std[j];
            }
        }
        return result;
    }

    public static double[][] readList(){
        BufferedReader reader = null;
        double[][] values = new double[1000][12];
        double[][] result = new double[1000][9];
        int i = 0;
        try {
            reader = new BufferedReader(new FileReader("F:\\PythonEXP\\SensorDataSetAnalysis\\data_set\\raw_data.csv"));
            reader.readLine();
            String line = null;
            while (i<1000 && (line=reader.readLine())!=null){
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

    public static Map<String, Double> analysisData(Map<String, Double> data, double[] list, String type) {
        double x = 0;
        double y = 0;
        double z = 0;
        if (type.equals("Accelerometer")){
            x = list[0];
            y = list[1];
            z = list[2];
        }else if (type.equals("Gravity")){
            x = list[3];
            y = list[4];
            z = list[5];
        }else if (type.equals("Linear")){
            x = list[6];
            y = list[7];
            z = list[8];
        }
        String strX = type + "X";
        String strY = type + "Y";
        String strZ = type + "Z";
        data.put(strX, x);
        data.put(strY, y);
        data.put(strZ, z);
        return data;
    }

    public static int readModel(double[] list, PMML pmml){
        ModelEvaluatorFactory modelEvaluatorFactory = ModelEvaluatorFactory.newInstance();
        ModelEvaluator<?> modelEvaluator = modelEvaluatorFactory.newModelEvaluator(pmml);
        Evaluator evaluator = modelEvaluator;
        Map<String, Double> data = new HashMap<>();
        analysisData(data, list,"Accelerometer");
        analysisData(data, list,"Gravity");
        analysisData(data, list,"Linear");
        Map<FieldName, FieldValue> arguments = new LinkedHashMap<>();
        List<InputField> inputFields = evaluator.getInputFields();
        FieldValue inputFieldValue = null;
        for (InputField inputField : inputFields) {
            FieldName inputFieldName = inputField.getName();
            Object str = null;
            if (inputFieldName.toString().equals("AccelerometerX")) {
                str = data.get("AccelerometerX");
            } else if (inputFieldName.toString().equals("AccelerometerY")) {
                str = data.get("AccelerometerY");
            } else if (inputFieldName.toString().equals("AccelerometerZ")) {
                str = data.get("AccelerometerZ");
            } else if (inputFieldName.toString().equals("GravityX")) {
                str = data.get("GravityX");
            } else if (inputFieldName.toString().equals("GravityY")) {
                str = data.get("GravityY");
            } else if (inputFieldName.toString().equals("GravityZ")) {
                str = data.get("GravityZ");
            } else if (inputFieldName.toString().equals("LinearX")) {
                str = data.get("LinearX");
            } else if (inputFieldName.toString().equals("LinearY")) {
                str = data.get("LinearY");
            } else if (inputFieldName.toString().equals("LinearZ")) {
                str = data.get("LinearZ");
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
        int code = 0;
        switch (unboxedTargetFieldValue.toString()) {
            case "2":
                code = 2;
                break;
            case "1":
                code = 1;
                break;
            case "0":
                code = 0;
                break;
        }
        return code;
    }

    public static void main(String[] args) throws FileNotFoundException, JAXBException, SAXException {
        InputStream is = new FileInputStream("F:\\PythonEXP\\SensorDataSetAnalysis\\models\\MLPClassifier_new.pmml");
        PMML pmml = org.jpmml.model.PMMLUtil.unmarshal(is);
        double[][] values = readList();
        double[][] list = normalize(values);
        int code;
        int countF = 0;
        int countS = 0;
        int countV = 0;
//        readModel(list[0],pmml);
        for (double[] v : list) {
            code = readModel(v,pmml);
            switch (code){
                case 2:
                    countV++;
                    break;
                case 1:
                    countS++;
                    break;
                case 0:
                    countF++;
                    break;
            }
        }
        if (countF >= countS && countF >= countV){
            System.out.println("OnFeet");
        } else if (countS >= countF && countS >= countV){
            System.out.println("Still");
        } else if (countV >= countF && countV >= countS){
            System.out.println("InVehicle");
        }
    }
}
