package com.shawn.pmmltoser;

import org.dmg.pmml.FieldName;
import org.dmg.pmml.PMML;
import org.dmg.pmml.neural_network.NeuralNetwork;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.evaluator.*;
import org.jpmml.evaluator.neural_network.NeuralNetworkEvaluator;
import org.jpmml.evaluator.tree.TreeModelEvaluator;

import java.io.FileInputStream;
import java.io.InputStream;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by ShawnG on 2017/3/20.
 */
public class PmmlTest {
    public static Map<String, Double> analysisData(Map<String, Double> data, double x, double y, double z, String type) {
        double a = Math.abs(x);
        double b = Math.abs(y);
        double c = Math.abs(z);
        double v = Math.pow(a, 2) + Math.pow(b, 2) + Math.pow(c, 2);
        String strX = type + "X";
        String strY = type + "Y";
        String strZ = type + "Z";
        String strV = type + "_value";
        data.put(strX, a);
        data.put(strY, b);
        data.put(strZ, c);
        data.put(strV, v);
        return data;
    }

    public static Map<String, Double> linearData(Map<String, Double> data, double[] list, String type) {
        double a = Math.abs(list[0] - list[3]);
        double b = Math.abs(list[1] - list[4]);
        double c = Math.abs(list[2] - list[5]);
        double v = Math.pow(a, 2) + Math.pow(b, 2) + Math.pow(c, 2);
        String strX = type + "X";
        String strY = type + "Y";
        String strZ = type + "Z";
        String strV = type + "_value";
        data.put(strX, a);
        data.put(strY, b);
        data.put(strZ, c);
        data.put(strV, v);
        return data;
    }

    public static void main(String[] args) throws Exception {
        InputStream is = new FileInputStream("F:\\PythonEXP\\SensorDataSetAnalysis\\models\\MLPClassifier.pmml");
        PMML pmml = org.jpmml.model.PMMLUtil.unmarshal(is);
        ModelEvaluatorFactory modelEvaluatorFactory = ModelEvaluatorFactory.newInstance();
        ModelEvaluator<?> modelEvaluator = modelEvaluatorFactory.newModelEvaluator(pmml);
        Evaluator evaluator = modelEvaluator;
        Map<String, Double> data = new HashMap<>();
        double[] list = {
                4.711789,0.038307,8.504204,
                4.706174,0.017293,8.603604
        };
        analysisData(data, list[0], list[1], list[2], "Accelerometer");
        analysisData(data, list[3], list[4], list[5], "Gravity");
        linearData(data, list, "Linear");
        Map<FieldName, FieldValue> arguments = new LinkedHashMap<>();
        List<InputField> inputFields = evaluator.getInputFields();
        for (InputField inputField : inputFields) {
            FieldName inputFieldName = inputField.getName();
            FieldValue inputFieldValue = null;
            Object str = null;
            if (inputFieldName.toString().equals("AccelerometerX")) {
                str = data.get("AccelerometerX");
            } else if (inputFieldName.toString().equals("AccelerometerY")) {
                str = data.get("AccelerometerY");
            } else if (inputFieldName.toString().equals("AccelerometerZ")) {
                str = data.get("AccelerometerZ");
            } else if (inputFieldName.toString().equals("Accelerometer_value")) {
                str = data.get("Accelerometer_value");
            } else if (inputFieldName.toString().equals("GravityX")) {
                str = data.get("GravityX");
            } else if (inputFieldName.toString().equals("GravityY")) {
                str = data.get("GravityY");
            } else if (inputFieldName.toString().equals("GravityZ")) {
                str = data.get("GravityZ");
            } else if (inputFieldName.toString().equals("Gravity_value")) {
                str = data.get("Gravity_value");
            } else if (inputFieldName.toString().equals("LinearX")) {
                str = data.get("LinearX");
            } else if (inputFieldName.toString().equals("LinearY")) {
                str = data.get("LinearY");
            } else if (inputFieldName.toString().equals("LinearZ")) {
                str = data.get("LinearZ");
            } else if (inputFieldName.toString().equals("Linear_value")) {
                str = data.get("Linear_value");
            }
            System.out.println(inputField);
//            inputFieldValue = inputField.prepare(str);
//            arguments.put(inputFieldName, inputFieldValue);
//        }
//        Map<FieldName, ?> results = evaluator.evaluate(arguments);
//        List<TargetField> targetFields = evaluator.getTargetFields();
//        Object targetFieldValue = null;
//        for (TargetField targetField : targetFields) {
//            FieldName targetFieldName = targetField.getName();
//            targetFieldValue = results.get(targetFieldName);
//        }
//        Object unboxedTargetFieldValue = null;
//        if (targetFieldValue instanceof Computable) {
//            Computable computable = (Computable) targetFieldValue;
//
//            unboxedTargetFieldValue = computable.getResult();
//        }
//        switch (unboxedTargetFieldValue.toString()) {
//            case "2":
//                System.out.println("InVehicle");
//                break;
//            case "1":
//                System.out.println("Still");
//                break;
//            case "0":
//                System.out.println("OnFeet");
//                break;
        }
    }
}