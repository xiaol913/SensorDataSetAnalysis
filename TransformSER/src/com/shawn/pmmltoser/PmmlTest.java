package com.shawn.pmmltoser;

import org.dmg.pmml.FieldName;
import org.dmg.pmml.PMML;
import org.dmg.pmml.neural_network.NeuralNetwork;
import org.jpmml.evaluator.*;
import org.jpmml.evaluator.neural_network.NeuralNetworkEvaluator;

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

    public static void main(String[] args) throws Exception {
        InputStream is = new FileInputStream("F:\\PythonEXP\\SensorDataSetAnalysis\\MLPClassifier.pmml");
        PMML pmml = org.jpmml.model.PMMLUtil.unmarshal(is);

        ModelEvaluator<NeuralNetwork> modelEvaluator = new NeuralNetworkEvaluator(pmml);
        Evaluator evaluator = modelEvaluator;
        Map<String, Double> data = new HashMap<>();
        double[] list = {-1.3024458,6.703765,7.431602,0.04765341,-0.23822932,-0.074155025,-0.616218,6.070806,7.6769786};
        analysisData(data, list[0], list[1], list[2], "Accelerometer");
        analysisData(data, list[3], list[4], list[5], "Gyroscope");
        analysisData(data, list[6], list[7], list[8], "Gravity");
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
            } else if (inputFieldName.toString().equals("GyroscopeX")) {
                str = data.get("GyroscopeX");
            } else if (inputFieldName.toString().equals("GyroscopeY")) {
                str = data.get("GyroscopeY");
            } else if (inputFieldName.toString().equals("GyroscopeZ")) {
                str = data.get("GyroscopeZ");
            } else if (inputFieldName.toString().equals("Gyroscope_value")) {
                str = data.get("Gyroscope_value");
            } else if (inputFieldName.toString().equals("GravityX")) {
                str = data.get("GravityX");
            } else if (inputFieldName.toString().equals("GravityY")) {
                str = data.get("GravityY");
            } else if (inputFieldName.toString().equals("GravityZ")) {
                str = data.get("GravityZ");
            } else if (inputFieldName.toString().equals("Gravity_value")) {
                str = data.get("Gravity_value");
            }
            System.out.println(str);
            inputFieldValue = inputField.prepare(str);
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
        switch (unboxedTargetFieldValue.toString()) {
            case "2":
                System.out.println("InVehicle");
                break;
            case "1":
                System.out.println("Still");
                break;
            case "0":
                System.out.println("OnFeet");
                break;
        }
    }
}
