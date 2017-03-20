package com.shawn.pmmltoser;

import org.dmg.pmml.PMML;
import org.jpmml.model.SerializationUtil;
import org.jpmml.model.visitors.LocatorNullifier;
import org.xml.sax.SAXException;

import javax.xml.bind.JAXBException;
import java.io.*;

/**
 * Created by ShawnG on 2017/3/20.
 */
public class PmmlToSer {
    public static void main(String[] args) throws FileNotFoundException, JAXBException, SAXException {
        InputStream is = new FileInputStream("F:\\PythonEXP\\SensorDataSetAnalysis\\models\\MLPClassifier.pmml");
        PMML pmml = org.jpmml.model.PMMLUtil.unmarshal(is);
        LocatorNullifier locatorNullifier = new LocatorNullifier();
        locatorNullifier.applyTo(pmml);
        OutputStream os = new FileOutputStream("F:\\PythonEXP\\SensorDataSetAnalysis\\models\\MLPClassifier.ser");
        try {
            SerializationUtil.serializePMML(pmml,os);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
