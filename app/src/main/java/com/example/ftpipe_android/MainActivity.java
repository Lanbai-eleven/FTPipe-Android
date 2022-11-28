package com.example.ftpipe_android;

import android.os.Bundle;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import com.example.ftpipe_android.model.MNISTCNN;
import com.example.ftpipe_android.model.SUBMNISTCNN;

import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.autodiff.util.SameDiffUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.ExternalErrorsFunction;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;

import java.io.File;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class MainActivity extends AppCompatActivity {
    private Thread trainThread;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        DL4JResources.setBaseDirectory(new File(getExternalFilesDir(null), ""));


        this.trainThread = new Thread(() -> {

            try {
//                testExternalErrorsSimple();
//                CIFARClassifier.train();
//                CenterLossLeNetMNIST.train();
                train();
            } catch (Exception e) {
                e.printStackTrace();
            }
        });
        trainThread.start();
    }

    private void train() throws Exception{
        MNISTCNN test = new MNISTCNN();
        SameDiff sd = test.makeMNISTNet();
        SUBMNISTCNN sd_1 = test.simpleMakeSubModel(0, 1);
        SUBMNISTCNN sd_2 = test.simpleMakeSubModel(2, 3);
        SUBMNISTCNN sd_3 = test.simpleMakeSubModel(4, 4);
        TextView msg = findViewById(R.id.train_message);
        msg.setText("Train Start");

        double learningRate = 1e-3;
        TrainingConfig config = new TrainingConfig.Builder()
                .l2(1e-4)                               //L2 regularization
                .updater(new Adam(learningRate))        //Adam optimizer with specified learning rate
                .dataSetFeatureMapping("input")         //DataSet features array should be associated with variable "input"
                .dataSetLabelMapping("label")           //DataSet label array should be associated with variable "label"
                .build();

        msg.setText("create config");
        sd_1.setConfig(config);
        sd_2.setConfig(config);
        sd_3.setConfig(config);
        msg.setText("set config");

        int batchSize = 256;
//        int count = 0;
        DataSetIterator trainData = new MnistDataSetIterator(batchSize, true, 12345);
            DataSet curData = trainData.next();


        //forward
        INDArray indarr_1 = sd_1.forward(new HashMap<>() {{
            put("input", curData.getFeatures());
        }});
        System.out.println(Arrays.toString(indarr_1.shape()));
        msg.setText("f 1");

        INDArray indarr_2 = sd_2.forward(new HashMap<>() {{
            put("input", indarr_1);
        }});
        System.out.println(Arrays.toString(indarr_2.shape()));
        msg.setText("f 2");

        INDArray indarr_3 = sd_3.forward(new HashMap<>() {{
            put("input", indarr_2);
            put("labels", curData.getLabels());
        }});
        System.out.println(indarr_3);
        msg.setText("f 3");

        //backward
        sd_3.getModel().getVariable("labels").setArray(curData.getLabels());
        sd_3.backward(null);
        msg.setText("b 1");

        INDArray externalGrad2 = sd_3.getGrads().get("reshapedInput").reshape(-1, 8, 5, 5);
        sd_2.backward(externalGrad2);
        msg.setText("b 2");

        INDArray externalGrad1 = sd_2.getGrads().get("input");
        sd_1.backward(externalGrad1);
        msg.setText("b 3");

//        count++;
        msg.setText("Train end");

    }

    public void testExternalErrorsSimple() {
        INDArray externalGrad = Nd4j.linspace(1, 12, 12).reshape(3, 4);

        SameDiff sd = SameDiff.create();
        SDVariable var = sd.var("var", externalGrad);
        SDVariable out = var.mul("out", 0.5);
        System.out.println("1");

        Map<String, INDArray> gradMap = new HashMap<>();
        gradMap.put("out", externalGrad);
        System.out.println("2");
        ExternalErrorsFunction fn = SameDiffUtils.externalErrors(sd, null, out);
        System.out.println("3");

        Map<String, INDArray> m = new HashMap<>();
        m.put("out-grad", externalGrad);
        Map<String, INDArray> grads = sd.calculateGradients(m, sd.getVariables().keySet());
        System.out.println("4");
        INDArray gradVar = grads.get(var.name());

//        assertEquals(externalGrad.mul(0.5), gradVar);

        //Now, update and execute again:
        externalGrad = Nd4j.linspace(1, 12, 12).reshape(3, 4).muli(10);

        m.put("out-grad", externalGrad);
        grads = sd.calculateGradients(m, sd.getVariables().keySet());

        gradVar = var.getGradient().getArr();
        System.out.println("5");
//        assertEquals(externalGrad.mul(0.5), gradVar);

        //Test model serialization:
    }
}