package com.example.ftpipe_android.model;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling2DConfig;
import org.nd4j.weightinit.impl.XavierInitScheme;

public class MNISTCNN {

    //TODO: 普适性的方法切割模型
    private long inputSize[][] = {{256, 784}, {256, 4, 26, 26}, {256, 4, 13, 13}, {256, 8, 11, 11}, {256, 8, 5, 5}};

    private int layers = 5;
    private SameDiff model;

    public SameDiff getModel() {
        return model;
    }

    public SameDiff makeMNISTNet() {
        SameDiff sd = SameDiff.create();

        //Properties for MNIST dataset:
        int nIn = 28 * 28;
        int nOut = 10;

        //Create input and label variables
        SDVariable in = sd.placeHolder("input", DataType.FLOAT, -1, nIn);                 //Shape: [?, 784] - i.e., minibatch x 784 for MNIST
        SDVariable label = sd.placeHolder("label", DataType.FLOAT, -1, nOut);             //Shape: [?, 10] - i.e., minibatch x 10 for MNIST

        SDVariable reshaped = in.reshape(-1, 1, 28, 28);

        Pooling2DConfig poolConfig = Pooling2DConfig.builder().kH(2).kW(2).sH(2).sW(2).build();

        Conv2DConfig convConfig = Conv2DConfig.builder().kH(3).kW(3).build();

        // layer 1: Conv2D with a 3x3 kernel and 4 output channels
        SDVariable w0 = sd.var("w0", new XavierInitScheme('c', 28 * 28, 26 * 26 * 4), DataType.FLOAT, 3, 3, 1, 4);
        SDVariable b0 = sd.zero("b0", 4);

        SDVariable conv1 = sd.cnn().conv2d(reshaped, w0, b0, convConfig);

        // layer 2: MaxPooling2D with a 2x2 kernel and stride, and ReLU activation
        SDVariable pool1 = sd.cnn().maxPooling2d(conv1, poolConfig);

        SDVariable relu1 = sd.nn().relu(pool1, 0);

        // layer 3: Conv2D with a 3x3 kernel and 8 output channels
        SDVariable w1 = sd.var("w1", new XavierInitScheme('c', 13 * 13 * 4, 11 * 11 * 8), DataType.FLOAT, 3, 3, 4, 8);
        SDVariable b1 = sd.zero("b1", 8);

        SDVariable conv2 = sd.cnn().conv2d(relu1, w1, b1, convConfig);

        // layer 4: MaxPooling2D with a 2x2 kernel and stride, and ReLU activation
        SDVariable pool2 = sd.cnn().maxPooling2d(conv2, poolConfig);

        SDVariable relu2 = sd.nn().relu(pool2, 0);

        SDVariable flat = relu2.reshape(-1, 5 * 5 * 8);

        // layer 5: Output layer on flattened input
        SDVariable wOut = sd.var("wOut", new XavierInitScheme('c', 5 * 5 * 8, 10), DataType.FLOAT, 5 * 5 * 8, 10);
        SDVariable bOut = sd.zero("bOut", 10);

        SDVariable z = sd.nn().linear("z", flat, wOut, bOut);

        // softmax crossentropy loss function
        SDVariable out = sd.nn().softmax("out", z, 1);
        SDVariable loss = sd.loss().softmaxCrossEntropy("loss", label, out, null);

        sd.setLossVariables(loss);

        this.model = sd;
        return sd;
    }

    public SUBMNISTCNN simpleMakeSubModel(int start, int end) {
        if(start > end){
            System.out.println("start > end, error!");
            return null;
        }
        SameDiff sd = SameDiff.create();

        long[] curInput = inputSize[start];

//        SDVariable inReshaped = sd.placeHolder("input", DataType.FLOAT, curInput);
        SDVariable inReshaped = sd.var("input", DataType.FLOAT, curInput);

        Pooling2DConfig poolConfig = Pooling2DConfig.builder().kH(2).kW(2).sH(2).sW(2).build();

        Conv2DConfig convConfig = Conv2DConfig.builder().kH(3).kW(3).build();


        for (int i = start; i <= end; i++) {
            String name = "inter_" + i;
            if (i == end) {
                name = "output";
            }
            switch (i) {
                case 0:
                    SDVariable w0 = sd.var("w0", new XavierInitScheme('c', 28 * 28, 26 * 26 * 4), DataType.FLOAT, 3, 3, 1, 4);
                    SDVariable b0 = sd.zero("b0", 4);

                    inReshaped = sd.reshape("reshapedInput", inReshaped, -1, 1, 28, 28);
                    // inReshaped = inReshaped.reshape(-1, 1, 28, 28);
                    inReshaped = sd.cnn().conv2d(name, inReshaped, w0, b0, convConfig);
                    break;
                case 1:
                    SDVariable pool1 = sd.cnn().maxPooling2d(inReshaped, poolConfig);

                    inReshaped = sd.nn().relu(name, pool1, 0);
                    break;
                case 2:
                    SDVariable w1 = sd.var("w1", new XavierInitScheme('c', 13 * 13 * 4, 11 * 11 * 8), DataType.FLOAT, 3, 3, 4, 8);
                    SDVariable b1 = sd.zero("b1", 8);

                    inReshaped = sd.cnn().conv2d(name, inReshaped, w1, b1, convConfig);
                    break;
                case 3:
                    SDVariable pool2 = sd.cnn().maxPooling2d(inReshaped, poolConfig);
                    inReshaped = sd.nn().relu(name, pool2, 0);
                    break;
                case 4:
                    int nOut = 10;
                    SDVariable labels = sd.placeHolder("labels", DataType.FLOAT, -1, nOut);

                    SDVariable wOut = sd.var("wOut", new XavierInitScheme('c', 5 * 5 * 8, 10), DataType.FLOAT, 5 * 5 * 8, 10);
                    SDVariable bOut = sd.zero("bOut", 10);

                    inReshaped = sd.reshape("reshapedInput", inReshaped, -1, 5 * 5 * 8);
                    //inReshaped = inReshaped.reshape(-1, 5 * 5 * 8);
                    SDVariable z = sd.nn().linear("z", inReshaped, wOut, bOut);

                    // softmax crossentropy loss function
                    SDVariable out = sd.nn().softmax("output", z, 1);
                    SDVariable loss = sd.loss().softmaxCrossEntropy("loss", labels, out, null);

                    sd.setLossVariables(loss);
                    break;
                default:
                    System.out.print("Unknown layer: " + i);
            }
        }

        return new SUBMNISTCNN(sd, this.layers, start, end);
    }


}
