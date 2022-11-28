package com.example.ftpipe_android.model;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.util.SameDiffUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.ExternalErrorsFunction;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class SUBMNISTCNN {
    private SameDiff model;
    private TrainingConfig config;

    private int end_num;
    private int start;
    private int end;

    private Map<String, INDArray> grads;

    public SUBMNISTCNN(SameDiff model, int end_num, int start, int end){
        this.model = model;
        this.end_num = end_num;
        this.start = start;
        this.end = end;
    }


    public SameDiff getModel() {
        return model;
    }

    public Map<String, INDArray> getGrads() {
        return grads;
    }

    public void setConfig(TrainingConfig config) {
        this.config = config;
        this.model.setTrainingConfig(this.config);
    }

    public INDArray forward(Map<String, INDArray> input){
        String output = (end == end_num-1)? "loss" : "output";
        this.model.output(input, output);
        return this.model.getArrForVarName(output);
    }

    public void backward(INDArray externalGrad){
        if(end == end_num-1){
            this.grads = this.model.calculateGradients(null, this.model.getVariables().keySet());
        }
        else{
            ExternalErrorsFunction fn = SameDiffUtils.externalErrors(this.model, null, this.model.getVariable("output"));
            Map<String, INDArray> externalGradMap = new HashMap<String, INDArray>(){{put("output-grad", externalGrad);}};
            this.grads = this.model.calculateGradients(externalGradMap, this.model.getVariables().keySet());
        }

        //select variables to update grad
        Set<SDVariable> variableToUpdate = new HashSet<>();
        for (SDVariable variable:this.model.variables()){
            if(variable.getVariableType() == VariableType.VARIABLE){
                variableToUpdate.add(variable);
            }
        }

        for (SDVariable variable: variableToUpdate){
            INDArray gradArr = variable.getGradient().getArr();
            INDArray nowArr = variable.getArr();
            if (gradArr != null){
                nowArr.sub(gradArr);
            }
        }

    }

}
