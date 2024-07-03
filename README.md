# TCR-EpiSev: Attention-Based Multiple Instance Learning Approach for Epitope-Driven Severity Prediction

The model effectively examines TCR sequences to precisely identify binding epitopes and how they associate with ones severity level. TCR-EpiSev utilizes a BERT-based language model for predicting TCR sequence specific epitopes. Subsequently, we applied multiple instance learning (MIL) using the predicted epitopes for predicting severity. To overcome the lack of publicly known TCR-epitope interactions, we applied the Open set Recognition method to effectively remove out relatively uninformative TCRs, which significantly improved the severity prediction accuracy.


![workflow](https://github.com/jaeminjj/TCR-EpiSev/blob/main/images/workflow.png)
