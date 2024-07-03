# EpicPred: attention-based multiple instance learning for predicting phenotypes driven by epitope binding TCRs

Correctly identifying TCR binding epitopes is important to both understand their underlying biological
mechanism in association to some phenotype and develop T-cell mediated immunotherapy treatments. Although the
importance of the CDR3 region in TCRs for epitope recognition is well recognized, methods for profiling their interactions
in association to a certain disease or phenotype remains less studied. We developed EpicPred to identify phenotype specific
TCR-epitope interactions. EpicPred first predicts and removes unlikely interactions to reduce false positives using the
Open-set Recognition. Subsequently, multiple instance learning was used to identify TCR-epitope interactions specific to
cancer subtypes and COVID-19 severity levels.


![workflow](https://github.com/jaeminjj/TCR-EpiSev/blob/main/images/Workflow.png)

# Usage
After filtering and preprocessing data, we recommend to use GPU with EpicPred.
# Clustering with Embedding vectors from encoder model
# Input data for training and predicting with EpicPred 
* 1. Meta data : composed with 3 columns
  * sample
  * WHO_label
  * patient
* 2. TCR label information for each patient/sample data with 3 columns
  * cell barcode
  * score : binding score (0~1)
  * label : Eptiope information
* 3. Encoded vector for each each patient/sample TCR data (CDR3)
  * npy file 
# Training EpicPred

# Predictions with EpicPred
