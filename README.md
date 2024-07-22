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

Download two fine-tuned models for making Embedding vectors.
```python
https://huggingface.co/jaeminjj/EpicPred/tree/main
```
Please put two models in model folder

# Clustering with Embedding vectors from encoder model (embedding.py)
```python
python embedding.py --count='reads'\
--model_dir='models'\
--sample_data='tutorial_data/sample1.csv'\
--sample_name='sample1'
--CDR3='cdr3'
--output_dir='/output/'
--epitopes_dir='tutorial_data/epitopes.csv'
--cluster_num=5
```
# Explanation using for training and predicting with EpicPred.py
* 1. Meta data : composed with 3 columns
  * sample
  * WHO_label (Ex, Severe, Healthy, cancer)
  * patient (same with input patient TCR information csv file)
* 2. TCR label information for each patient/sample data with 3 columns
  * cell barcode
  * score : binding score (0~1)
  * label : Eptiope information
* 3. Encoded vector for each each patient/sample TCR data (CDR3)
  * npy file
  * freq='frequency' ## weight of abundance column name in each dataset
 
* 4. parameters for EpicPred
  * frequency_or_not : using TCR reads or ratio for weight (0,1)
  * length_or_not : using length parameter to remove TCRs for equal comparison (0,1)
  * length : fixed length or TCRs (int)
  * vector_name : folder name of embedded TCR files
  * label_name : folder name of label information as clustering information for each TCR and binding score
  * dataset_name='tutorial' : folder name of dataset , for each datset you should input vector folder and label folder
  * save inner, saver outer : saving attention score files (0,1)
  * save_outer_name='outer_attention'
  * save_inner_name='inner_attention'
  * save_score_name='label'
  *lab='WHO_label' :  label information of samples
  * elements=['Severe','Healthy']
  * sample_name_col='sample'
  * metadata_dir = 'github/tutorial_data/metadata/'
  * output_dir='github/tutorial_data/output/'+dataset_name+'/'
  * output_dir1='github/tutorial_data/output1/'+dataset_name+'/'
  * sample_info_dir='github/tutorial_data/metadata/'+dataset_name+'/'
