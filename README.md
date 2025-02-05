# HB-LT
Haplotype Block LSTM Hierarchical Transformer.  
This is for the article `Leveraging hierarchical structures for genetic block interaction studies using the hierarchical transformer` doi: https://doi.org/10.1101/2024.11.18.24317486. 
**Python version**: This code is in Python3.9

**Package Requirements**: pytorch tensorflow scikit-learn

The current framework is inspired by Liu and Lapata (2019) (https://github.com/nlpyang/hiersumm.git)

### Pre-analysis
Researchers are strongly suggested to perform covariant pre-filtering methods to limit potential confounding effects. A PCA-based pre-filtering method is provided and can be found in "covariants_filtering.py."

The current framework takes a unique data structure as input (see the example data file). We provided a script, "VCFtoBlock.py," to help users convert their VCF files into files that can be processed by the current pipeline. The structure of the input data will be more clear after referring to the sample_data folder. 

### HB-LT 
The main framework Haplotype Block LSTM hierarchical Transformer (HB-LT) can be run by HB-LT.py. HB-LT mainly consists of two parts, Long short-term memory (LSTM) filtering and block "interaction" (both within and cross) mapping. 
![image](https://github.com/user-attachments/assets/c085e521-133f-4507-836e-ac36b88c5bb9)

#### LSTM
The LSTM filtering stage takes each block individually and tests its association with the phenotype based on the root of mean square error (RMSE). There is no universal way to decide what is the best RMSE cut-off threshold. The easiest option is to look through the line plot and take the top x% e.g. 10%, 20%. 

#### HB-LT 
The main output of the HB-LT framework is within and cross-block "attention" heatmap. HB-LT utilizes a multi-head attention mechanism, the default setting is to output the mean of the multi-attention filter and use it as the output. However, the users can also take the sum or directly output multiple heatmaps without aggregation. 

#### Of Note
HB-LT framework is intended to be used as a mining framework, any signals that HB-LT has mapped cannot be directly interpreted as a "real" interaction/epistasis signal. It should be used as a first-stage mining, any signals observed should be subjected to the following statistical and biological validations.




