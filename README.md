# UmlsBERT: Clinical Domain Knowledge Augmentation of Contextual Embeddings Using the Unified Medical Language System Metathesaurus

## General info


This is the code that was used of the paper :  [UmlsBERT: Augmenting Contextual Embeddings with a Clinical Metathesaurus](https://arxiv.org/abs/2010.10391) (NAACL  2021).

In this work, we introduced UmlsBERT, a contextual embedding model capable of integrating domain knowledge during pre-training. It was trained on biomedical corpora and uses the Unified Medical Language System (UMLS) clinical metathesaurus in two ways:

- We proposed a new  multi-label loss function for the pre-training of the  Masked Language Modelling (Masked LM) task of UmlsBERT that considers the connections between medical words using the CUI attribute of UMLS. 

<p align="center">
 <img src="/images/cuiumls_updated.png" height="250" width="500">
 </p>

-  We introduced a  semantic group embedding  that enriches the input  embeddings process of UmlsBERT  by forcing the model  to take into consideration the association of the words that are part of the same semantic group.

<p align="center">
 <img src="/images/umlsbert_updated.png" height="180" width="600">
 </p>


## Technologies
This project was created with python 3.7 and PyTorch 0.4.1 and it is based on the transformer github repo of the huggingface [team](https://huggingface.co/)

## Setup
We recommend installing and running the code from within a virtual environment.

### Creating a Conda Virtual Environment
First, download Anaconda  from this [link](https://www.anaconda.com/distribution/)

Second, create a conda environment with python 3.7.
```
$ conda create -n umlsbert python=3.7
```
Upon  restarting your terminal session, you can activate the conda environment:
```
$ conda activate umlsbert 
```
### Install the required python packages
In the project root directory, run the following to install the required packages.
```
pip3 install -r requirements.txt
```
#### Install from a VM
If you start a VM, please run the following command sequentially before install the required python packages.
The following code example is for a vast.ai Virtual Machine.

```
apt-get update
apt install git-all
apt install python3-pip
apt-get install jupyter

```


## Dowload pre-trained UmlsBERT model

In order to use pre-trained  UmlsBERT model for the word embeddings (or the semantic embeddings), you need to dowload it into the folder examples/checkpoint/ from the link:
```
 wget -O umlsbert.tar.xz https://www.dropbox.com/s/kziiuyhv9ile00s/umlsbert.tar.xz?dl=0
```

into the folder examples/checkpoint/ and unzip it with the following command:
```
tar -xvf umlsbert.tar.xz
```
## Reproduce UmlsBERT


## Pretraining
- The UmlsBERT was pretrained on the MIMIC data. Unfortunately, we cannot provide the text of the MIMIC III dataset as training course is mandatory in order to access the particular dataset.

- The MIMIC III dataset can be downloaded from the following [link](https://mimic.physionet.org/gettingstarted/access/)

- The pretraining an UmlsBERT model depends on data from NLTK  so you'll have to download them. Run the Python interpreter (python3) and type the commands:
```python
>>> import nltk
>>> nltk.download('punkt')
```

- After downloading the NOTEEVENTS table in the *examples/language-modeling/* folder, run the following python code that we provide in the *examples/language-modeling/*  folder to create the  *mimic_string.txt* on the folder *examples/language-modeling/*:

```
python3 mimic.py
```

you can pre-trained a UmlsBERT model by running the following command on the *examples/language-modeling/*:

Example for pretraining Bio_clinicalBert:
```
python3 run_language_modeling.py --output_dir ./models/clinicalBert-v1  --model_name_or_path  emilyalsentzer/Bio_ClinicalBERT  --mlm     --do_train     --learning_rate 5e-5     --max_steps 150000   --block_size 128   --save_steps 1000     --per_gpu_train_batch_size 32     --seed 42     --line_by_line      --train_data_file mimic_string.txt  --umls --config_name  config.json --med_document ./voc/vocab_updated.txt
```

### Downstream Tasks
 **MedNLi task**
- MedNLI is available through the MIMIC-III derived data repository. Any individual certified to access MIMIC-III can access MedNLI through the following [link](https://physionet.org/content/mednli/1.0.0/) 

   - **Converting into an appropriate format**: After downloading and unzipping the MedNLi dataset (mednli-a-natural-language-inference-dataset-for-the-clinical-domain-1.0.0.zip) on the folder *examples/text-classification/dataset/mednli/*, run the following python code in the *examples/text-classification/dataset/mednli/* folder  that we provide in order to convert the dataset into a format that is appropriate for the UmlsBERT model
```
python3  mednli.py
```
- This python code will create the files: train.tsv,dev_matched.tsv and test_matched.tsv in the  *text-classification/dataset/mednli/mednli* folder
- We provide an example-notebook under the folder  *experiements/*:
  - [*experiements/MedNLI_task.ipynb*](https://github.com/gmichalo/umls_bert/blob/umls_clean/experiments/MedNLI_task.ipynb)

or directly run UmlsBert on the *text-classification/* folder: 
```
python3 run_glue.py --output_dir ./models/medicalBert-v1 --model_name_or_path  ../checkpoint/umlsbert   --data_dir  dataset/mednli/mednli  --num_train_epochs 3 --per_device_train_batch_size 32  --learning_rate 1e-4   --do_train --do_eval  --do_predict  --task_name mnli --umls --med_document ./voc/vocab_updated.txt
```

**NER task**
- Due to the copyright issue of i2b2 datasets, in order to download them follow the [link](https://www.i2b2.org/NLP/DataSets/Main.php).

   - **Converting into an appropriate format**: Since we wanted to directly compare with the Bio_clinical_Bert we used their code in order to convert the i2b2 dataset to a format which is appropriate for the BERT architecture which can be found in the following link: [link](https://github.com/EmilyAlsentzer/clinicalBERT/tree/master/downstream_tasks/i2b2_preprocessing) 
   
   We  provide the code for converting the i2b2 dataset with the following instruction for each dataset:

- i2b2 2006:
  - In the folder *token-classification/dataset/i2b2_preprocessing/i2b2_2006_deid* unzip the **deid_surrogate_test_all_groundtruth_version2.zip** and **deid_surrogate_train_all_version2.zip**
  - run the **create.sh** scrip with the command
    ./create.sh
  - The script will create the files: label.txt, dev.txt, test.txt, train.txt  in the *token-classification/dataset/NER/2006* folder
- i2b2 2010:
  - In the folder *token-classification/dataset/i2b2_preprocessing/i2b2_2010_relations* unzip the **test_data.tar.gz**, **concept_assertion_relation_training_data.tar.gz** and **reference_standard_for_test_data.tar.gz**
  - Run the jupyter notebook **Reformat.ipynb**
  - The notebook will create the files: label.txt, dev.txt, test.txt, train.txt  in the *token-classification/dataset/NER/2010* folder

- i2b2 2012:
  - In the folder *token-classification/dataset/i2b2_preprocessing/i2b2_2012* unzip the **2012-07-15.original-annotation.release.tar.gz** and **2012-08-08.test-data.event-timex-groundtruth.tar.gz**
  - Run the jupyter notebook **Reformat.ipynb**
  - The notebook will create the files: label.txt, dev.txt, test.txt, train.txt  in the *token-classification/dataset/NER/2012* folder

- i2b2 2014:
  - In the folder *token-classification/dataset/i2b2_preprocessing/i2b2_2014_deid_hf_risk* unzip the **2014_training-PHI-Gold-Set1.tar.gz**,**training-PHI-Gold-Set2.tar.gz** and **testing-PHI-Gold-fixed.tar.gz**
  - Run the jupyter notebook **Reformat.ipynb**
  - The notebook will create the files: label.txt, dev.txt, test.txt, train.txt  in the *token-classification/dataset/NER/2014* folder

- We provide an example-notebook under the folder  *experiements/*:
  - [ *experiements/NER_task.ipynb*](https://github.com/gmichalo/umls_bert/blob/umls_clean/experiments/2006_task.ipynb)

or directly run UmlsBert on the *token-classification/* folder:

```
python3 run_ner.py --output_dir ./models/medicalBert-v1 --model_name_or_path  ../checkpoint/umlsbert    --labels dataset/NER/2006/label.txt --data_dir  dataset/NER/2006 --do_train --num_train_epochs 20 --per_device_train_batch_size 32  --learning_rate 1e-4  --do_predict --do_eval --umls --med_document ./voc/vocab_updated.txt
```

If you find our work useful, can cite our paper using:

```
@misc{michalopoulos2020umlsbert,
      title={UmlsBERT: Clinical Domain Knowledge Augmentation of Contextual Embeddings Using the Unified Medical Language System Metathesaurus}, 
      author={George Michalopoulos and Yuanxin Wang and Hussam Kaka and Helen Chen and Alex Wong},
      year={2020},
      eprint={2010.10391},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


