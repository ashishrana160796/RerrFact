# <div align="center"> RerrFact </div>
This repository contains the code for: RerrFact model for SciVer shared task.

# Setup for Inference
## 1. Download SciFact database
Download the SciFact database from [here](https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz).

## 2. Installing requirements
Install the requirements using the following command for abstract retrieval and rationale selection module.
```
pip install -r abstract,rationale_requirements.txt
```
Install the requirements using the following command for label prediction module.
```
pip install -r label_requirements.txt 
```

## 3. Download trained models
Download the trained models using [this](https://drive.google.com/drive/folders/1ZBmHmOUjrOReGEPLOjnTm3cv78wNgP3B?usp=sharing) link.

## 4. Using pre-trained models
Abstract Retrieval
```
python ./inference/abstract-retrieval.py \
--corpus ./data/corpus.jsonl \
--dataset ./data/claims_test.jsonl \
--model ./saved_models/abstract_retrieval_model_here \
--output ./prediction/abstract_retrieval_test_predictions.jsonl
```

Rationale Selection
```
python ./inference/rationale-selection.py \
--corpus ./data/corpus.jsonl \
--dataset ./data/claims_test.jsonl \
--abstract ./prediction/abstract_retrieval_test_predictions.jsonl \
--model ./saved_models/rationale_selection_model_here \
--output ./prediction/
```

Label Prediction
```
python inference/label-prediction.py \
--corpus '/data/corpus.jsonl' \
--dataset './data/claims_test.jsonl' \
--rationale-selection './prediction/rationale_selection.jsonl' \
--model_n './saved_models/neutral_classifer_here' \
--model_s './saved_models/support_classifier_here' \
--output './prediction/label_pred_test.jsonl'
```

## Retrain models
Refer to `training/Abstract-retrieval.ipynb` for training abstract retrieval module.

Refer to `training/Rationale-selection.ipynb` for training rationale selection module.

Refer to `training/Label-prediction.ipynb` for training label prediction module.
