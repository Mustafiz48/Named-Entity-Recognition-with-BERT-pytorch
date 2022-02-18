# Entity Extraction Model with BERT

We will be using BertForTokenClassification for the task of Entity Extraction

## Dataset
The ner dataset is collected from kaggle. The dataset can be found [Here](https://www.kaggle.com/rajnathpatel/ner-data)

About the license of the database: 

License:
Database: Open Database, Contents: Database Contents


## Installation

The necessary packages are listed in 'requirements.txt' file. To install them run the following command

```bash
pip install -r requirements.txt
```

## Usage
- First, train the model with the command:
```bash
  python train_model.py
```

This will train the model and then save the model in 'saved_model' folder that will be used later.

- Then, to evaluate the model's performance, run:
```bash
python evaluate_model.py
```
- Finally, to make inference, run:
```bash
python infer_model.py
```
To make inference, to get input from an input file, there's line in the code that is commented out. Using that "read_csv" line will take input from a file. For the sake of test, a portion of data is taken and then tested with. The result will be saved in a file named output.csv 

## Fine-Tune Consideration

- Learning rate
- input seq length
- optimizer etc.