# Description
This repo holds files to setup and run a pipeline job in Azure Machine Learning. The goal is to build 
a model that **predicts the survival of a passenger** encountering the Titanic disaster. Kaggle provides the datasets 
and hosts this as a competition. Two datasets are given; one to build the model and the other to make predictions 
and submit on Kaggle for scoring.

# Model Details
The model used is sci-kit learn's **Random Forest Classifier**. My model was built with the following transformations 
made to the training dataset:
- first and last names extracted into separate new features
- applying target encoding for 'pclass', 'sex', and 'embarked' features
- applying catboost encoding 'lname', 'ticket', and 'cabin' features
- using iterative and KNN imputers for imputing missing data

# Model Results
The model that these notebooks provide scored a **accuracy of 0.80143** on Kaggle. This corresponds to the 778th spot 
on the leader board (out of ~16,000).

This is the ROC curve that corresponds to the training dataset from kaggle (i.e. dataset used for building model, not the dataset used for submissions).
![ROC-Curve](https://github.com/user-attachments/assets/423bd1fc-6441-4f09-b3f1-6cbf388fdcef)

# Files
---
- data
    -  **titanic.csv** - titanic training dataset, used for building model
    -  **titanic_sample.csv** - sample dataset used for making predictions and submitting to Kaggle for scoring
- src
    - **make-predictions.py** - script for making predictions step in pipeline job
    - **prep-data.py** - script for prepping data step in pipeline job
    - **train-model.py** - script for training model step in pipeline job
- **make-predictions.yml** - specifications file for *make-predictions.py* script
- **pipeline-job.ipynb** - jupyter notebook used to orchestrate pipeline job
- **prep-data.yml** - specifications file for *prep-data.py* script
- **setup.sh** - shell script used to setup workspace, compute, and data assets
- **titanic-env.txt** - file with context for building enviroment for job
- **train-model.yml** - specifications file for *train-model.py* script

# Wiki Page 
See [Wiki page](https://github.com/a-arev/titanic-competition/wiki) (menu above) for information on recreating Azure environment and visualizing scripts with schematics.
