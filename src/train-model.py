# import libraries
import mlflow
import glob
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

def main(args):
    # enable autologging
    mlflow.autolog()

    # read data
    df = get_data(args.training_data)
    # split data
    X_train, X_test, y_train, y_test = split_data(df)
    # train model
    model = train_model(X_train, y_train)
    # evaluate model
    eval_model(model, X_test, y_test)

# function reads in data
def get_data(path):
    print("Reading training data...")
    #df = pd.read_csv(path)
    all_files = glob.glob(path + "/*.csv")
    df = pd.concat((pd.read_csv(f) for f in all_files), sort=False)
    df.dropna(inplace=True)
    return df

# function that splits data
def split_data(df):
    print("Splitting data...")
    X, y = df[['pclass', 'sex', 'age', 'sibsp', 'parch', 'ticket', 'fare','cabin', 'embarked']].values, df['survived'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2)

    return X_train, X_test, y_train, y_test

# function that trains the model
def train_model(X_train, y_train):
    print("Training model...")
    model = RandomForestClassifier(max_depth=7, random_state=2)
    model.fit(X_train, y_train)
    mlflow.sklearn.save_model(model, args.trained_model)

    return model

# function that evaluates the model
def eval_model(model, X_test, y_test):
    # calculate accuracy
    y_pred = model.predict(X_test)
    acc = np.average(y_pred == y_test)
    print('Accuracy: {:.3}'.format(acc))

    # calculate AUC
    y_scores = model.predict_proba(X_test)
    auc = roc_auc_score(y_test,y_scores[:,1])
    print('AUC: {:.3}'.format(auc))

    # plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])
    fig = plt.figure(figsize=(6, 4))
    # Plot the diagonal 50% line
    plt.plot([0, 1], [0, 1], 'k--')
    # Plot the FPR and TPR achieved by our model
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig("ROC-Curve.png") 

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data',
                        type=str)
    parser.add_argument("--trained_model", dest='trained_model',
                        type=str)

    # parse args
    args = parser.parse_args()

    # return args
    return args

# run script
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
