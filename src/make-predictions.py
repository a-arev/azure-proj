# import libraries
import mlflow
import glob
import argparse
import pandas as pd
from pathlib import Path

def main(args):
    # get data
    df = get_data(args.sample_data)
    # make predictions
    preds = make_predictions(df, args.trained_model)
    # prepare predictions
    df_preds = prepare_predictions(df, preds)
    print("Saving predictions data...")
    output_df = df_preds.to_csv((Path(args.predictions_data) / "predictions_data.csv"), index = False)

# function reads in data
def get_data(path):
    print("Reading submission data...")
    all_files = glob.glob(path + "/*.csv")
    df = pd.concat((pd.read_csv(f) for f in all_files), sort=False)
    return df

# function that makes predictions
def make_predictions(df, model_path):
    print("Making predictions...")
    df1 = df.drop(['passengerid'], axis=1)
    # load model
    model = mlflow.pyfunc.load_model(model_path)
    preds = model.predict(df1[['pclass', 'sex', 'age', 'sibsp', 'parch', 'ticket', 'fare','cabin', 'embarked']])

    return preds

# function prepares predictions
def prepare_predictions(df, preds_arr):
    print("Preparing predictions...")
    preds = pd.DataFrame(preds_arr)
    df_preds = pd.concat([df[['passengerid']], preds], axis=1)
    df_preds.columns = ['PassengerId', 'Survived']
    
    return df_preds

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--sample_data", dest='sample_data',
                        type=str)
    parser.add_argument("--trained_model", dest='trained_model',
                        type=str) 
    parser.add_argument("--predictions_data", dest='predictions_data',
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
