# import libraries
import argparse
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer

def main(args):
    # read data
    df_raw = get_data(args.unclean_data)
    # prep data
    df_prep = prep_data(df_raw)
    print("Saving preprocessed data...")
    output_df = df_prep.to_csv((Path(args.clean_data) / "prepped_data.csv"), index = False)

# function reads in data
def get_data(path):
    print('Reading raw data...')
    df = pd.read_csv(path)
    
    return df

# funtion preprocesses data
def prep_data(df):
    print('Preprocessing raw data...')
    # lowercase column names
    df.columns = [str_colName.lower() for str_colName in df.columns]
    
    # drop unwanted features
    dropFeats = ['name']
    for col in dropFeats:
        if col in df.columns:
            df = df.drop(col, axis=1)

    # numerical and categorical features
    numFeats = ['age', 'sibsp', 'parch', 'fare']
    catFeats = ['pclass', 'sex', 'ticket', 'cabin', 'embarked']
    otherFeats = [feat for feat in df.columns if feat not in numFeats+catFeats]
    
    # impute numerical data
    ii = IterativeImputer(random_state=2)
    ii_data = ii.fit_transform(df[numFeats])
    dfNum = pd.DataFrame(ii_data, columns=numFeats)
    
    # impute categorical data
    si = SimpleImputer(strategy='most_frequent')
    si_data = si.fit_transform(df[catFeats])
    dfCat = pd.DataFrame(si_data, columns=catFeats)
    # concatenate df
    df = pd.concat([df[otherFeats], dfNum, dfCat], axis=1)
    
    # scale numerical features
    scaler = MinMaxScaler()
    df[numFeats] = scaler.fit_transform(df[numFeats])
    
    # encode categorical features
    encoder = LabelEncoder()
    for col in catFeats:
        df[col] = encoder.fit_transform(df[col])
    
    return df

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--unclean_data", dest='unclean_data',
                        type=str)
    parser.add_argument("--clean_data", dest='clean_data',
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
    
