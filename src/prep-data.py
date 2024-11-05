# import libraries
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import TargetEncoder
from category_encoders.cat_boost import CatBoostEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer

def main(args):
    # read data
    df_rw1 = get_data(args.unclean_titanic)
    df_rw2 = get_data(args.unclean_sample)

    # prep data
    df_ppd1, df_ppd2 = prep_data(df_rw1, df_rw2)
    print("Saving preprocessed data...")
    output_df1 = df_ppd1.to_csv((Path(args.clean_titanic) / "clean_titanic.csv"), index = False)
    output_df2 = df_ppd2.to_csv((Path(args.clean_sample) / "clean_sample.csv"), index = False)

# function reads in data
def get_data(path):
    print('Reading raw data...')
    df = pd.read_csv(path)
    
    return df

# funtion preprocesses data
def prep_data(df1, df2):
    print('Preprocessing raw data...')
    # lowercase column names
    df1.columns = [col.lower() for col in df1.columns]
    df2.columns = [col.lower() for col in df2.columns]

    # create first name and last name columns
    df1[['fname', 'lname']] = df1['name'].apply(lambda x: pd.Series(fl_names(x)))
    df2[['fname', 'lname']] = df2['name'].apply(lambda x: pd.Series(fl_names(x)))
    
    # target encode some features
    teFeats = ['pclass', 'sex', 'embarked']
    df1, df2 = targEncoder(df1, df2, teFeats)

    # one hot encode last names
    cbeFeats = ['lname', 'ticket', 'cabin']
    df1, df2 = catBoostEncoder(df1, df2, cbeFeats)

    # impute missing data
    numFeats = ['age', 'sibsp', 'parch', 'fare']
    catFeats = ['pclass', 'sex', 'ticket', 'cabin', 'embarked']
    df1, df2 = dataImputer(df1, df2, numFeats, catFeats)

    return df1, df2

# function creates first and last name features
def fl_names(text):
    # first, middle, last name
    fml = text.split(',')
    # last name
    l = fml[0].strip().lower()
    if 'Mrs.' in text:
        if '(' in text:
            # first and middle name
            fm = fml[1].split('(')[1].split()
            # first name
            f = fm[0].replace(')', '') \
                     .strip() \
                     .lower()
            return f, l
        else:  
            # first and middle name
            fm = fml[1].split()
            # first name
            f = fm[1].strip().lower()
            return f, l
    else:  
        # first and middle name
        fm = fml[1].split()
        # first name
        f = fm[1].replace('(', '') \
                 .replace(')', '') \
                 .strip() \
                 .lower()
        return f, l    

# function to target encode features
def targEncoder(df1, df2, feats):
    te = TargetEncoder(random_state=2)
    data1 = te.fit_transform(df1[feats], df1[['survived']])
    data2 = te.transform(df2[feats])
    df1[feats] = pd.DataFrame(data1, columns=feats)
    df2[feats] = pd.DataFrame(data2, columns=feats)

    return df1, df2

# function to catboost encode features
def catBoostEncoder(df1, df2, feats):
    cbe = CatBoostEncoder()
    df1[feats] = cbe.fit_transform(df1[feats], df1['survived'])
    df2[feats] = cbe.transform(df2[feats])

    return df1, df2

# function to impute missing data
def dataImputer(df1, df2, numFeats, catFeats):
    # get list of other features
    otherFeats1 = [feat for feat in df1.columns if feat not in numFeats+catFeats]
    otherFeats2 = [feat for feat in df2.columns if feat not in numFeats+catFeats]

    # impute data in numerical features
    ii = IterativeImputer(random_state=2)
    ii_data1 = ii.fit_transform(df1[numFeats])
    ii_data2 = ii.transform(df2[numFeats])
    dfNum1 = pd.DataFrame(ii_data1, columns=numFeats)
    dfNum2 = pd.DataFrame(ii_data2, columns=numFeats)
    # impute data in categorical features
    knni = KNNImputer(n_neighbors=7)
    knni_data1 = knni.fit_transform(df1[catFeats])
    knni_data2 = knni.transform(df2[catFeats])
    dfCat1 = pd.DataFrame(knni_data1, columns=catFeats)
    dfCat2 = pd.DataFrame(knni_data2, columns=catFeats)

    # concatenate df's into one
    df1 = pd.concat([df1[otherFeats1], dfNum1, dfCat1], axis=1)
    df2 = pd.concat([df2[otherFeats2], dfNum2, dfCat2], axis=1)

    return df1, df2

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--unclean_titanic", dest='unclean_titanic',
                        type=str)
    parser.add_argument("--unclean_sample", dest='unclean_sample',
                        type=str)
    parser.add_argument("--clean_titanic", dest='clean_titanic',
                        type=str)
    parser.add_argument("--clean_sample", dest='clean_sample',
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
    
