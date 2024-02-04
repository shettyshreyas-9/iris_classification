import pathlib
import yaml
import sys
import pandas as pd
from sklearn.model_selection import train_test_split


def main():

    curr_dir= pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    # print(curr_dir)

    # input : load data
    input_file= sys.argv[1]
    data_path= home_dir.as_posix()+input_file

    def load_data(data_path):
        df= pd.read_csv(data_path)
        return df

    data_df= load_data(data_path)


    # split data

    params_file = home_dir.as_posix()+ sys.argv[3] 
    params = yaml.safe_load(open(params_file))['make_dataset']

    def split_data(data_df, test_split,seed):
        train,test = train_test_split(data_df,test_size= test_split, random_state= seed)
        return train,test

    train_df, test_df = split_data(data_df, params['test_split'],params['seed'])

    # output: save data

    output_path= home_dir.as_posix()+ sys.argv[2]

    def save_data(train_df,test_df,output_path):
        train_df.to_csv(output_path + '/train_iris.csv', index=False)
        test_df.to_csv(output_path + '/test_iris.csv', index=False)

    save_data(train_df,test_df,output_path)


if __name__ == '__main__':
    main()

