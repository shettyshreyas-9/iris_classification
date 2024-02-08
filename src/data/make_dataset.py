import pathlib
import yaml
import sys
import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def setup_logging():
    # Configure logging to write to a file and also print to the console
    log_file_path = pathlib.Path(__file__).parent.as_posix() + sys.argv[4]  # Specify the path to your log file
    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def main():

    setup_logging()

    curr_dir= pathlib.Path(__file__)
    print ('\n',curr_dir,'\n')
    home_dir = curr_dir.parent.parent.parent

    print ('\n',home_dir,'\n')

    # input : load data
    input_file= sys.argv[1]
    data_path= home_dir.as_posix()+input_file

    def load_data(data_path):
        df= pd.read_csv(data_path)
        return df

    data_df= load_data(data_path)


    # preprocess data

    output_path= home_dir.as_posix()+ sys.argv[2]

    def preprocess_data(data_df,output_path):
        le= LabelEncoder()
        data_df['encoded_labels']= le.fit_transform(data_df['Species'])
        data_df.to_csv(output_path + '/processed_data.csv', index=False)
        logging.info("Data preprocessing completed")
        

    preprocess_data(data_df,output_path)

    # split data

    params_file = home_dir.as_posix()+ sys.argv[3] 
    params = yaml.safe_load(open(params_file))['make_dataset']

    process_df= pd.read_csv(output_path+'/processed_data.csv')

    def split_data(process_df, test_split,seed):
        train,test = train_test_split(process_df,test_size= test_split, random_state= seed)
        return train,test

    train_df, test_df = split_data(process_df, params['test_split'],params['seed'])

    # output: save data

    output_path= home_dir.as_posix()+ sys.argv[2]

    def save_data(train_df,test_df,output_path):
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
        train_df.to_csv(output_path + '/train_iris.csv', index=False)
        test_df.to_csv(output_path + '/test_iris.csv', index=False)
        logging.info("Data saved successfully")

    save_data(train_df,test_df,output_path)


if __name__ == '__main__':
    main()

