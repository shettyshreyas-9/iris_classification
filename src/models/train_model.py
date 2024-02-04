import pathlib
import yaml
import sys

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def main():

    curr_dir= pathlib.Path(__file__)
    home_dir= curr_dir.parent.parent.parent

    params_file= home_dir.as_posix()+ sys.argv[2]
    print(params_file)

    








if __name__ == '__main__':
    main()
