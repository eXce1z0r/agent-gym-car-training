import gym
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle
import warnings

import contributors.Shazex_workspace.ShazexMain
import contributors.NotHappyDyadik_workspace.NotHappyDyadikMain
import contributors.DimaKovalchuk_workspace.DimaKovalchukMain
import contributors.eXce1z0r_workspace.eXce1z0rMain

warnings.filterwarnings("ignore")


class Regression:
    def __init__(self, dataset, dependent_feature='DepDelay'):
        # =============== TODO: Your code here ===============
        # One of subtask. Receives dataset, must return prediction vector
        # DepDelay - flight departure delay. You should predict departure delay depending on other features.
        # You should add the code that will:
        #   - read dataset from file
        pass

    def coolMethodThatWillDoAllTheNeededStuffWithDataset(self):
        # =============== TODO: Your code here ===============
        # This is the method that will prepare dataset
        # You should add the code that will:
        #   - clean dataset from useless features
        #   - split data set to train and test parts
        #   - implement lable encoding and one hot encoding if needed
        #   - create regression model and fit it
        #   - make prediction values of dependant feature
        #   - calculate r2_score to check prediction accuracy
        #   - save the model
        #   - return r2_score, modified dataset, predicted values vector, saved regression model
        print('Hey, sexy mama, wanna kill all humans?')