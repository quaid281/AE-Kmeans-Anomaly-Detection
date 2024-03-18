import numpy as np
import os
import logging
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
logger = logging.getLogger('telemanom')

class Channel:
    def __init__(self, config, chan_id):
        """
        Load and reshape channel values (predicted and actual).

        Args:
            config (obj): Config object containing parameters for processing
            chan_id (str): channel id

        Attributes:
            id (str): channel id
            config (obj): see Args
            X_train (arr): training inputs with dimensions
                [timesteps, l_s, input dimensions)
            X_test (arr): test inputs with dimensions
                [timesteps, l_s, input dimensions)
            y_train (arr): actual channel training values with dimensions
                [timesteps, n_predictions, 1)
            y_test (arr): actual channel test values with dimensions
                [timesteps, n_predictions, 1)
            train (arr): train data loaded from .npy file
            test(arr): test data loaded from .npy file
        """

        self.id = chan_id
        self.config = config
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_hat = None
        self.train = None
        self.test = None
        self.latent = None
        
        if self.config.TranAD==True: self.config.l_s=10

    def feature_engineering(self,data):
        df = pd.DataFrame(data)

        #####  Statistical features ######
        df['mean'] = df[0].rolling(20).mean()
        df['std'] = df[0].rolling(20).std()
        df['max'] = df[0].rolling(20).max()
        df['min'] = df[0].rolling(20).min()

        df['pct_change3'] = df[0].pct_change(3)
    
        ##### Seasonality and Trend ######
        df['trend'] = df[0].rolling(20).mean().diff()
        df['seasonality'] = df[0] - df['trend']
        df['abs_diff'] = df[0].diff().abs()
        
        df.ffill(inplace=True)
        df.dropna(inplace=True)
        numpy_array = df.values
        return numpy_array

    def shape_data(self, arr, train=True):
        """Shape raw input streams for ingestion into LSTM. config.l_s specifies
        the sequence length of prior timesteps fed into the model at
        each timestep t.

        Args:
            arr (np array): array of input streams with
                dimensions [timesteps, 1, input dimensions]
            train (bool): If shaping training data, this indicates
                data can be shuffled
        """

        data = []
        for i in range(len(arr) - self.config.l_s - self.config.n_predictions):
            data.append(arr[i:i + self.config.l_s + self.config.n_predictions])
        data = np.array(data)

        assert len(data.shape) == 3

        if train:
            np.random.shuffle(data)
            self.X_train = data[:, :-self.config.n_predictions, :]
            self.y_train = data[:, -self.config.n_predictions:, 0]  # telemetry value is at position 0
            
        else:
            self.X_test = data[:, :-self.config.n_predictions, :]
            self.y_test = data[:, -self.config.n_predictions:, 0]  # telemetry value is at position 0

    def load_data(self):
        """
        Load train and test data from local.
        """
        try:
            
            parent_dir = os.path.dirname(os.getcwd())
            self.train = np.load(os.path.join(parent_dir,"data",'raw', "train", "{}.npy".format(self.id)))   
            self.test = np.load(os.path.join(parent_dir,"data", 'raw',"test", "{}.npy".format(self.id)))
            #self.train = self.feature_engineering(self.train)
            #self.test = self.feature_engineering(self.test)
        except FileNotFoundError as e:
            logger.critical(e)
            logger.critical("Source data not found, may need to add data to repo: <link>")

        self.shape_data(self.train)
        self.shape_data(self.test, train=False)