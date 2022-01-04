from comet_ml import Experiment
import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.momentum import RSIIndicator, rsi
from sklearn.ensemble import RandomForestClassifier
import os
import alpaca_trade_api as tradeapi
import warnings
from alpaca_trade_api.rest import REST, TimeFrame, TimeFrameUnit
from ta.volatility import BollingerBands
from ta.trend import MACD
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from metaflow import FlowSpec, step, Parameter, IncludeFile, current
from datetime import datetime
import requests, json
import torch
from torch import nn
#from torch._C import float64
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.nn.modules.loss import MSELoss
import tqdm

warnings.filterwarnings('ignore')
# import comet_ml at the top of your file


# Create an experiment with your api key
experiment = Experiment(
    api_key="rtbRK1bGHyBQIfwLmTLbwgXRn",
    project_name="dustinfinalproject",
    workspace="nyu-fre-7773-2021",
)
#class DifferentRegressionFlow(FlowSpec):
    
    #@step
   # def start(self):
"""
Start up and print out some info to make sure everything is ok metaflow-side

print("Starting up at {}".format(datetime.utcnow()))
# debug printing - this is from https://docs.metaflow.org/metaflow/tagging
# to show how information about the current run can be accessed programmatically
print("flow name: %s" % current.flow_name)
print("run id: %s" % current.run_id)
print("username: %s" % current.username)
#self.next(self.security)
"""
    #@step
    #def security(self):
        # Specify paper trading environment
os.environ['APCA_API_BASE_URL'] = 'https://paper-api.alpaca.markets'

APCA_API_BASE_URL = 'https://paper-api.alpaca.markets'
APCA_API_KEY_ID = 'PKUXDA9XH5DO656AWGP0'
APCA_API_SECRET_KEY = 'KSedQnzL5sbZNEfaCIVZJNeRHJqnt6iVWSUtGcVR'
api = tradeapi.REST(APCA_API_KEY_ID, APCA_API_SECRET_KEY,  APCA_API_BASE_URL, api_version='v2')
account = api.get_account()

    #@step
    #def process_data(self, api):
aapl_df = api.get_bars("AAPL", TimeFrame(4, TimeFrameUnit.Hour), "2016-02-22", "2018-02-22", adjustment='raw').df
close_series = aapl_df['close']

#print(close_series)
#print(aapl_df['close'])
#close_series = aapl_df['close']
#print(close_series)
#print(len(aapl_df.index))
#print(close_df)
#print(close_df)
#rsi_series = RSIIndicator(close_series, window=17, fillna=True)
#print(rsi_series)
'''
bb_indicator = BollingerBands(close_series, 1)
upper_band = bb_indicator.bollinger_hband()
print(upper_band)
'''
rsi_obj = RSIIndicator(close_series, 2, fillna=True)
rsi_ind = rsi_obj.rsi()
rsi_df = rsi_ind[2:].to_frame()
rsi_df.dropna()
print(rsi_df)

macd_obj = MACD(close_series, fillna=True)
macd_ind = macd_obj.macd_diff()
macd_df = macd_ind[2:].to_frame()
macd_df.dropna() 
print(macd_ind)

new_df = close_series.pct_change()
new_df = new_df[2:]
new_df.dropna()
print(new_df)

'''
print(len(rsi_df.index))
print(len(macd_df.index))
print(len(new_df.index))
'''
rsi_np = rsi_df.to_numpy().astype(np.float64).flatten()
#print(f"RSI Shape: {rsi_np.shape}")
macd_np = macd_df.to_numpy().astype(np.float64).flatten()
#print(macd_np.shape)
pct_change_np = new_df.to_numpy().astype(np.float64)
#print(pct_change_np.shape)

#rsi_macd_pct_torch = torch.DoubleTensor(np.hstack((rsi_np, macd_np, pct_change_np)))

#print(f'Torch Shape: {rsi_macd_pct_np.shape}')
#self.next()
#print(rsi_macd_np.shape)







aapl_df_test = api.get_bars("AAPL", TimeFrame(4, TimeFrameUnit.Hour), "2018-02-23", "2020-02-29", adjustment='raw').df
close_series_test = aapl_df_test['close'][:-1]

#print(close_series)
#print(aapl_df['close'])
#close_series = aapl_df['close']
#print(close_series)
#print(len(aapl_df.index))
#print(close_df)
#print(close_df)
#rsi_series = RSIIndicator(close_series, window=17, fillna=True)
#print(rsi_series)
'''
bb_indicator = BollingerBands(close_series, 1)
upper_band = bb_indicator.bollinger_hband()
print(upper_band)
'''
rsi_obj_test = RSIIndicator(close_series_test, 2, fillna=True)
rsi_ind_test = rsi_obj_test.rsi()
rsi_df_test= rsi_ind_test[2:].to_frame()
#print(rsi_df_test)

macd_obj_test = MACD(close_series_test, fillna=True)
macd_ind_test = macd_obj_test.macd_diff()
macd_df_test = macd_ind_test[2:].to_frame()
#print(macd_ind)

new_df_test = close_series_test.pct_change()
new_df_test = new_df_test[2:]
#print(new_df)
pct_change_np_test = new_df_test.to_numpy().astype(np.float64).reshape(2190, 1)
#print(pct_change_np.shape)
rsi_np_test = rsi_df_test.to_numpy().astype(np.float64)
#print(f"RSI Shape: {rsi_np.shape}")
macd_np_test = macd_df_test.to_numpy().astype(np.float64)
#print(macd_np.shape)



#frames = [rsi_df, macd_df, new_df]
rsi_macd_pct_df_train = []

class TrainingDataset(torch.utils.data.Dataset):
    def __init__(self, rsi, macd, pct_change, predict_days=60):
      self.rsi_macd = self.get_rsi_macd(rsi, macd)

      #self.pct_change = F.normalize(
      #    torch.Tensor(pct_change).unsqueeze(0)
      #).squeeze(0)
      self.pct_change = torch.Tensor(pct_change)

      self.predict_days = predict_days

    @classmethod
    def get_rsi_macd(cls, rsi, macd, norm=True):
        rsi = torch.Tensor(rsi).unsqueeze(0)
        macd = torch.Tensor(macd).unsqueeze(0)
        if norm:
          rsi = F.normalize(rsi)
          macd = F.normalize(macd)

        rsi_macd = torch.cat((
            rsi.unsqueeze(2), macd.unsqueeze(2)
        ), axis=2).squeeze(0)
        return rsi_macd
    
    def __len__(self):
        return len(self.rsi_macd) - self.predict_days

    def __getitem__(self, idx):
        rsi_macd = self.rsi_macd[idx: idx + self.predict_days]
        pct_change = self.pct_change[idx]
        return rsi_macd, pct_change


class MLP(nn.Module):
    def __init__(self, layers, input_size, output_size, dropout_rate=0, device=None):
        super(MLP, self).__init__()
        
        layers = [
            nn.Sequential(nn.Linear(input_size, input_size), nn.ReLU(), nn.Dropout(dropout_rate))
            for i in range(layers - 1)
        ]
        layers.append(nn.Linear(input_size, output_size))
        
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x):
        layer_ins = [x]
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            layer_ins.append(out)
        return out


class StockPredModel(nn.Module):
    def __init__(self, predict_days):
        super(StockPredModel, self).__init__()
        
        self.rsi_macd_reduce = nn.Linear(2, 1)
        self.mlp = MLP(3, predict_days, 1, .08)
        
    def forward(self, x):
        reduced_rsi_macd = self.rsi_macd_reduce(x)
        out = self.mlp(reduced_rsi_macd.squeeze(2))
        return out


def train(model, dataset, batch_size, loss_func, epochs, 
          learn_rate=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    train_iterator = tqdm.tqdm(range(epochs), position=0, leave=True)
    epoch_losses = []
    for epoch in train_iterator:
        losses = []
        for i, [rsi_macd, pct_change] in enumerate(dataloader):
            pred = model(rsi_macd)
            loss = loss_func(pred, pct_change)
            losses.append(loss.item())

            # Gradient descent
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            bs = batch_size
            avg_loss = epoch_losses[-1] if epoch_losses else -1
            loss_string = "Loss: avg-{:.3f} | {:.3f}"
            to_print = loss_string.format(avg_loss/bs, loss.data/bs)
            train_iterator.set_postfix_str(to_print)
        epoch_losses.append(sum(losses) / len(losses))
        '''
        experiment.log_parameter("epoch_losses", epoch_losses)
        experiment.log_parameter("losses", losses)
        experiment.log_parameter("avg_loss", avg_loss)
        experiment.log_parameter("pred", pred)
        experiment.log_metrics("epoch_losses", epoch_losses)
        experiment.log_metrics("losses", losses)
        experiment.log_metrics("avg_loss", avg_loss)
        experiment.log_metrics("pred", pred)
        '''
        
    model = model.eval()
    return model, epoch_losses



rsi = rsi_np
macd = macd_np
pct_change = pct_change_np
predict_days = 30
dataset = TrainingDataset(rsi, macd, pct_change, predict_days)
model = StockPredModel(predict_days)
model, epoch_losses = train(model, dataset, 60, MSELoss(), 100)


plt.plot(epoch_losses)

# Checking
test_rsi_macd = TrainingDataset.get_rsi_macd(rsi, macd, norm=True)[3:33]
#print(test_rsi_macd)
test_pct_change = torch.tensor(pct_change)[3:33]
pred = model(test_rsi_macd.unsqueeze(0))
print(pred, pct_change[30])
experiment.log_metric(pred, pct_change[30])

Experiment(log_env_details=True, log_env_gpu=True, log_env_cpu=True, log_env_host=True)
Experiment(auto_output_logging="default")

'''
model = keras.Sequential(
[
    layers.Dense(2, activation="relu", name="layer1"),
    layers.Dense(3, activation="relu", name="layer2"),
    layers.Dense(3, activation="relu", name="layer3"),
    layers.Dense(3, activation="relu", name="layer4"),
    layers.Dense(3, activation="relu", name="layer5"),
    layers.Dense(3, activation="relu", name="layer6"),
    layers.Dense(4, activation="softmax", name="layer7"),
]
)

model = keras.Sequential()
# Add an Embedding layer expecting input vocab of size 1000, and
# output embedding dimension of size 64.
model.add(layers.Embedding(input_dim=1000, output_dim=64))

# Add a LSTM layer with 128 internal units.
model.add(layers.LSTM(128))

# Add a Dense layer with 10 units.
model.add(layers.Dense(10))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy'])

'''
#def ml_training(self, rsi_macd_np, pct_change_np):


'''
class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, input_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim



        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, inputs):
        lstm_out, _ = self.lstm(torch.DoubleTensor(inputs).view(4358, 3).double())
        tag_space = self.hidden2tag(lstm_out.view(4358, 3, 1).double())
        return lstm_out


model = LSTMTagger(4538, 4358, 4358, 3).double()
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    tag_scores = model(rsi_macd_pct_np.type('torch.DoubleTensor'))
    print(tag_scores)

for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for rsi, macd, pct_change in rsi_macd_pct_np:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        

        # Step 3. Run our forward pass.
        tag_scores = model((rsi, macd, pct_change))

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, pct_change)
        loss.backward()
        optimizer.step()

# See what the scores are after training
with torch.no_grad():
    tag_scores = model(rsi_macd_pct_np.type('torch.DoubleTensor'))

    # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
    # for word i. The predicted tag is the maximum scoring tag.
    # Here, we can see the predicted sequence below is 0 1 2 0 1
    # since 0 is index of the maximum value of row 1,
    # 1 is the index of maximum value of row 2, etc.
    # Which is DET NOUN VERB DET NOUN, the correct sequence!
    print(tag_scores)

'''



'''
batch_size = 2048
# Each MNIST image batch is a tensor of shape (batch_size, 28, 28).
# Each input sequence will be of size (28, 28) (height is treated like time).
input_dim = 2
units = 64
output_size = 1  # labels are from 0 to 9
def build_model(allow_cudnn_kernel=True):
    # CuDNN is only available at the layer level, and not at the cell level.
    # This means `LSTM(units)` will use the CuDNN kernel,
    # while RNN(LSTMCell(units)) will run on non-CuDNN kernel.
    if allow_cudnn_kernel:
        # The LSTM layer with default options uses CuDNN.
        lstm_layer = keras.layers.LSTM(units, input_shape=(None, input_dim))
    else:
        # Wrapping a LSTMCell in a RNN layer will not use CuDNN.
        lstm_layer = keras.layers.RNN(
            keras.layers.LSTMCell(units), input_shape=(None, input_dim)
        )
    model = keras.models.Sequential(
        [
            lstm_layer,
            keras.layers.BatchNormalization(),
            keras.layers.Dense(output_size),
        ]
    )
    return model
X_train, X_test, y_train, y_test = train_test_split(rsi_macd_np, pct_change_np, test_size=0.33, random_state=42)
model = build_model()
opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(loss='sparse_categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy'])
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=2048,
    epochs=1000
)
'''
        

