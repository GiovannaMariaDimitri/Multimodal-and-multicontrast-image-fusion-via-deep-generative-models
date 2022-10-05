#Example Script

from keras import regularizers
import pickle as pickle
from keras.callbacks import ModelCheckpoint, CSVLogger
import os
import sys
filepath = 'path_to_files'
sys.path.append(filepath)
from utils.preprocess import DataLoader
from utils.models import Parameters, Net
import pandas as pd

#Example dimension input images 
target_rows = 128
target_cols = 128
depth = 96
axis = 1

#Example hyperparameters settings 
drop_rate = 0.1
w_regularizer = regularizers.l2(5e-5)
batch_size = 1
params_dict = { 'w_regularizer': w_regularizer, 'batch_size': batch_size,
               'drop_rate': drop_rate, 'epochs': 200,
          'gpu': "/gpu:0", 'model_filepath': filepath,
          'image_shape': (target_rows, target_cols, depth, axis)}

params = Parameters(params_dict)

seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

#Load data
data_loader = DataLoader('path_to_file',(target_rows, target_cols, depth, axis), seed = seeds[0])
Mod_1 = data_loader.get_mri()
data_loader = DataLoader('path_to_file',(target_rows, target_cols, depth, axis), seed = seeds[0])
Mod_2 = data_loader.get_mri()

net = Net(params)


#Train the network
history = net.train([Mod_1,Mod_2])
#Predict 
reconstr1,reconstr2,embed = net.predict([Mod_1,Mod_2])

#Save a couple of reconstructions as an example
with open('./example.pickle', 'wb') as f:
    pickle.dump([reconstr1[0],reconstr2[0],embed], f, protocol = 2)


#Save History
hist_df = pd.DataFrame(history)
hist_csv_file = 'history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
