"""
 * @author Krishna Karthik Gadiraju
 * @email 
 * @desc [LSTM Network for multi-temporal crop classification. Network structure described in [1]]

Reference:
Zhong, Liheng, Lina Hu, and Hang Zhou. "Deep learning based multi-temporal crop classification." Remote sensing of environment 221 (2019): 430-443.

"""
import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import numpy as np
np.random.seed(100)
tf.set_random_seed(100)
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=tf_config))
from tensorflow.python.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import Model
from sklearn.metrics import classification_report
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard
from time import time
import configparser, argparse, datetime, json, os
from data_generator import crop_generator 
from tensorflow.python.keras.utils import plot_model
import socket
import sys
import pandas as pd
import itertools

start_time_n = datetime.datetime.now()

start_time = start_time_n.strftime("%Y-%m-%d %H:%M:%S")


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--configPath', help="""path to config file""", default='./config.ini')
parser.add_argument('-t', '--task', help="""classification task you are performing - either crop, water or energy""", default='zhong-lstm')
parser_args = parser.parse_args()

config = configparser.ConfigParser()
config.read(parser_args.configPath)


batch_size = int(config[parser_args.task]['BATCH_SIZE'])
tr_path = str(config[parser_args.task]['TRAIN_FOLDER'])
val_path = str(config[parser_args.task]['VAL_FOLDER'])
num_classes = int(config[parser_args.task]['NUM_CLASSES'])
#learning_rate = float(config[parser_args.task]['LEARNING_RATE'])
learning_rates = [0.0001, 0.0005, 0.00001, 0.00005]
num_epochs = int(config[parser_args.task]['NUM_EPOCHS'])
#dropout_rate = float(config[parser_args.task]['DROPOUT_RATE']) 
dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
epsilon = float(config[parser_args.task]['EPSILON'])
plot_path = str(config['plotting']['PLOT_FOLDER'])
measures_folder = str(config['plotting']['MEASURE_FOLDER'])


grid_file = './{}_grid.csv'.format(config[parser_args.task])

if os.path.exists(grid_file):
    grid_df = pd.read_csv(grid_file)

else:
    param_grid = list(itertools.product(learning_rates, dropout_rates))
    grid_df = pd.DataFrame(param_grid, columns = ['lr', 'd'])
    grid_df['timestamp'] = 'None'

print(grid_df)

        
# parse through the generated grid and perform training separately
# record all the values (loss, accuracy) etc
# finally search through to find best result in separate file
# Also code is saved after finishing run for every pair of parameters to ensure that 
# if the server kills the job, you can restart it from previous location
# and not from starting

for idx, row in grid_df.iterrows():
    if row['timestamp'] == 'None':
        learning_rate = row['lr']
        dropout_rate = row['d']
        curr_time_n = datetime.datetime.now()
        curr_time = curr_time_n.strftime("%Y-%m-%d %H:%M:%S")
        time_passed_since_start = curr_time_n - start_time_n
        time_passed_since_start_in_min = time_passed_since_start/datetime.timedelta(minutes=1)
        time_passed_since_start_in_hours = time_passed_since_start_in_min/60
        grid_df.loc[idx, 'timestamp'] = curr_time


        # Custom data generator that reads the csv files, extracts the NDVI column, fixes missing values
        train_generator = crop_generator(input_path=tr_path, batch_size=batch_size, mode="train", do_shuffle=True, epsilon=0)

        val_generator = crop_generator(input_path=val_path, batch_size=batch_size, mode="val", do_shuffle=True, epsilon=0)

        # network starts here
        temporal_input_layer = Input(batch_shape = (None, 23, 1), name='time_input_layer')


        # add an LSTM layer - network structure based on network diagram shown in [1] 
        # Paper reference: [1] (see description at top)
        lstm_1 = LSTM(32, input_shape=(23,1), dropout=dropout_rate, return_sequences=True)(temporal_input_layer)

        lstm_2 = LSTM(32, dropout=dropout_rate, return_sequences=True)(lstm_1)

        lstm_3 = LSTM(32, dropout=dropout_rate, return_sequences=True)(lstm_2)

        lstm_4 = LSTM(32, dropout=dropout_rate)(lstm_3)

        fc_layer = Dense(64, activation='relu')(lstm_4)

        # and a softmax layer equal to num_classes
        predictions = Dense(num_classes, activation='softmax')(fc_layer)

        optimizer = Adam(lr=learning_rate)

        # finally create Model with correct input and output layers
        model = Model(inputs= temporal_input_layer, outputs=predictions)

        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy']) 

        # print(model.summary())


        ## NN ends here

        # callbacks 
        # tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
        model_name = str(config[parser_args.task]['MODEL_FOLDER']) + '_temporal' + curr_time + '.h5'
        model_checkpoint = ModelCheckpoint(model_name, verbose=1, monitor='val_loss',save_best_only=True, mode='auto')
        log_file_name = os.path.join(config['plotting']['MEASURE_FOLDER'], 'evals-{}.json'.format(curr_time))
        csv_logger = CSVLogger(filename=log_file_name, append = True)

        history = model.fit_generator(train_generator,  
			epochs=num_epochs,
			steps_per_epoch=36690//batch_size,
			validation_data= val_generator, 
			validation_steps=12227//batch_size,
			verbose=1, 
			callbacks=[model_checkpoint, csv_logger])

        with open('{}-config_{}.ini'.format(config[parser_args.task]['MODEL_FOLDER'], curr_time), 'w') as cfgfile:
            newConfig = configparser.ConfigParser() 
            newConfig[parser_args.task] = config[parser_args.task]
            #print(config[parser_args.task])
            newConfig.write(cfgfile)

        # write grid to disk during current iteration to ensure that temporary results are stored
        grid_df.to_csv(grid_file, index=False)

#print(f"Hostname: {socket.gethostname()}, Number of classes = {num_classes}\n Learning rate = {learning_rate}\n dropout rate = {dropout_rate}\n timestamp = {curr_time}") 
#print("Model is saved at: {}".format(model_name)) 
