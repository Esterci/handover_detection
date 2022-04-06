from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
import pandas as pd
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from datetime import datetime
import argparse
import pickle as pk
from tensorflow.keras.layers import LSTM, Dropout, RepeatVector,TimeDistributed, Dense
import glob


def d3_d2(mat):
    import numpy as np
    mat  = np.array(mat)
    n,m,l = mat.shape
    return mat.reshape(n,m*l)


parser = argparse.ArgumentParser(description = '', add_help = False)
parser = argparse.ArgumentParser()


parser.add_argument('-b','--batch_size', action='store',
        dest='batch_size', required = True,
            help = "Batch size.")

parser.add_argument('-e','--encoding_dim', action='store',
        dest='encoding_dim', required = False, default = None,
            help = "Number o neurons in the first layer.")

parser.add_argument('-t','--time_step', action='store',
        dest='time_step', required = False, default = None,
            help = "Number of time instances to use.")

parser.add_argument('-f','--file', action='store',
        dest='file', required = False, default = None,
            help = "The subset file.")

args = parser.parse_args()

batch_size = int(args.batch_size)
encoding_dim = int(args.encoding_dim)
timesteps = int(args.time_step)
file = args.file

it = file.split('__')[4]

##### Creates structure name #####

struct_name = (
    'batch_size' + '__' + str(batch_size) + '__' +
    'encoding_dim' + '__' + str(encoding_dim) + '__' +
    'it' + '__' + it
)

results_list = glob.glob('Results/*')

results_list = [result.split('/')[-1].split('.')[0] for result in results_list]

if not ('results__' + struct_name) in results_list:

    with open(file, 'rb') as f:
        data_dict = pk.load(f)

    train_data = data_dict['train_data'][:,-timesteps:]
    test_labels = data_dict['test_labels']
    test_data = data_dict['test_data'][:,-timesteps:]

    train_data = train_data.reshape(
        train_data.shape[0],
        train_data.shape[1],
        1
    )

    test_data = test_data.reshape(
        test_data.shape[0],
        test_data.shape[1],
        1
    )

    # Fixed parameters

    nb_epoch = 4
    n_features = 1
    hidden_dim_1 = int(encoding_dim / 2) #
    learning_rate = 0.001   

    ###### Creates NN structure #####

    # Setup network
    # make inputs

    autoencoder = tf.keras.Sequential()
    autoencoder.add(LSTM(encoding_dim, activation='sigmoid', input_shape=(timesteps,n_features), return_sequences=True))
    autoencoder.add(Dropout(0.2))
    autoencoder.add(LSTM(hidden_dim_1, activation='sigmoid', return_sequences=False))
    autoencoder.add(RepeatVector(timesteps))
    autoencoder.add(LSTM(hidden_dim_1, activation='sigmoid', return_sequences=True))
    autoencoder.add(Dropout(0.2))
    autoencoder.add(LSTM(encoding_dim, activation='sigmoid', return_sequences=True))
    autoencoder.add(TimeDistributed(Dense(n_features)))
    autoencoder.compile(optimizer='adam', loss='mse')

    #Defining early stop

    cp = tf.keras.callbacks.ModelCheckpoint(filepath="autoencoder_fraud.h5",
                                mode='min', monitor='val_loss', verbose=0, save_best_only=True)
    # define our early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,
        patience=10,
        verbose=0, 
        mode='min',
        restore_best_weights=True
    )

    # Compiling NN

    opt = Adam(learning_rate=learning_rate)

    autoencoder.compile(metrics=['accuracy'],
                        loss='mean_squared_error',
                        optimizer=opt)                

    # Training
    #try:

    history = autoencoder.fit(train_data, train_data,
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(test_data, test_data),
                        verbose=0,
                        callbacks=[cp, early_stop]
                        ).history

    # Ploting Model Loss

    fig, ax = plt.subplots()
    plt.plot(history['loss'], linewidth=2, label='Train')
    plt.plot(history['val_loss'], linewidth=2, label='Validation')
    plt.legend(loc='upper right')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    fig.savefig('Figures/model-loss__' + struct_name + '__.png', 
                bbox_inches='tight'
            )

    # Predicting Test values

    #start = datetime.now()

    test_x_predictions = autoencoder.predict(test_data)

    #end = datetime.now()

    # Calculating MSE
    mse = np.mean(np.power(d3_d2(test_data) - d3_d2(test_x_predictions), 2), axis=1)

    error_df = pd.DataFrame({'Reconstruction_error': mse,
                            'True_class': test_labels})

    # Covnert pandas data-frame to array

    results = error_df.values

    # Saving Results

    #np.save('Results/results__' + struct_name + '__ ', results)

    with open('Results/results__' + struct_name + '.pkl', 'wb') as f:
        pk.dump(results,f)

    # Clear everything for the next init

    K.clear_session()        
