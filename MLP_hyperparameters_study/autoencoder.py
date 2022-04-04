from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense, Input, BatchNormalization, Dropout, LeakyReLU
from keras.models import Model
from keras.losses import binary_crossentropy
from datetime import datetime
import argparse
import pickle as pk

def get_block(L, size):
    L = BatchNormalization()(L)

    L = Dense(size)(L)
    L = Dropout(0.5)(L)
    L = LeakyReLU(0.2)(L)
    return L


def time_stamp():
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y-%H.%M.%S")
    return timestampStr



parser = argparse.ArgumentParser(description = '', add_help = False)
parser = argparse.ArgumentParser()


parser.add_argument('-b','--batch_size', action='store',
        dest='batch_size', required = True,
            help = "The job config file that will be used to configure the job (sort and init).")

parser.add_argument('-e','--encoding_dim', action='store',
        dest='encoding_dim', required = False, default = None,
            help = "The volume output.")

parser.add_argument('-f','--file', action='store',
        dest='file', required = False, default = None,
            help = "The volume output.")

args = parser.parse_args()

batch_size = int(args.batch_size)
encoding_dim = int(args.encoding_dim)
file = args.file

print(file.split('__')[4])

it = file.split('__')[4]

print(it)

s = file

with open(s, 'rb') as f:
    data_dict = pk.load(f)

train_data = data_dict['train_data']
test_labels = data_dict['test_labels']
test_data = data_dict['test_data']

# Fixed parameters

nb_epoch = 15
input_dim = train_data.shape[1]
hidden_dim_1 = int(encoding_dim / 2)
hidden_dim_2 = int(hidden_dim_1 / 2)
learning_rate = 0.001   

##### Creates structure name #####

struct_name = (
    'batch_size' + '__' + str(batch_size) + '__' +
    'encoding_dim' + '__' + str(encoding_dim) + '__' +
    'it' + '__' + it + '__' + time_stamp()
)

###### Creates NN structure #####

# Setup network
# make inputs

#input Layer
input_layer = tf.keras.layers.Input(shape=(input_dim, ))
#Encoder
encoder = tf.keras.layers.Dense(encoding_dim, activation="tanh",                                activity_regularizer=tf.keras.regularizers.l2(learning_rate))(input_layer)
encoder=tf.keras.layers.Dropout(0.2)(encoder)
encoder = tf.keras.layers.Dense(hidden_dim_1, activation='relu')(encoder)
encoder = tf.keras.layers.Dense(hidden_dim_2, activation=tf.nn.leaky_relu)(encoder)
# Decoder
decoder = tf.keras.layers.Dense(hidden_dim_1, activation='relu')(encoder)
decoder=tf.keras.layers.Dropout(0.2)(decoder)
decoder = tf.keras.layers.Dense(encoding_dim, activation='relu')(decoder)
decoder = tf.keras.layers.Dense(input_dim, activation='tanh')(decoder)
#Autoencoder
autoencoder = tf.keras.Model(inputs=input_layer, outputs=decoder)

#Defining early stop

cp = tf.keras.callbacks.ModelCheckpoint(filepath="autoencoder_fraud.h5",
                               mode='min', monitor='val_loss', verbose=2, save_best_only=True)
# define our early stopping
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0.0001,
    patience=10,
    verbose=1, 
    mode='min',
    restore_best_weights=True
)

# Compiling NN

opt = Adam(lr=learning_rate)

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
                    verbose=1,
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

mse = np.mean(np.power(test_data - test_x_predictions, 2), axis=1)

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
