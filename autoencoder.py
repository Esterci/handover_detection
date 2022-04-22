import numpy as np
import pandas as pd
import tensorflow as tf

tf.get_logger().setLevel("INFO")
tf.autograph.set_verbosity(0)

from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import argparse
import glob
import pickle as pk

import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam


def d3_d2(mat):
    import numpy as np

    mat = np.array(mat)
    n, m, l = mat.shape
    return mat.reshape(n, m * l)


def argsDictionary(args):

    var_dict = vars(args)

    structure_name = []

    for var in var_dict:
        if not var == "file":
            hp_n_values = str(var) + "-" * 2 + str(var_dict[var])
            structure_name.append(hp_n_values)

    structure_name = ("_" * 2).join(structure_name)

    return var_dict, structure_name


# defining hyper-parameters constructor

parser = argparse.ArgumentParser(description="", add_help=False)
parser = argparse.ArgumentParser()

parser.add_argument(
    "-b",
    "--batch_size",
    type=int,
    action="store",
    dest="batch_size",
    required=True,
    help="Batch size.",
)

parser.add_argument(
    "-e",
    "--encoding_dim",
    type=int,
    action="store",
    dest="encoding_dim",
    required=True,
    default=None,
    help="Number o neurons in the first layer.",
)

parser.add_argument(
    "-t",
    "--time_step",
    type=int,
    action="store",
    dest="time_step",
    required=True,
    default=None,
    help="Number of time instances to use.",
)

parser.add_argument(
    "-it",
    "--it",
    type=int,
    action="store",
    dest="it",
    required=True,
    default=None,
    help="Number of iteration.",
)

parser.add_argument(
    "-f",
    "--file",
    type=str,
    action="store",
    dest="file",
    required=True,
    default=None,
    help="The subset file.",
)

args = parser.parse_args()

##### Creates structure name #####

args, struct_name = argsDictionary(args)

results_list = glob.glob("Results/*")

results_list = [result.split("/")[-1].split(".")[0] for result in results_list]

if not (struct_name) in results_list:
    with open(args["file"], "rb") as f:
        data_dict = pk.load(f)

    train_data = data_dict["train_data"][:, -args["time_step"] :]
    test_labels = data_dict["test_labels"]
    test_data = data_dict["test_data"][:, -args["time_step"] :]

    train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], 1)

    test_data = test_data.reshape(test_data.shape[0], test_data.shape[1], 1)

    # Fixed parameters

    nb_epoch = 4
    n_features = 1
    hidden_dim_1 = int(args["encoding_dim"] / 2)  #
    learning_rate = 0.001

    ###### Creates NN structure #####

    # Setup network
    # make inputs

    autoencoder = tf.keras.Sequential()
    autoencoder.add(
        LSTM(
            args["encoding_dim"],
            activation="sigmoid",
            input_shape=(args["time_step"], n_features),
            return_sequences=True,
        )
    )
    autoencoder.add(Dropout(0.2))
    autoencoder.add(LSTM(hidden_dim_1, activation="sigmoid", return_sequences=False))
    autoencoder.add(RepeatVector(args["time_step"]))
    autoencoder.add(LSTM(hidden_dim_1, activation="sigmoid", return_sequences=True))
    autoencoder.add(Dropout(0.2))
    autoencoder.add(
        LSTM(args["encoding_dim"], activation="sigmoid", return_sequences=True)
    )
    autoencoder.add(TimeDistributed(Dense(n_features)))
    autoencoder.compile(optimizer="adam", loss="mse")

    # Defining early stop

    cp = tf.keras.callbacks.ModelCheckpoint(
        filepath="autoencoder_fraud.h5",
        mode="min",
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
    )
    # define our early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.0001,
        patience=10,
        verbose=0,
        mode="min",
        restore_best_weights=True,
    )

    # Compiling NN

    opt = Adam(learning_rate=learning_rate)

    autoencoder.compile(metrics=["accuracy"], loss="mean_squared_error", optimizer=opt)

    # Training
    # try:

    history = autoencoder.fit(
        train_data,
        train_data,
        epochs=nb_epoch,
        batch_size=args["batch_size"],
        shuffle=True,
        validation_data=(test_data, test_data),
        verbose=0,
        callbacks=[cp, early_stop],
    ).history

    # Ploting Model Loss

    fig, ax = plt.subplots()
    plt.plot(history["loss"], linewidth=2, label="Train")
    plt.plot(history["val_loss"], linewidth=2, label="Validation")
    plt.legend(loc="upper right")
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")

    fig.savefig("Figures/model_loss/" + struct_name + "__.png", bbox_inches="tight")

    # Predicting Test values

    # start = datetime.now()

    test_x_predictions = autoencoder.predict(test_data)

    # end = datetime.now()

    # Calculating MSE
    mse = np.mean(np.power(d3_d2(test_data) - d3_d2(test_x_predictions), 2), axis=1)

    error_df = pd.DataFrame({"Reconstruction_error": mse, "True_class": test_labels})

    # Covnert pandas data-frame to array

    results = error_df.values

    # Saving Results

    # np.save('Results/results__' + struct_name + '__ ', results)

    with open("Results/" + struct_name + ".pkl", "wb") as f:
        pk.dump(results, f)

    # Clear everything for the next init

    K.clear_session()
