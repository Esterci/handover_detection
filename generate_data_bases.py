import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle as pk


dataset = pd.read_csv("classification_base.csv")

raw_data = dataset.values
# The last element contains if the transaction is normal which is represented by a 0 and if fraud then 1
labels = raw_data[:, -1]
# The other data points are the electrocadriogram data
data = raw_data[:, 0:-1]

data = data.T

sc=MinMaxScaler(feature_range=(0,1))

data = sc.fit_transform(data)

data = data.T

##########################################################
# ------------------------------------------------------ #
# --------------------- INITIATION --------------------- #
# ------------------------------------------------------ #
##########################################################

# Number of events
#total = 500000

# Percentage of background samples on the testing phase
#background_percent = 0.99

# Percentage of samples on the training phase
test_size = 0.3

# Number of iterations

n_it = 33


for it in range(n_it):

    train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=test_size, random_state=2021, stratify=labels )

    train_labels = train_labels.astype(bool)
    #creating normal and fraud datasets
    normal_train_data = train_data[~train_labels]

    print('\n      ==== Pre-processing Complete ====\n')
    print(".Train data shape: {}".format(train_data.shape))
    print(".Test data shape: {}".format(test_data.shape))

    print('=*='*17 )

    Output = {'train_data'  : train_data,
              'test_data'   : test_data,
              'test_labels' : test_labels}

    struct_name = ('data_base/data__test_size__' + str(test_size) +
                   '__n_it__' + str(it) + '__.pkl')
    
    with open(struct_name, 'wb') as f:
        pk.dump(Output, f)

