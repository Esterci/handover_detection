import os
import glob
import numpy as np

# define the paths into the container
data_path = "../data_base/*"

# Defining hyper-parameters range

min_batch_size = 100

max_batch_size = 500

min_hidden_dim = 8

max_hidden_dim = 10

# Parameters in study

batch_size_list = list(np.linspace(max_batch_size, min_batch_size, num=10, dtype=int))
encoding_dim_list = list(np.linspace(min_hidden_dim, max_hidden_dim, num=3, dtype=int))


# create a list of config files
file_list = glob.glob(data_path)

for file in file_list:
    for batch_size in batch_size_list:
        for encoding_dim in encoding_dim_list:

            m_command = """python3 autoencoder.py -b {BACH} \\
            -e {EDIM} \\
            -f {FILE}""".format(
                BACH=batch_size, EDIM=encoding_dim, FILE=file
            )

            print(m_command)
            # execute the tuning
            os.system(m_command)
