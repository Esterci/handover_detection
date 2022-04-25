import os
import glob
import numpy as np
from utils import ProgBar


# define the paths into the container
data_path = "data_divisions/*"

# Defining hyper-parameters range

min_batch_size = 100
max_batch_size = 500

min_hidden_dim = 20
max_hidden_dim = 200

min_time_step = 10
max_time_step = 50

# Parameters in study

batch_size_list = list(np.linspace(max_batch_size, min_batch_size, num=1, dtype=int))
encoding_dim_list = list(np.linspace(min_hidden_dim, max_hidden_dim, num=2, dtype=int))
time_step_list = list(np.linspace(min_time_step, max_time_step, num=12, dtype=int))


# create a list of config files
file_list = glob.glob(data_path)

bar = ProgBar(
    int(
        len(file_list)
        * len(time_step_list)
        * len(batch_size_list)
        * len(encoding_dim_list)
    ),
    "\nExecuting training and testing...",
)

for file in file_list:
    for time_step in time_step_list:
        for batch_size in batch_size_list:
            for encoding_dim in encoding_dim_list:

                it = file.split("__")[4]

                m_command = """python3 autoencoder.py -b {BACH} -e {EDIM} -t {TIME} -it {IT} -f {FILE}""".format(
                    BACH=batch_size, EDIM=encoding_dim, TIME=time_step, IT=it, FILE=file
                )

                # execute the tuning
                os.system(m_command)

                bar.update()

print("\n")
