from __future__ import division
from __future__ import print_function

import json
import pdb
import math

all_params = json.load(open('config.json'))
sample_rate = all_params["experiment_setup"]['sample_rate']
tcn_model_dir=all_params["JIGSAWS"]['tcn_model_dir']
tcn_model_params = all_params["JIGSAWS"]["tcn_params"]
input_size = all_params["JIGSAWS"]["input_size"]
num_class=all_params["JIGSAWS"]["gesture_class_num"]
dataset_name = all_params['dataset_name']
raw_feature_dir = all_params["JIGSAWS"]["raw_feature_dir"]
test_trial=all_params["JIGSAWS"]["test_trial"]
train_trial = all_params["JIGSAWS"]["train_trial"]
sample_rate = all_params["experiment_setup"]["sample_rate"]
gesture_class_num = all_params["JIGSAWS"]["gesture_class_num"]
# for parameter tuning
validation_trial = all_params["JIGSAWS"]["validation_trial"]
validation_trial_train = [2,3,4,5]