# MICCAI2022_model

## directories
```
model
 ┣ JIGSAWS
 ┃ ┣ tcn
 ┃ ┃ ┣ test_1
 ┃ ┃ ┃ ┣ log
 ┃ ┃ ┃ ┃ ┗ train_test_result.csv
 ┃ ┃ ┃ ┣ checkpoint_0.pth
 ┣ Knot_Tying
 ┃ ┣ kin_ges
 ┃ ┃ ┣ Knot_Tying_B001.txt ... [contains the kinematics columns +'Y' (categorical values of the gestures)]
 ┣ Needle_Passing
 ┃ ┣ kin_ges
 ┃ ┃ ┣ Needle_Passing_B001.txt...
 ┣ Suturing
 ┃ ┣ kin_ges
 ┃ ┃ ┣ Suturing_B001.txt...
 ┣ JIGSAWS-TRANSFORM.pkl
 ┣ calculate_mean_cv.ipynb
 ┣ config.json
 ┣ config.py
 ┣ data_loading.py
 ┣ dataprep_code.py
 ┣ get_festure.py
 ┣ logger.py
 ┣ lstm_model.py
 ┣ parameter_tuning.py (use this code to perform parameter tuning)
 ┣ requirements.txt
 ┣ tcn_model.py
 ┣ train_test_cross.py
 ┣ try.ipynb
 ┗ utils.py
 ```
## setting up the environment 
* install anaconda 
* run  conda install --file requirements.txt
## data preparation
* use/ modifiy the code in the dataprep_code to create the datasets under kin_ges. This code assigns each time step a corresponding gesture from the transcription dataset.
* change directories in the json file
```
        "input_size": 14,---feature size
        "gesture_class_num":14,---unique classes
        "split_num":8,# need to be removed
        "validation_trial":1, -- for parameter tuning train on [2,3,4,5], validation on trial 1
        "test_trial":[1,2,3,4,5],---for cross validation test set
        "train_trial":[[2,3,4,5],[1,3,4,5],[1,2,4,5],[1,2,3,5],[1,2,3,4]], ---for cross validation train set
        "raw_feature_dir":["/home/aurora/Documents/MICCAI2022_baseline/LSTM_model/Needle_Passing/kin_ges","/home/aurora/Documents/MICCAI2022_baseline/LSTM_model/Suturing/kin_ges","/home/aurora/Documents/MICCAI2022_baseline/LSTM_model/Knot_Tying/kin_ges"],
 "tcn_params":{
            "model_params":{
                "class_num":14, ---need to be changed to be same as gesture_class_num
        
```
