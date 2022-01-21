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
## 
