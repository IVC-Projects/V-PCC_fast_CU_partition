# V-PCC_fast_CU_partition
A fast CU partition method dedicated to V-PCC to accelerate the coding process.

## Requirement
TMC2-18.0  
VTM-13.0  
Python 3.6  
Lightgbm 3.2  

## Lgbm model
We provide LightGBM models for nine CU sizes, including 32×32, 32×16, 32×8, 16×32, 16×16, 16×8, 8×32, 8×16, and 8×8. You can use these models directly for testing.
```C++
LGBM_model
```

## Usage
### Training Set
We selected four Dynamic Poind Clouds defined in V-PCC common test condition, including Longdress, Loot, RedAndBlack, and Soldier, and used their first five frames to create the training dataset. These DPC frames were encoded under the All Intra (AI) configuration at five bitrate points from the low bitrate (r1) to the high bitrate (r5), generating the desired features and labels.

### Creating training set labels
```C++
make_trainset\EncSlice.cpp
```
We added some code to the EncSlice.cpp file in the VTM. This file will output the partition mode of CU. You can use this EncSlice.cpp to replace EncSlice.cpp of VTM.

### Data preprocessing
There will be duplicate datas in the generated label datas. Use 
```Python
train_LGBM\statisticData_norepeat.py
```
to delete the duplicate data.

### Feature collection
Use 
```Python
train_LGBM\make_dataset.py
```
to extract the features corresponding to each label data

### LGBM training
Use 
```Python
train_LGBM\LightGBM.py
```
to train the LGBM model.

### LGBM test
Add the files that calls LGBM in the TMC2 project
```C++
VTM_calls_LGBM
```
Modify the model path in the file, run the TMC2 project, and get the test results. 

## Continually update
This repository's README file will be continuously updated.
