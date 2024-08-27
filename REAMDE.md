
## Install morpholayers
Install morpholayer: 
```
cd JMIV_cell_counting
git clone https://github.com/Jacobiano/morpholayers.git
```

# Experiments on yellow cells dataset 
## Step0: Download the dataset
Please refer to the [yellow cells repository](https://github.com/robomorelli/cell_counting_yellow) for the dataset download.
Unzip and put the downloaded dataset into the folder `/JMIV_cell_counting/data/fluocells`. The dataset folder should look like this:

```commandline
/JMIV_cell_counting/data/
├── fluocells
│   ├── all_images
│   ├── all_masks
│   └── README.md
...
```

## Step1: Split train/val/test
Run the following jupyter notebook to split the dataset into train-val-test:
```
JMIV_cell_counting/yellow_cells/001organise_yellow_cells_dataset.ipynb
```
The preprocessed dataset will be saved in the folder `/JMIV_cell_counting/data/yellow_cells`. The dataset folder should look like this:

```commandline
/JMIV_cell_counting/data/
├── fluocells
│   ├── all_images
│   ├── all_masks
│   └── README.md
└── fluocells_organised_with_zeros_official_testsplit
    ├── set1
    │   ├── images
    │   └── labels
    └── set2
        ├── images
        └── labels
```

## Step2: Preprocess the dataset 
Run the following command to train the model:
```
python /JMIV_cell_counting/yellow_cells/002generate_best_h_dataset_opening_closing.py
```
This will generate the best h dataset with opening-closing operations. 

<ins>Notes<ins>: For the cell counting only based methods, this step is only for put the opening-closing step offline and accelerate training, the best h generated will not be used for training.


## Step3: Training
Run the following command to train with only cell counting loss:
```
python /JMIV_cell_counting/yellow_cells/004main_only_cell_counting_loss.py --mode train  --Explicit_backpropagation_mode minus_one
```
<ins>Notes<ins>: --Explicit_backpropagation_mode minus_one use the modified backpropagation rule, denoted as Ours(MB,-1) in the paper. --Explicit_backpropagation_mode Puissance_N_2 use the backpropagation rule with exponential approximation with N=2, denoted as Ours(MB,N=2) in the paper.

## Step4: Test
Run the following command to test trained model:
```
python /JMIV_cell_counting/yellow_cells/004main_only_cell_counting_loss.py --mode test  --model_weight_path pretrained/model/weight.h5
```

<ins>Notes<ins>: The training and test with joint loss function can be found in 003main_joint_loss.py

# Experiments on CellPose (subset) dataset 
## Step0: Download the dataset
Please refer to the [CellPose dataset](https://www.cellpose.org/) for the dataset download.
Unzip and put the downloaded dataset into the folder `/JMIV_cell_counting/data/cellpose`. The dataset folder should look like this:

```commandline
/JMIV_cell_counting/data/cellpose/
├── CellPose
│   ├── train
│   ├── test
│   └── train_cyto2
...
```

## Step1: Select the subset
Run the following jupyter notebook to select the subset that comforme the hypothesis described in the paper:
```
JMIV_cell_counting/cellpose/000Select_cell_pose_subset.ipynb
```

The folder should now look like this:
```commandline
/JMIV_cell_counting/data/cellpose/
├── CellPose
│   ├── train
│   ├── test
│   └── train_cyto2
├── CellPose_selected_organised
│   ├── train
│   └── test
...
```

## Step2: Preprocess the dataset - 1
Run the following jupyter notebook to convert the RGB images into uni-channel images (taking the maximum of 3 channels as described in the paper)
```
/JMIV_cell_counting/cellpose/001Convert_cell_pose.ipynb
```

The folder should now look like this:
```commandline
/JMIV_cell_counting/data/cellpose/
├── CellPose
│   ├── train
│   ├── test
│   └── train_cyto2
├── CellPose_selected_organised
│   ├── train
│   └── test
├── CellPose_converted
│   ├── train
│   └── test
...
```

## Step3: Preprocess the dataset - 2
Run the following command to train the model:
```
python /JMIV_cell_counting/cellpose/002generate_best_h_dataset_opening_closing.py
```
This will generate the best h dataset with opening-closing operations. 

<ins>Notes<ins>: For the cell counting only based methods, this step is only for put the opening-closing step offline and accelerate training, the best h generated will not be used for training.

## Model training and test
The command lines are same as described in Experiments on yellow cells dataset 


# Experiments on Green cells (florecent melanocyte cells) dataset
## For the data download and preprocessing, please refer to the [DGMM cell countings repository](https://github.com/peter12398/DGMM2024-comptage-cellule).

## Training and test with only cell counting loss
The command lines are same as described in Experiments on yellow cells dataset

## Training and test with joint loss
## For the training and test with joint loss, please refer to the [DGMM cell countings repository](https://github.com/peter12398/DGMM2024-comptage-cellule).