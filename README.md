#Crowd Counting

#Files
This project contains the following files and folders:
1) preprocessing.py - generates density maps from images
2) new_pretraining.py - performs pretraining for individual regressors
3) differential.py - performs differential training
4) new_coupled_training.py - performs coupled training on switch and regressors
5) main.py - calls the other training algos
6) hyperparameters - stores hyperparameters used
7) data - contains images and ground truth annotations

#How to Run

Run main.py which runs the following pipeline:
1) Preprocessing by generating the density maps
2) Pretraining
3) Differential Training
4) Coupled Training

Weights are stored as checkpoints in the folders

Please remember to install all requirements before running this. The entire pipeline can take a long, long time to run :)
