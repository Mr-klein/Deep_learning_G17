# Group 17
###### A breakthrough in fingerspelling-to-text translation using CNN classification and Ngram modelling
This repository contains all the files created and used by group 17. With these files a CNN classifier can be trained and optimized, and an Ngram model can use this classifier to predict words.

## CNN
This folder contains all files used for training the classifier. All code is written in Python and can be found in de `code` folder. The files `network_trainer.py` and `network_loader.py` train and load the optimal found network, respectively. The `network_weights.pth` file are the weights used for loading the optimal found network. The files `network_optimizer_trainer.py` and `network_optimizer_loader.py` are used to optimize the network and to load these networks, respectively. The code creates the files found in the folders `loss` and `weights`. The folder `results` is generated by the loader. All files written in `Matlab` are used for data visualizing the data distribution, the loss, and the optimizer results.

## NGRAM
This folder contains the files used for Training and testing the Ngram predictor. All code is written in Python and can be found in the `code` folder. The files `Preprocess_trainset.py` and `preprocess_testset.py` preprocess the text datasets used for training and testing. The Ngrams are calculated using the file `ngramModelTrainer.py`. Calculated Ngrams are stored in the `savedNgram` folder. The file `Predictor.py` can be used to test the Ngram model by itself. The file `total_model.py` combines the classifier with the calculated Ngrams. The accuracy of the model is then tested using the processed version of the google1000 dataset in the `processed_data` folder.

Run order:

Not strictly required since preprocessing has already been done
1. `preprocess_trainset.py`
2. `preprocess_testset.py`
3. `ngramModelTrainer.py`

Testing model

4. `total_model.py`

Instructions for running for each file are found as comments at the start of the respective file.
