# MSc-DeepLearning
### 1. Insurance_Fraud:
Context - Using a dataset with unbalanced classes to train a neural network classifier that can identify insurance claims that are fraudulent. The dataset comprises of 85% of non-fraudulent claims and only 15% of fraudulent claims. Using an autoencoder approach to recreate the features, the aim is to train the neural network with the majority class so the reconstruction error is high for the minority class, and therefore, those high errors will pertain to the fraudulent insurance claims. 
##### i. Insurance_claims.csv:
The labelled dataset with the incident and claimant information (1=Fraud; 0=Not Fraud).
##### ii. Autoencoder-Insurance.ipynb:
Jupyter notebook with the code for the trained autoencoder, data cleaning steps, training steps, hyperparameter tuning, model evaluation, and predictions. The notebook also contains comments and analysis on the dataset along with the theoritcal explanation of key concepts. 
##### iii. ADL_Autoencoder_FinalModel.h5
The final model in h5 format. 

### 2. RNN-CNN_sensors:
Context - To predict, as accurately as possible, the operating mode of a wind turbine based on the time series data from two sensors. The data are sensor readings and operating modes for 4,000 turbine runs. Based on the paper: “Convolutional neural network fault classification based on time series analysis for benchmark wind turbine machine” by Rahimilarki, Gao, Jin, and Zhang, available at https://www.sciencedirect.com/science/article/abs/pii/S0960148121017778

##### i. time_series_1.pickle, time_series_2.pickle, y.pickle
time_series_1 and time_series_2 are NumPy arrays of shape (4000,5000). Each observation corresponds to 5,000 records of the turbine over time by one of the two sensors (time_series_1 measures the pitch angle in each second of operation, and time_series_2 measures the generator torque). y is the operating mode for each of the 4,000 turbine runs (0 if the turbine is healthy, 1 if the generator torque is faulty, 2 if the pitch angle is faulty, and 3 if both are faulty). Note that the dataset is balanced in that each operating mode is represented equally often.
##### ii. RNN-CNN_timeSeries.ipynb
The Jupyter notebook with detailed steps for feature transformation, hyperparameter tuning, trial and comparison of various networks. In particular, trying different iterations of Recurrent Neural Networks, such as SimpleRNN, LSTM and Conv1D. Then, comparing with the outcome of using a Convolutional Neural Network by converting time series into "images" (matrices of numbers). It includes comments on why one approach may be better than the other. 

Further, the CNN with three convolutional layers displayed in Figure 12 of the paper is replicated and attempts have been made to improve upon that, including using tranfer learning through pre-trained image recognition models from GoogleNet. 

Lastly, the notebook presents a comparison of models to choose the final one and train that on a combined training and validation set to make predictions on the test set.
##### iii. RNN-CNN_FinalModel.h5 
The final trained model.
