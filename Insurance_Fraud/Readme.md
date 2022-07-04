## 1. Insurance_Fraud:

Context - Using a dataset with unbalanced classes to train a neural network classifier that can identify insurance claims that are fraudulent. The dataset comprises of 85% of non-fraudulent claims and only 15% of fraudulent claims. Using an autoencoder approach to recreate the features, the aim is to train the neural network with the majority class so the reconstruction error is high for the minority class, and therefore, those high errors will pertain to the fraudulent insurance claims.

#### i. Insurance_claims.csv:

The labelled dataset with the incident and claimant information (1=Fraud; 0=Not Fraud).

#### ii. Autoencoder-Insurance.ipynb:

Jupyter notebook with the code for the trained autoencoder, data cleaning steps, training steps, hyperparameter tuning, model evaluation, and predictions. The notebook also contains comments and analysis on the dataset along with the theoritcal explanation of key concepts.

#### iii. ADL_Autoencoder_FinalModel.h5

The final model in h5 format.

