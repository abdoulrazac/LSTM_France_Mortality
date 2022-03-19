# -*- coding: utf-8 -*-
"""
@author: Abdoul Razac SANE

"""


# Import des librairies 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
from numpy.random import seed
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error , mean_absolute_error
from itertools import product
from tqdm import tqdm

# Google Drive
try :
    from google.colab import drive
    drive.mount('/content/drive')
    Path = "/content/drive/MyDrive/Colab Notebooks/SARIMA_vs_LSTM/"
except :
    Path = ""

# Import des données
with open(Path + "Mortalite.txt", mode = "r") as f :
    tmp = []
    for line in f.readlines() :
        tmp += line.split(" ")[1:]
    tmp = tmp[19:]

data = pd.DataFrame({
        "month" : pd.date_range(start='1995-12-01', end='2015-4-30', freq='M').values,
        "valeur" : [int(i) for i in tmp]
    })
del tmp, line
data.set_index('month', inplace=True)
data = data[:-1]
data.head()

# Transformation en vecteur numeric
dataset = data.values
dataset = data.astype('float32')

# Création des données d'entrainement et de test
train_size = int(len(dataset) - 24)
train, test = dataset[:train_size], dataset[train_size:]
print(f"Train_data_size: {len(train)}\nTest_data_size {len(test)}")

# Normalisation des données avec la méthode MinMax
scaler = MinMaxScaler(feature_range=(0, 1)).fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

# Fonction pour transformer un vecteur en matrice de données X et Y
def transform_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
	    a = dataset[i:(i+look_back), 0]
	    dataX.append(a)
	    dataY.append(dataset[i + look_back, 0])
        
    dataX = np.array(dataX)
    
    # Transformation des X en [samples, time_steps, features]
    dataX = np.reshape(dataX, (dataX.shape[0], dataX.shape[1], 1))
    dataY = np.array(dataY)
    return dataX, dataY

# Fonction de création des entrées et sortie
def create_dataset(dataset, look_back=1, train_size = train_size) :
    dataset = scaler.transform(dataset)
    train, test = dataset[:train_size], dataset[train_size-look_back:]
    trainX, trainY = transform_dataset(train, look_back)
    testX, testY = transform_dataset(test, look_back)
    return trainX, trainY, testX, testY

# Fonction d'entrainement du modèle
def make_model(dataX, dataY, look_back=2, nb_noeux = 10,  epochs=200, batch_size=1) :
    seed(1)
    tf.random.set_seed(1)
    model = Sequential()
    model.add(LSTM(nb_noeux, activation='relu', input_shape=(dataX.shape[1], dataX.shape[2])))
    #model.add(LSTM(16, activation='relu', return_sequences=False))
    #model.add(Dropout(0.1))
    model.add(Dense(1))    
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(dataX, dataY, epochs=epochs, batch_size=batch_size, verbose=0)
    return model

# Fonction pour calculer le RMSE et le MAE
def model_performance(Y_obs, Y_pred, log_print = False) : 
    RMSE = np.sqrt(mean_squared_error(Y_obs, Y_pred))
    MAE  = mean_absolute_error(Y_obs, Y_pred)    
    if log_print :
        print('RMSE : %.2f \nMAE : %.2f' % (RMSE, MAE))
    return RMSE, MAE

# Fonction pour évaluer plusieurs modèles
def evaluate_models(train, test, look_backs = [2], nb_noeux = [10],  epochs = [100], batch_size = [1]) :
    resultats = {
            'look_back' : [],
            'nb_noeux' : [],
            'epochs' : [],
            'batch_size' : [],
            'Train_RMSE' : [],
            'Train_MAE' : [],
            'Test_RMSE' : [],
            'Test_MAE' : []
        }

    for params in tqdm(product(look_backs, nb_noeux,  epochs, batch_size)) :
        
        look_back = int(params[0])
        trainX, trainY, testX, testY = create_dataset(dataset, look_back)
    
        model = make_model(trainX, trainY, look_back=look_back, nb_noeux=params[1], epochs=params[2], batch_size=params[3])
        
        train_out   = scaler.inverse_transform(model.predict(trainX))
        trainY_out  = scaler.inverse_transform([trainY])
        
        test_out   = scaler.inverse_transform(model.predict(testX))
        testY_out  = scaler.inverse_transform([testY])
    
        trainScore = model_performance(trainY_out[0], train_out[:,0])
        testScore = model_performance(testY_out[0], test_out[:,0])
        
        resultats['look_back'].append(params[0])
        resultats['nb_noeux'].append(params[1])
        resultats['epochs'].append(params[2])
        resultats['batch_size'].append(params[3])
        resultats['Train_RMSE'].append(trainScore[0])
        resultats['Train_MAE'].append(trainScore[1])
        resultats['Test_RMSE'].append(testScore[0])
        resultats['Test_MAE'].append(testScore[1])
        
    return pd.DataFrame(resultats)

# Choix du meilleur modèle
resultats = evaluate_models(train, test,
            look_backs = [4, 8, 12, 16], 
            nb_noeux = [10, 20, 60, 100, 150],
            epochs = [200], 
            batch_size = [1]
        )

# resultats par RMSE
resultats[['look_back', 'nb_noeux', 'epochs', 'Train_RMSE', 'Train_MAE', 'Test_RMSE', 
           'Test_MAE']].sort_values(by = ['Test_RMSE'], ascending=True).head(5).round(2)

# resultats part MAE
resultats[['look_back', 'nb_noeux', 'epochs', 'Train_RMSE', 'Train_MAE', 'Test_RMSE', 
           'Test_MAE']].sort_values(by = ['Test_MAE'], ascending=True).head(5).round(2)

# indice du meilleur modèle
best_config = 9

## Meilleur config
look_back, nb_noeux, epochs = resultats.sort_values(by = ['Test_MAE'], ascending=True)[['look_back', 'nb_noeux', 'epochs']].iloc[best_config]
print(f"look_back : {look_back}\nnb_noeux : {nb_noeux}\nepochs : {epochs}")

# Implementation du meilleur modèle
trainX, trainY, testX, testY = create_dataset(dataset, look_back)

# Entrainement
model = make_model(trainX, trainY, look_back=look_back, nb_noeux=nb_noeux, epochs=epochs)

# Faire les predictions
trainPredict = model.predict(trainX)
testPredict  = model.predict(testX)

# Inverser de la normalisation précédante sur les predictions
train_out  = scaler.inverse_transform(trainPredict)
trainY_out = scaler.inverse_transform([trainY])
test_out   = scaler.inverse_transform(testPredict)
testY_out  = scaler.inverse_transform([testY])

# calculate de nouveau les performence
print("Données d'entrainement")
trainScore = model_performance(trainY_out[0], train_out[:,0], log_print=True)

print("\nDonnées test")
testScore = model_performance(testY_out[0], test_out[:,0], log_print=True)

# Représentation graphique ====================================================
# Données d'entrainement pour graphique
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_out)+look_back, :] = train_out

# # Données d'entrainement pour graphique
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_out)+(look_back)+1:len(dataset)-1, :] = test_out

# Graphique
plt.figure(figsize = (10, 6), dpi=100)
sns.lineplot(x=data.index, y = data['valeur'], label="Oberservé").set(
    #title = 'Evoulotion du nombre de décès en France entre 1995 et 2015',
    xlabel = "Temps",
    ylabel = "Nombre de décès"
)
sns.lineplot(x=data.index[:], y = [float(i) for i in trainPredictPlot][:], label="Entrainement")
sns.lineplot(x=data.index[:], y = [float(i) for i in testPredictPlot][:], label="Test")
plt.legend()
plt.show()

# Fin =============================================================================
