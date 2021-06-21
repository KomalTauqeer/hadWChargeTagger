import os
import sys
import uproot
import awkward
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

#Read root file with uproot
file = uproot.open('/ceph/ktauqeer/TTSemiLeptonic/2016/RecoNtuples/uhh2.AnalysisModuleRunner.MC.TTToSemiLeptonic_2016v3.root')

#Access the tree
AnalysisTree = file['AnalysisTree']

#Access branches as NumPy arrays
#branches = AnalysisTree.arrays(["fatjet_charge_k0.5", "fatjet_charge_subjets", "fatjet_subjet1_charge", "fatjet_subjet2_charge", "charge_lep"],namedecode="utf-8")

#Instead of saving data in NumPY arrays from branches you can also convert them to pandas dataframe via build in uproot function TTree.pandas.df. I did this because I am more comfortable with pandas
data = AnalysisTree.pandas.df(["fatjet_charge_k0.5", "fatjet_charge_subjets_k0.5", "fatjet_subjet1_charge_k0.5", "fatjet_subjet2_charge_k0.5","charge_lep"],namedecode="utf-8")

#visualize your data
#print (data)

#Returns a NumPy value of the pandas dataframe 
dataset = data.values

#input features
X = dataset[:,0:4]

#labels to predict
Y = dataset[:,4]

#print(X)
#print(Y)

#Transform labels from {-1,1} to {0,1}
lb = preprocessing.LabelBinarizer()
Y = lb.fit_transform(Y)
#print("transformed labels")
#print(Y)

#Split the dataset into training, validation and testing set with 70% train and 30% val and test size
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X, Y, test_size=0.3)

#Split the val_test dataset into half(15% each) for having seperate validation and testing sets
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

#see the shape of train, test and validation dataset
#print("train and test dataset")
#print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)

#Load model
model = keras.models.load_model('NNmodel')

#Predict labels for test set
Y_test_predicted = model.predict(X_test)
#Y_test_predicted = Y_test_predicted.round()

print(Y_test_predicted)

print("true labels")
print(Y_test)

#Plot DNN output
plt.hist(Y_test_predicted[Y_test[:]<0.5],30,histtype='step',color='red',label='$\mathrm{W^+}$')
plt.hist(Y_test_predicted[Y_test[:]>0.5],30,histtype='step',color='blue',label='$\mathrm{W^-}$')
plt.legend(loc='upper right')
plt.ylabel('Events')
plt.xlabel('DNN score')
plt.savefig('DNNScore.pdf')
plt.close()

