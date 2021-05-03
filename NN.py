import os
import sys
import uproot
import awkward
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

#Read root file with uproot
file = uproot.open('/ceph/ktauqeer/TTSemiLeptonic/2016/RecoNtuples/uhh2.AnalysisModuleRunner.MC.TTToSemiLeptonic_2016v3_0_new.root')

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

#Split the dataset into training, validation and testing set with 70% train and 30% val and test size
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X, Y, test_size=0.3)

#Split the val_test dataset into half(15% each) for having seperate validation and testing sets
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

#see the shape of train, test and validation dataset
#print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)


#define architecture of the Neural Network using keras
model = Sequential([
    Dense(32, activation='relu', input_shape=(4,)),
    Dense(32, activation='relu'),
    Dense(2, activation='sigmoid'),
])

#configuring the model
model.compile(optimizer='sgd',
              loss='hinge',
              metrics=['accuracy'])

#Summarize the model
model.summary()
#tf.keras.utils.plot_model(model,'model.pdf',show_shapes=True)

#Saving training history
hist = model.fit(X_train, Y_train, batch_size=32, epochs=100, validation_data=(X_val, Y_val))

#Evaluating model (do this when you are satisfied with your model hyperparamters and training)
model.evaluate(X_test, Y_test)[1] #0 returns model loss and 1 returns model accuracy

#Predict labels for test set
Y_test_predicted = model.predict_classes(X_test)
#Y_test_predicted = Y_test_predicted.round()

print(Y_test)
print(Y_test_predicted)

#some visulaization of the model loss and accuracy with matplotlib
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.savefig('LossCurve.pdf')

plt.close()

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.savefig('AccuracyCurve.pdf')

plt.close()
