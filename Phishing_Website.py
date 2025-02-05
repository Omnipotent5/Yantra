#importing basic packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data = pd.read_csv("DataFiles\\5.urldata.csv")

# Sepratating & assigning features and target columns to X & y
y = data['Label']
X = data.drop('Label',axis=1)
X.shape, y.shape

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 12)
print(X_train.shape, X_test.shape)

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras.optimizers import Adam

# If there is a 'URL' or 'Domain' column, drop or extract features from it
if 'URL' in X_train.columns:
    X_train = X_train.drop('URL', axis=1)
    X_test = X_test.drop('URL', axis=1)

if 'Domain' in X_train.columns:
    X_train = X_train.drop('Domain', axis=1)
    X_test = X_test.drop('Domain', axis=1)

# List to store model results
results = []

# Decision Tree
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
y_train_tree = tree.predict(X_train)
y_test_tree = tree.predict(X_test)
acc_train_dt = accuracy_score(y_train, y_train_tree)
acc_test_dt = accuracy_score(y_test, y_test_tree)
results.append(['Decision Tree', acc_train_dt, acc_test_dt])

# Random Forest
forest = RandomForestClassifier()
forest.fit(X_train, y_train)
y_train_forest = forest.predict(X_train)
y_test_forest = forest.predict(X_test)
acc_train_rf = accuracy_score(y_train, y_train_forest)
acc_test_rf = accuracy_score(y_test, y_test_forest)
results.append(['Random Forest', acc_train_rf, acc_test_rf])

# Multilayer Perceptrons
mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 100), alpha=0.001)
mlp.fit(X_train, y_train)
y_train_mlp = mlp.predict(X_train)
y_test_mlp = mlp.predict(X_test)
acc_train_mlp = accuracy_score(y_train, y_train_mlp)
acc_test_mlp = accuracy_score(y_test, y_test_mlp)
results.append(['Multilayer Perceptrons', acc_train_mlp, acc_test_mlp])

# XGBoost
xgb = XGBClassifier(learning_rate=0.4, max_depth=7)
xgb.fit(X_train, y_train)
y_train_xgb = xgb.predict(X_train)
y_test_xgb = xgb.predict(X_test)
acc_train_xgb = accuracy_score(y_train, y_train_xgb)
acc_test_xgb = accuracy_score(y_test, y_test_xgb)
results.append(['XGBoost', acc_train_xgb, acc_test_xgb])

# Support Vector Machine
svm = SVC()
svm.fit(X_train, y_train)
y_train_svm = svm.predict(X_train)
y_test_svm = svm.predict(X_test)
acc_train_svm = accuracy_score(y_train, y_train_svm)
acc_test_svm = accuracy_score(y_test, y_test_svm)
results.append(['SVM', acc_train_svm, acc_test_svm])

# Autoencoder
input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim,))
encoder = Dense(input_dim, activation="relu",
                activity_regularizer=regularizers.l1(10e-4))(input_layer)
encoder = Dense(int(input_dim / 2), activation="relu")(encoder)
encoder = Dense(int(input_dim / 4), activation="relu")(encoder)
decoder = Dense(int(input_dim / 4), activation='relu')(encoder)
decoder = Dense(int(input_dim / 2), activation='relu')(decoder)
decoder = Dense(input_dim, activation='sigmoid')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, shuffle=True, validation_data=(X_test, X_test), verbose=1)
train_loss, acc_train_auto = autoencoder.evaluate(X_train, X_train, verbose=0)
test_loss, acc_test_auto = autoencoder.evaluate(X_test, X_test, verbose=0)
results.append(['AutoEncoder', acc_train_auto, acc_test_auto])

# Create DataFrame and sort results
results_df = pd.DataFrame(results, columns=['ML Model', 'Train Accuracy', 'Test Accuracy'])
sorted_results_df = results_df.sort_values(by=['Test Accuracy', 'Train Accuracy'], ascending=False)

print(sorted_results_df)

import pickle

with open('models\\decision_tree_model.pkl', 'wb') as file:
    pickle.dump(tree, file)

with open('models\\random_forest_model.pkl', 'wb') as file:
    pickle.dump(forest, file)

with open('models\\mlp_model.pkl', 'wb') as file:
    pickle.dump(mlp, file)

with open('models\\xgboost_model.pkl', 'wb') as file:
    pickle.dump(xgb, file)

with open('models\\autoencoder_model.pkl', 'wb') as file:
    pickle.dump(autoencoder, file)

with open('models\\svm_model.pkl', 'wb') as file:
    pickle.dump(svm, file)