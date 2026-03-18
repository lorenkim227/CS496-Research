# Loren Kim
# ML Old Notes/Code to refer to when writing new code
# Purpose is to run 2 identical ensemble grid searches with accuracy and f1 scoring


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np

# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html


# read in data file, printed features with types, dropped ID, double checked each step
df = pd.read_csv("diabetes_binary.csv")
label="Diabetes_binary"
df #double check import was complete

print("Features: ", df.columns)
print("Data Types: ", df.dtypes)

df.isnull().sum() #double check no missing values
df = df.drop(["ID"], axis=1)
df


# split data after minimal preprocessing
X_train, X_test, y_train, y_test = train_test_split(df.drop([label],axis=1), df[label], 
                                                    test_size=0.30, stratify=df[label], random_state=1)


# tested various hyperparameters, this set had the highest overall accuracy
param_grid = {
    'n_estimators': [100, 500],
    'max_features': ['sqrt', 'log2', None],
    'oob_score': [True],
    'class_weight': [{0: 1, 1: 15}, {0: 1, 1: 3}, 'balanced_subsample', {0: 1, 1: 5}], #unbalanced data
    'max_depth': [10, 20, 50]
}


# purpose: run the grid search twice
# input: scoring name for f1 or accuracy, filename to save to
# output: returns the gridsearch, prints the classification report and saves both results as csv files
def run_grid_search(scoring_name, filename):
    rf = RandomForestClassifier(random_state=1, bootstrap=True, oob_score=True)

    gs = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring=scoring_name,
        refit=True,
        cv=5,
        n_jobs=-1,
        verbose=2
    )

    gs.fit(X_train, y_train)

    results_df = pd.DataFrame(gs.cv_results_)
    # changed to None
    results_df['param_class_weight'] = results_df['param_class_weight'].apply(lambda x: 'None' if x is None else x)
    results_df.to_csv(filename, index=False)

    print(f"\nClassification Report on Training Data ({scoring_name}):")
    y_train_pred = gs.predict(X_train)
    print(classification_report(y_train, y_train_pred))

    print(f"\nClassification Report on Testing Data ({scoring_name}):")
    y_test_pred = gs.predict(X_test)
    print(classification_report(y_test, y_test_pred))

    return gs


# run the grid search with accuracy and f1 scoring
gs_accuracy = run_grid_search("accuracy", "grid_search_accuracy.csv")
gs_f1 = run_grid_search("f1", "grid_search_f1.csv")  

# Loren Kim
# CS484 PA1
# Performs one vs all wine analysis with imported perceptron code

# CHANGES: reduced the white space, added clearer comments, added 2 new functions, made clear seaparation of training and testing data, the last function uses the best prediction of each class, changed column numbers to Alcohol and Proline, separated code into 3 "chunks": processing data, train data, test data with final outputs


#imports
from perceptron import Perceptron #imports book's code, which is in perceptron.py in this folder
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


# PREPROCESS DATA, SPLIT INTO LABELS/FEATURES, ASSIGN BINARY TO ONE V REST CLASSES
# import data, drop differently scaled features, and check updated data
df = pd.read_csv('wine.data', header=0, encoding='utf-8')
df = df.drop(['Alcohol', 'Proline'], axis=1)
df.head()
print(df)

# split into labels and features
X = df.iloc[1:, 1:].values
y = df.iloc[1:, 0].values

# purpose: divide labels
# parameters:
    # y: labels
    # class_label: class label
# return value: one class is 1 and other 2 classes are -1
def individual_labels(y, class_label):
    return np.where(y == class_label, 1, -1) 


# SPLIT INTO TRAIN/TEST, SCALE, AND TRAIN EACH MODEL WITH PPN WITH TRAINING DATA
# split into test and training data and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# purpose: train each model with perceptron
# parameters:
    # X_train: training features
    # y_train: training label
# return value: the model and the errors per epoch
def train_models(X_train, y_train):
    models = {}
    errors_per_epochs = {}
    accuracies = []
    for class_label in np.unique(y_train):  
        print("training class ", class_label)
        y_train_each = individual_labels(y_train, class_label)
        ppn = Perceptron(eta=0.1, n_iter=50, random_state=1)
        ppn.fit(X_train, y_train_each)  
        models[class_label] = ppn
        errors_per_epochs[class_label] = ppn.errors_
    return models, errors_per_epochs
models, errors_per_epochs = train_models(X_train_std, y_train) #train w scaled training data

# prints class with final weights and bias per model
for class_label, model in models.items():
    print("class:", class_label, ", final weights:", model.w_, ", final bias:", model.b_)


# PLOT EPOCH ERRORS, EVALUATE MODELS AND SEND TEST DATA FOR PREDICTIONS ALONG WITH FINAL REPORT
# purpose: plot errors per epoch
# parameters:
    # errors_per_epochs: the num of errors per epoch
# return value: nothing, plots graph
def plot_errors(errors_per_epochs):
    for class_label, errors in errors_per_epochs.items():
        epochs = range(1, len(errors) + 1)
        plt.plot(epochs, errors, label='Class'+str(class_label))
    plt.xlabel('Epochs')
    plt.ylabel('Number of errors')
    plt.legend()
    plt.title('Errors per Epoch for each Class')
    plt.tight_layout()
    plt.savefig('pa1_epocherr.png', dpi=300)
    plt.show()
plot_errors(errors_per_epochs)

# purpose: evaluates the models and gets the highest scoring model
# parameters:
    # X: features
    # models: array of models generated with ppn
# return value: array of predicted class for the best models
def predictions(X, models):
    y_pred = []
    for x in X:
        scores = {}
        for class_label, model in models.items():
            score = model.net_input(x)
            scores[class_label] = score
        predicted_class = max(scores, key=scores.get) #pick class with highest accuracy
        y_pred.append(predicted_class)
    return np.array(y_pred)

# test the best model using scaled test data on the highest scoring classes
y_pred = predictions(X_test_std, models)
print(f'Misclassified examples: {(y_test != y_pred).sum()}')
print(f'Accuracy: {metrics.accuracy_score(y_test, y_pred):.3f}')

# Display confusion matrix and classification report
cm = metrics.confusion_matrix(y_test, y_pred)
metrics.ConfusionMatrixDisplay(cm).plot()
plt.show()
print(metrics.classification_report(y_test, y_pred))


# loren kim
# cs484 PA2
# this file applies 2 models onto training data with pipelines, random grid search, and k-fold. The best hyperparameters are selected, then the best model is selected to make predictions on testing data.


# all my import statements
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score

# import the training and test data files
train_df = pd.read_csv('train_data.csv')
test_df = pd.read_csv('test_data.csv')

# used to double check
#train_df.head()
#train_df.shape

# assign train and test label and features
X_train = train_df.iloc[:, 1:].values 
y_train = train_df.iloc[:, 0].values 
X_test = test_df.iloc[:, 1:].values
y_test = test_df.iloc[:, 0].values

print("Check 1")



# LOGIC REGRESSION (LR) PIPELINE, GRIDSEARCH, AND K-FOLD CROSS VALIDATION

# took out PCA since already did that in preprocessing
pipe_lr = make_pipeline(StandardScaler(), LogisticRegression(random_state=1, max_iter=10000))

# create the hyperparameters for the grid search
param_grid_lr = {
    'logisticregression__C': [0.1, 1, 10, 50, 100, 1000, 10000],
    'logisticregression__solver': ['lbfgs', 'saga']
}

# I used a randomized grid search to speed up the process
rs_lr = RandomizedSearchCV(estimator=pipe_lr,
                        param_distributions=param_grid_lr,
                        scoring='f1_weighted', #used f1 weighted to deal with potential class imbalance
                        refit=True,
                        cv=5,
                        n_jobs=-1,
                        verbose=2)


print("Check 2")

# complete the grid search and fit to training data
#gs_lr = GridSearchCV(estimator=pipe_lr, param_grid=param_grid_lr, scoring='accuracy', refit=True, cv=2, n_jobs=-1, verbose=2)
#gs_lr.fit(X_train, y_train)

rs_lr = rs_lr.fit(X_train, y_train)


print("Check 3")

# k-fold cross validation
scores_lr = cross_val_score(estimator=rs_lr.best_estimator_, 
                            X=X_train, 
                            y=y_train, 
                            cv=5, 
                            n_jobs=-1)
print(f'CV f1 scores for LR: {scores_lr}')
print(f'CV f1 for LR: {np.mean(scores_lr):.3f} +/- {np.std(scores_lr):.3f}')



# print the final result
print("Best score for LR: ", rs_lr.best_score_)
print("Best parameters for LR: ", rs_lr.best_params_)






# kNN PIPELINE, GRIDSEARCH, AND K-FOLD CROSS VALIDATION

# kNN pipeline
pipe_knn = make_pipeline(StandardScaler(), KNeighborsClassifier())

# create the hyperparameters for the grid search
param_grid_knn = {
    'kneighborsclassifier__n_neighbors': [3, 5, 7, 9, 11],
    'kneighborsclassifier__weights': ['uniform', 'distance'],
    'kneighborsclassifier__p': [1, 2]
}


# complete the random grid search and fit to training data

rs_knn = RandomizedSearchCV(estimator=pipe_knn,
                        param_distributions=param_grid_knn,
                        scoring='f1_weighted',
                        refit=True,
                        cv=5,
                        n_jobs=-1,
                        verbose=2)



rs_knn = rs_knn.fit(X_train, y_train)

# k-fold cross validation
scores_knn = cross_val_score(estimator=rs_knn.best_estimator_, 
                             X=X_train, 
                             y=y_train, 
                             cv=5, 
                             n_jobs=-1)
print(f'CV f1 scores for kNN: {scores_knn}')
print(f'CV f1 for kNN: {np.mean(scores_knn):.3f} +/- {np.std(scores_knn):.3f}')


# print the final result
print("Best score for kNN: ", rs_knn.best_score_)
print("Best parameters for kNN: ", rs_knn.best_params_)







# FIND THE BEST MODEL

# find best model from grid search
if rs_lr.best_score_ > rs_knn.best_score_:
    best_model = rs_lr.best_estimator_
    print("The best model is logistic regression")
    print("Best score: ", rs_lr.best_score_, " with parameters: ", rs_lr.best_params_)
else:
    best_model = rs_knn.best_estimator_
    print("The best model is kNN")
    print("Best score: ", rs_knn.best_score_, "with parameters: ", rs_knn.best_params_)





# send test data to best model
predictions = best_model.predict(X_test)



# print accuracy score for best model
test_f1 = f1_score(y_test, predictions, average='weighted')
print(f'Test F1-score: {test_f1:.3f}')



# print the classification report for best model
print("Classification report:")
print(classification_report(y_test, predictions))





## OUTPUT


print("EXPECTED OUTPUT\n")

print("CV f1 scores for LR: [0.6523649 0.65195628 0.66390847 0.65890285 0.65685974]")
print("CV f1 for LR: 0.657 +/- 0.004")
print("Best score for LR: 0.6168675309222499")
print("Best parameters for LR: {'logisticregression__solver': 'lbfgs','logisticregression__C': 100}\n")

print("CV f1 scores for kNN: [0.70569006 0.70201246 0.70640515 0.70712024, 0.71641639]")
print("CV accuracy for kNN: 0.708 +/- 0.005")
print("Best score for kNN: 0.6960316069592516")
print("Best parameters for kNN: {'kneighborsclassifier__weights': 'distance','kneighborsclassifier__p': 1, 'kneighborsclassifier__n_neighbors': 11}\n")

print("The best model is kNN")
print("Best score: 0.6960316069592516 with parameters:")
print("{'kneighborsclassifier__weights': 'distance', 'kneighborsclassifier__p': 1,'kneighborsclassifier__n_neighbors': 11}\n")

print("Test F1-score: 0.706\n")

print("Classification report:")
print("precision recall f1-score support")
print("1 0.75 0.82 0.78 12196")
print("2 0.67 0.66 0.67 6747")
print("3 0.56 0.28 0.37 2034")
print("accuracy 0.72 20977")
print("macro avg 0.66 0.59 0.61 20977")
print("weighted avg 0.71 0.72 0.71 20977")
