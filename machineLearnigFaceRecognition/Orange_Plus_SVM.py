import csv
import numpy
from sklearn.model_selection import KFold
import sklearn.metrics as metrics
from sklearn.multiclass import OneVsRestClassifier
import sklearn.svm as svm


#############################################
def map_indices_to_x_set(indices_list):
    training_features_list = []
    for index in indices_list:
        training_features_list.append(X_train[index])

    return training_features_list


def map_indices_to_y_set(indices_list):
    testing_labels_list = []
    for index in indices_list:
        testing_labels_list.append(Y_labels[index])

    return testing_labels_list


def precision(class_label, confusion_matrix):
    col = confusion_matrix[:, class_label]
    return confusion_matrix[class_label, class_label] / col.sum()


def recall(class_label, confusion_matrix):
    row = confusion_matrix[class_label, :]
    return confusion_matrix[class_label, class_label] / row.sum()


def precision_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_precisions = 0
    for label in range(rows):
        sum_of_precisions += precision(label, confusion_matrix)
    return sum_of_precisions / rows


def recall_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_recalls = 0
    for label in range(columns):
        sum_of_recalls += recall(label, confusion_matrix)
    return sum_of_recalls / columns


def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements

#########################################
with open('features_vector.csv') as csv_file: #training data
    reader = csv.reader(csv_file) #read the file
    data = [r for r in reader] #loop on csv
    data.pop(0) #

_classes = {'ANGER': 0, 'CONTEMPT': 1, 'DISGUST': 2, 'FEAR': 3, 'HAPPY': 4, 'NEUTRAL': 5, 'SADNESS': 6, 'SURPRISE': 7} #dic
X_train = [] #features
Y_labels = [] #labels
for _row in data:
    _row[len(_row) - 1] = _classes[_row[len(_row) - 1]] #the last row - get the index
    Y_labels.append(_row.pop(len(_row) - 1))
    X_train.append(_row) #without classes -

accuracy_scores = 0
allfolds_y_test = []
allfolds_y_predicted = []
fold_number = 1

svm_classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, tol=1e-3)) #tollerance
kfolds_cross_validaton = KFold(n_splits=10, shuffle=True)
print('[INFO]: Performing 10-Folds Cross Validation.... ')
for train_index, test_index in kfolds_cross_validaton.split(X_train):
    print('[INFO]: Fold %d ' % fold_number)
    Xtrain_Cross_Val = map_indices_to_x_set(train_index)
    Xtest_Cross_Val  = map_indices_to_x_set(test_index)
    Ytrain_Cross_Val = map_indices_to_y_set(train_index)
    Ytest_Cross_Val  = map_indices_to_y_set(test_index)
    svm_classifier.fit(Xtrain_Cross_Val, Ytrain_Cross_Val) #fit model
    predictions = svm_classifier.predict(Xtest_Cross_Val) #
    allfolds_y_test.extend(Ytest_Cross_Val) #classess
    allfolds_y_predicted.extend(predictions)
    accuracy_scores += metrics.accuracy_score(Ytest_Cross_Val, predictions) ## how much the model was accurate
    fold_number += 1

classes_list = ['ANGER', 'CONTEMPT', 'DISGUST', 'FEAR', 'HAPPY', 'NEUTRAL', 'SADNESS', 'SURPRISE']
allfolds_confusion_matrix = metrics.confusion_matrix(allfolds_y_test, allfolds_y_predicted)
allfolds_confusion_matrix = numpy.asarray(allfolds_confusion_matrix)
model_accuracy = accuracy(allfolds_confusion_matrix)
model_precision = precision_macro_average(allfolds_confusion_matrix)
model_recall = recall_macro_average(allfolds_confusion_matrix)
print('[INFO]: Printing Cross-Validation Evaluation Metrics:')
print('[INFO]: Confusion Matrix:::')
print(metrics.classification_report(allfolds_y_test, allfolds_y_predicted, target_names=classes_list))
print('[INFO]: Accuracy = ', model_accuracy)
print('[INFO]: Precision = ', model_precision)
print('[INFO]: Recall = ', model_recall)

################################################################################
with open('test_set.csv') as csv_file:
    reader = csv.reader(csv_file)
    data = [r for r in reader]
    data.pop(0)

classes_dict = {0: 'ANGER', 1: 'CONTEMPT', 2: 'DISGUST', 3: 'FEAR', 4: 'HAPPY', 5: 'NEUTRAL', 6: 'SADNESS', 7: 'SURPRISE'}
for test_instance in data:
    test_instance = numpy.asarray(test_instance).reshape(-1, 1).transpose()
    prediction_probabilities = list(svm_classifier.predict_proba(test_instance)[0])
    max_probability = max(prediction_probabilities)
    class_index = prediction_probabilities.index(max_probability)
    for i in range(len(prediction_probabilities)):
        print('[%s]:%0.3f %%' % (classes_dict[i], prediction_probabilities[i] * 100))

    print('Prediction: %s' % classes_dict[svm_classifier.predict(test_instance)[0]])
    print('==========================================================')
