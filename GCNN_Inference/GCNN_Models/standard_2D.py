import numpy as np


def Standardize_Train_Min_Max(train_input):
    train_max=np.amax(train_input)
    train_min=np.amin(train_input)
    train_labels=(train_input-train_min)/(train_max-train_min)
    return train_labels

def Standardize_Test_Min_Max(train_input, test_input):
    train_max=np.amax(train_input)
    train_min=np.amin(train_input)
    test_labels=(test_input-train_min)/(train_max-train_min)
    return test_labels

def Standardize_Min_Max_2D(train_input_1, train_input_2, test_input_1, test_input_2):
    train_labels_1 = Standardize_Train_Min_Max(train_input_1)
    train_labels_2 = Standardize_Train_Min_Max(train_input_2)
    test_labels_1  = Standardize_Test_Min_Max(train_input_1, test_input_1)
    test_labels_2  = Standardize_Test_Min_Max(train_input_2, test_input_2)
    train_labels   = np.zeros((len(train_labels_1),2))
    test_labels    = np.zeros((len(test_labels_1),2))
    for i in range(len(train_labels)):
        train_labels[i]=np.array((train_labels_1[i], train_labels_2[i]))
    for i in range(len(test_labels)):
        test_labels[i]=np.array((test_labels_1[i], test_labels_2[i]))
    return train_labels, test_labels

def Rescale_Min_Max_2D(prediction, train_input):
    train_max = np.amax(train_input)
    train_min = np.amin(train_input)
    estimated_labels = prediction*(train_max-train_min)+train_min
    return estimated_labels

def Rescale_MSE_Min_Max_2D(prediction, train_input):
    train_max = np.amax(train_input)
    train_min = np.amin(train_input)
    estimated_labels = prediction*(train_max-train_min)
    return estimated_labels
    