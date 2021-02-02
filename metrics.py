import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0
    #print('prediction', prediction)
    #print('ground_truth', ground_truth)
    ##print('sss', prediction == ground_truth)
    #print('sum', (prediction == ground_truth).sum())
    accuracy = (prediction == ground_truth).sum() / prediction.size 
    
    TP = 0
    TN = 0 
    FP = 0
    FN = 0
    
    #print('precision[i] == ground_truth[i]', prediction[0] == ground_truth[0])
    
    for i in range(len(ground_truth)):
        if prediction[i] == ground_truth[i] and ground_truth[i] == True:
            TP += 1
        elif prediction[i] == ground_truth[i] and ground_truth[i] == False:
            TN += 1
        elif ground_truth[i] == False and prediction[i] == True:
            FP += 1
        elif ground_truth[i] == True and prediction[i] == False:
            FN += 1
      
    #print('TP', TP, 'TN', TN, 'FP', FP, 'FN', FN)
    if (TP + FP) != 0:
        precision = TP / (TP + FP)
    else:
        precision = 0
    
    if (TP + FN) != 0:
        recall = TP / (TP + FN)
    else:
        recall = 0 
    
    if (precision + recall) != 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0 
    
    
    #TP = (prediction == ground_truth & prediction==True)
    #print('TP', TP)

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification
    
    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    accuracy = (prediction == ground_truth).sum() / prediction.size 
    # TODO: Implement computing accuracy
    return accuracy
