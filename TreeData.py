import numpy as np

def compute_confusion_matrix(actual, predicted):    
    actualVec = actual.ravel()
    predictedVec = predicted.ravel()
    
    TP = sum(np.logical_and(actualVec == True, predictedVec == True))
    FN = sum(np.logical_and(actualVec == True, predictedVec == False))
    FP = sum(np.logical_and(actualVec == False, predictedVec == True))
    TF = sum(np.logical_and(actualVec == False, predictedVec == False))
    
    confusionMatrix = np.array([[TP, FN], \
                       [FP, TF]])
             
    stats = compute_stats(confusionMatrix)
    return confusionMatrix, stats

def compute_stats(confusionMatrix):
    TP = confusionMatrix[0][0]
    FN = confusionMatrix[0][1]
    FP = confusionMatrix[1][0]
    TN = confusionMatrix[1][1]
    stats = {'accuracy' : 0.0, 'recall' : 0.0, 'precision' : 0.0}
    
    if(TP + FN + FP + TN != 0):
        accuracy = float(TP + TN)/(TP + FN + FP + TN)
        stats['accuracy'] = float(format(accuracy, '.3f'))
    if(TP + FN != 0):
        recall = float(TP)/(TP + FN)
        stats['recall'] = float(format(recall, '.3f'))
    if(TP + FP != 0):
        precision = float(TP)/(TP + FP)
        stats['precision'] = float(format(precision, '.3f'))
    
    return stats