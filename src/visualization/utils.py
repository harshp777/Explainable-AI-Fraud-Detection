from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np

def model_eval_classification(model, x, y):
    '''This function takes a classification model, input and output data and returns a scoring dictionary.'''
    
    # Predicting output from the model
    y_pred = model.predict(x)
    
    # Accuracy
    accuracy = round(accuracy_score(y, y_pred), 2)
    
    # Precision
    precision = round(precision_score(y, y_pred), 2)
    
    # Recall
    recall = round(recall_score(y, y_pred), 2)
    
    # F1 Score
    f1 = round(f1_score(y, y_pred), 2)
    
    # ROC AUC Score (if applicable)
    roc_auc = None
    if len(np.unique(y)) > 1:  # ROC AUC is undefined for a single class
        roc_auc = round(roc_auc_score(y, y_pred), 2)
    
    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    
    # Dictionary storing all these testing scores
    score_dict = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall,
                  'F1 Score': f1, 'ROC AUC Score': roc_auc, 'Confusion Matrix': cm}
    
    return score_dict
