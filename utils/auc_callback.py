from sklearn.metrics import roc_curve, auc
import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""
The right thing to do is to run predictions on all 
of your test data at the end of an epoch, 
then run the sklearn function on your predictions, 
and display the result. 
You can do this in a callback.
"""

class AucHistory(keras.callbacks.Callback):
    def __init__(self, train_data, train_labels, validation_data, validation_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.validation_data = validation_data
        self.validation_labels = validation_labels
    
    def on_train_begin(self, logs):
        self.auc_scores_train = []
        self.auc_scores_validation = []
        
    def on_train_end(self, logs):
        #create image of the results
        plt.figure()
        plt.plot(range(len(self.auc_scores_train)), self.auc_scores_train, label='training auc')
        plt.plot(range(len(self.auc_scores_validation)), self.auc_scores_validation, label='validation auc')
        plt.ylim([0.0, 1.05])
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.title('Areas under the receiver operating characteristic curves')
        plt.legend()
        
        plt.savefig('auc_scores')
    
    def on_epoch_end(self, epoch, logs):
        train_predictions = self.model.predict(self.train_data)
        val_predictions = self.model.predict(self.validation_data)
        self.auc_scores_train.append(self.compute_auc(self.train_labels, train_predictions))
        self.auc_scores_validation.append(self.compute_auc(self.validation_labels, val_predictions))
        
    def compute_auc(self, y_true, y_score):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        return roc_auc

