from sklearn.metrics import roc_curve, auc
import keras

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

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
        self.epoch_count = 0
    
    def on_train_begin(self, logs):
        self.auc_scores_train = []
        self.auc_scores_validation = []
        
    def on_train_end(self, logs):
        #create image of the results
        plt.figure()
        fig, ax = plt.subplots()
        
        ax.plot(range(len(self.auc_scores_train)), self.auc_scores_train, label='training auc')
        ax.plot(range(len(self.auc_scores_validation)), self.auc_scores_validation, label='validation auc')
        plt.ylim([0.0, 1.05])
        
        #grey horizontal lines to make changes more clear
        plt.minorticks_on()
        plt.grid(b=True, which='major', axis='y', color='0.60', linestyle='solid')
        plt.grid(b=True, which='minor', axis='y', color='0.80', linestyle='solid')
        
        # Set ticks for y axis
        locator = MultipleLocator(0.1)
        minor_locator = AutoMinorLocator(5)
        ax.yaxis.set_major_locator(locator)
        ax.yaxis.set_minor_locator(minor_locator)
        
        # show vertical line for the best epoch
        best_epoch_number = self.auc_scores_validation.index(max(self.auc_scores_validation))
        plt.axvline(x=best_epoch_number)
        
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.title('Areas under the receiver operating characteristic curves')
        plt.legend(loc='center left', bbox_to_anchor=(0.1, 0.2))
        
        plt.savefig('auc_scores.svg', dpi='figure')
    
    def on_epoch_end(self, epoch, logs):
        self.epoch_count += 1
        
        if self.epoch_count % 2 == 0:
            train_predictions = self.model.predict(self.train_data)
            val_predictions = self.model.predict(self.validation_data)
            self.auc_scores_train.append(self.compute_auc(self.train_labels, train_predictions))
            self.auc_scores_validation.append(self.compute_auc(self.validation_labels, val_predictions))
        
    def compute_auc(self, y_true, y_score):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        return roc_auc

