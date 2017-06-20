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
    SAVE_TRAIN_INTERVAL=10
    SAVE_VALIDATION_INTERVAL= 3
    
    def __init__(self, train_data, train_labels, validation_data, validation_labels, output_graph_name ='auc_scores'):
        self.train_data = train_data
        self.train_labels = train_labels
        self.validation_data = validation_data
        self.validation_labels = validation_labels
        self.output_graph_name = output_graph_name
    
    def on_train_begin(self, logs):
        self.auc_scores_train = []
        self.auc_scores_validation = []
        self.best_auc_score_validation = -1
        self.best_model = None
        
    def on_train_end(self, logs):
        # Save the model that scored the best validation AUC
        self.best_model.save('best_model.hdf5')

        #create image of the results
        plt.figure()
        fig, ax = plt.subplots()
        
        train_epochs, train_aucs = zip(*self.auc_scores_train)
        ax.plot(train_epochs, train_aucs, label='training auc')
        
        val_epochs, val_aucs = zip(*self.auc_scores_validation)
        ax.plot(val_epochs, val_aucs, label='validation auc')
        
        plt.ylim([0.0, 1.05])
        
        # show vertical line for the best epoch
        best_epoch_number = val_epochs[val_aucs.index(max(val_aucs))]
        plt.axvline(x=best_epoch_number)
        
        #grey horizontal lines to make changes more clear
        plt.minorticks_on()
        plt.grid(b=True, which='major', axis='y', color='0.60', linestyle='solid')
        plt.grid(b=True, which='minor', axis='y', color='0.80', linestyle='solid')
        
        # Set ticks for y axis
        locator = MultipleLocator(0.1)
        minor_locator = AutoMinorLocator(5)
        ax.yaxis.set_major_locator(locator)
        ax.yaxis.set_minor_locator(minor_locator)
        
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.title('Areas under the receiver operating characteristic curves')
        plt.legend(loc='center left', bbox_to_anchor=(0.1, 0.2))
        
        plt.savefig(self.output_graph_name + '.svg', dpi='figure')
        plt.savefig(self.output_graph_name, dpi='figure')
    
    def on_epoch_end(self, epoch, logs):
        if epoch % self.SAVE_TRAIN_INTERVAL == 0:
            train_predictions = self.model.predict(self.train_data)
            auc = self.compute_auc(self.train_labels, train_predictions)
            self.auc_scores_train.append((epoch, auc))
            
        if epoch % self.SAVE_VALIDATION_INTERVAL == 0:
            val_predictions = self.model.predict(self.validation_data)
            auc = self.compute_auc(self.validation_labels, val_predictions)
            self.auc_scores_validation.append((epoch, auc))
            if auc > self.best_auc_score_validation:
                print('New best validation score:\n{} > {}'.format(auc, self.best_auc_score_validation))
                self.best_auc_score_validation = auc
                self.best_model = self.model
                self.best_model.save('best_model_{}.hdf5'.format(epoch))
        
    def compute_auc(self, y_true, y_score):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        return roc_auc

