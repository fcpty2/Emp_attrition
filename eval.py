from sklearn.metrics import confusion_matrix, classification_report

class Report:
    def classification_report(self, y_true, y_pred):
        print(classification_report(y_true, y_pred), "\n")
    
    def confusion_matrix(self, y_true, y_pred):
        print(confusion_matrix(y_true, y_pred))
