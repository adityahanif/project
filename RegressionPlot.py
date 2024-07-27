import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class Plot:
    def __init__(self):
        pass
    
    @staticmethod
    def KDEplot(y_test, y_test_pred, title):
        sns.kdeplot(y_test, color='green', label="Actual")
        sns.kdeplot(y_test_pred, color='red', label="Prediction")
        plt.title(f"KDE Plot: Prediction vs Actual {title}")
        plt.xlabel(title)
        plt.legend();
        
    @staticmethod    
    def ActPredPlot(y_train, y_test, y_train_pred, y_test_pred, title):
        sns.scatterplot(x=y_train, y=y_train_pred, label="Train")
        sns.scatterplot(x=y_test, y=y_test_pred, label="Test")
        plt.title(f"Actual vs Predicted: {title}")
        plt.xlabel("Actual")
        plt.ylabel("Predicted");
    
    @staticmethod
    def ResidualPlot(y_train, y_test, y_train_pred, y_test_pred, title):
        resid_train = y_train - y_train_pred
        resid_test = y_test - y_test_pred
        sns.scatterplot(x=y_train_pred, y=resid_train, label="Train")
        sns.scatterplot(x=y_test_pred, y=resid_test, label="Test")
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title(f"Residual Plot: {title}")
        plt.xlabel("Residual")
        plt.ylabel("Y Predict");
    
    @staticmethod
    def ModelScore(y_test, y_test_pred):
        print(f"MSE Score: {mean_squared_error(y_test, y_test_pred)}")
        print(f"MAE Score: {mean_absolute_error(y_test, y_test_pred)}")
        print(f"R2 Score: {r2_score(y_test, y_test_pred)}")