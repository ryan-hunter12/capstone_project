import csv
import pickle
import os
import numpy as np
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from re import IGNORECASE
from flask import Flask, request, jsonify
from scipy.stats import f_oneway
from sklearn import metrics
from sklearn import tree
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from IPython.display import Image
import pydotplus
from xgboost import XGBRegressor
from xgboost import plot_tree
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
warnings.filterwarnings('ignore')

data = pd.read_csv("Laptop_price.csv")
print("Succcessfully loaded data.")
data.info()
mean_accuracy = {"Linear Regression":[], "Decision Tree Regression": [], "Random Forest Regression": [], "Adaboost Decision Tree Regression": [], 
                 "XGB Regression": [], "KNeighbors Regression": [], "Support Vector Machine Regression": []}

print("The dataframe has", data.shape[0], "rows and", data.shape[1], "columns.")

data.isnull().sum()
duplicates = data.duplicated().sum()
print("The data contains", duplicates, "duplicates")
data = data.drop_duplicates()
print("Shape after deleting duplicates:", data.shape)
print("Printing Sample Data for Analysis:/n", data.head(10))

data.sort_values(["Price"], axis = 0, ascending=True, inplace=True)
print("The data has been sorted by price in ascending order.")
print("The First 10 Entries")
print(data.head(10))
print("The Last 10 Entries")
print(data.tail(10))
print("A description of the data, displaying number of unique values and statistical properties of each column")
print(data.describe(include='all'))
print("Printing only the number of unique values for each column")
print(data.nunique())

data['Price'].hist()
plt.show()

def PlotBarCharts(inpData, colsToPlot):
    fig, subPlot=plt.subplots(nrows=1, ncols=len(colsToPlot), figsize=(10,5))
    fig.suptitle('Bar charts of: '+ str(colsToPlot))
    for colName, plotNumber in zip(colsToPlot, range(len(colsToPlot))):
        inpData.groupby(colName).size().plot(kind='bar',ax=subPlot[plotNumber])
    plt.show()

PlotBarCharts(inpData=data, colsToPlot=['RAM_Size','Storage_Capacity'])
data.hist(['Processor_Speed', 'Screen_Size', 'Weight'], figsize=(15,8))
plt.show()

ContinuousCols=['Processor_Speed', 'Screen_Size', 'Weight']
ContinuousCols2=['Price','Processor_Speed', 'Screen_Size', 'Weight']
CategoricalCols = ['RAM_Size', 'Storage_Capacity']

for predictor in ContinuousCols:
    data.plot.scatter(x=predictor, y='Price', figsize=(10,5), title=predictor+" VS "+ 'Price')
    plt.show()

correlation_continous = data[ContinuousCols2].corr()
print(correlation_continous['Price'][abs(correlation_continous['Price']) > 0.05 ])



fig, PlotCanvas=plt.subplots(nrows=1, ncols=len(CategoricalCols), figsize=(18,5))
for PredictorCol , i in zip(CategoricalCols, range(len(CategoricalCols))):
    data.boxplot(column='Price', by=PredictorCol, figsize=(5,5), vert=True, ax=PlotCanvas[i])
plt.show()

def FunctionAnova(inpData, TargetVariable, CategoricalPredictorList):
    SelectedPredictors=[]
    print('##### ANOVA Results ##### \n')
    for predictor in CategoricalPredictorList:
        CategoryGroupLists=inpData.groupby(predictor)[TargetVariable].apply(list)
        AnovaResults = f_oneway(*CategoryGroupLists)
        if (AnovaResults[1] < 0.1):
            print(predictor, 'is correlated with', TargetVariable, '| P-Value:', AnovaResults[1])
            SelectedPredictors.append(predictor)
        else:
            print(predictor, 'is NOT correlated with', TargetVariable, '| P-Value:', AnovaResults[1])
    return(SelectedPredictors)
CategoricalPredictorList=['RAM_Size', 'Storage_Capacity']
FunctionAnova(inpData=data, TargetVariable='Price', CategoricalPredictorList=CategoricalPredictorList)
print("#--------------------------------------------------------------------- Machine Learning ---------------------------------------------------------------------#")

SelectedColums = ['Processor_Speed', 'RAM_Size', 'Storage_Capacity']
model_data = data[SelectedColums]
model_data.to_pickle('model_data.pkl')

numeric_data = pd.get_dummies(model_data)
numeric_data['Price'] = data['Price']

target_var = 'Price'
predictors = ['Processor_Speed', 'RAM_Size', 'Storage_Capacity']
X = numeric_data[predictors].values
Y = numeric_data[target_var].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=428)
PredictorScaler = StandardScaler()
PredictorScalerfit = PredictorScaler.fit(X)
X = PredictorScalerfit.transform(X)

print("---------------- Model Validation and Accuracy Calculations----------------")
print("Linear Regression")
reg_model = LinearRegression()
LREG = reg_model.fit(X_train, Y_train)
prediction = LREG.predict(X_test)
print('R2 Value:',metrics.r2_score(Y_train, LREG.predict(X_train)), ": This value is the goodness of fit.")

testing_results = pd.DataFrame(data=X_test, columns=predictors)
testing_results[target_var] = Y_test
testing_results[("Predicted" + target_var)] = np.round(prediction)
testing_results['APE'] = 100 *((abs(testing_results['Price'] - testing_results['PredictedPrice']))/testing_results['Price'])
MAPE = np.mean(testing_results['APE'])
median_MAPE = np.median(testing_results['APE'])

accuracy = 100- MAPE
median_accuracy = 100 - median_MAPE
print("Mean Accuracy on testing data:", accuracy)
print("Median accuracy on testing data:", median_accuracy)

def accuracy_score(orig, pred):
    MAPE = np.mean(100 * (np.abs(orig-pred)/orig))
    print('#########',"Accuracy:", 100-MAPE)
    return (100-MAPE)

custom_scoring = make_scorer(accuracy_score, greater_is_better=True)
accuracy_values = cross_val_score(reg_model, X, Y, cv=10, scoring=custom_scoring)
print("Accuracy values for 10-fold Cross Validation:", accuracy_values)
print("Final Average Accuracy of model:", round(accuracy_values.mean(),2))
mean_accuracy["Linear Regression"].append(round(accuracy_values.mean(), 2))

reg_model = DecisionTreeRegressor(max_depth=5, criterion='friedman_mse')

DT = reg_model.fit(X_train, Y_train)
prediction = DT.predict(X_test)
print('R2 Value:',metrics.r2_score(Y_train, DT.predict(X_train)), ": Goodness of fit")


feature_importance = pd.Series(DT.feature_importances_, index= predictors)
feature_importance.nlargest(10).plot(kind='barh')
plt.show()

print("---------------- Model Validation and Accuracy Calculations----------------")
print("Decision Tree Regression")
testing_results=pd.DataFrame(data=X_test, columns=predictors)
testing_results[target_var]=Y_test
testing_results[('Predicted'+target_var)]=np.round(prediction)

testing_results['APE'] = 100 *((abs(testing_results['Price'] - testing_results['PredictedPrice']))/testing_results['Price'])
MAPE = np.mean(testing_results['APE'])
median_MAPE = np.median(testing_results['APE'])

accuracy = 100- MAPE
median_accuracy = 100 - median_MAPE
print("Mean Accuracy on testing data:", accuracy)
print("Median accuracy on testing data:", median_accuracy)

def accuracy_score(orig, pred):
    MAPE = np.mean(100 * (np.abs(orig-pred)/orig))
    print('#########',"Accuracy:", 100-MAPE)
    return (100-MAPE)

custom_scoring = make_scorer(accuracy_score, greater_is_better=True)
accuracy_values = cross_val_score(reg_model, X, Y, cv=10, scoring=custom_scoring)
print("Accuracy values for 10-fold Cross Validation:", accuracy_values)
print("Final Average Accuracy of model:", round(accuracy_values.mean(),2))
mean_accuracy["Decision Tree Regression"].append(round(accuracy_values.mean(), 2))

dot_data = tree.export_graphviz(reg_model, out_file=None, feature_names=predictors, class_names=["Price"])
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png('tree.png')

reg_model = RandomForestRegressor(max_depth=5, n_estimators = 500, criterion='friedman_mse')

RF = reg_model.fit(X_train, Y_train)
prediction = RF.predict(X_test)
print('R2 Value:',metrics.r2_score(Y_train, RF.predict(X_train)))

feature_importance = pd.Series(RF.feature_importances_, index=predictors)
feature_importance.nlargest(10).plot(kind='barh')
# plt.show()

print("---------------- Model Validation and Accuracy Calculations----------------")
print("Random Forest Regression")
testing_results=pd.DataFrame(data=X_test, columns=predictors)
testing_results[target_var]=Y_test
testing_results[('Predicted'+target_var)]=np.round(prediction)

testing_results['APE'] = 100 *((abs(testing_results['Price'] - testing_results['PredictedPrice']))/testing_results['Price'])
MAPE = np.mean(testing_results['APE'])
median_MAPE = np.median(testing_results['APE'])

accuracy = 100- MAPE
median_accuracy = 100 - median_MAPE
print("Mean Accuracy on testing data:", accuracy)
print("Median accuracy on testing data:", median_accuracy)

def accuracy_score(orig, pred):
    MAPE = np.mean(100 * (np.abs(orig-pred)/orig))
    print('#########',"Accuracy:", 100-MAPE)
    return (100-MAPE)

custom_scoring = make_scorer(accuracy_score, greater_is_better=True)
accuracy_values = cross_val_score(reg_model, X, Y, cv=10, scoring=custom_scoring)
print("Accuracy values for 10-fold Cross Validation:", accuracy_values)
print("Final Average Accuracy of model:", round(accuracy_values.mean(),2))
mean_accuracy["Random Forest Regression"].append(round(accuracy_values.mean(), 2))

dot_data = tree.export_graphviz(reg_model.estimators_[5], out_file=None, feature_names=predictors, class_names=["Price"])
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png('forest.png')

DTR = DecisionTreeRegressor(max_depth=3)
reg_model = AdaBoostRegressor(n_estimators=100, estimator=DTR, learning_rate=0.04)

AB = reg_model.fit(X_train, Y_train)
prediction = AB.predict(X_test)
print('R2 Value:',metrics.r2_score(Y_train, AB.predict(X_train)))

feature_importances = pd.Series(AB.feature_importances_, index=predictors)
feature_importances.nlargest(10).plot(kind='barh')
plt.show()

print("---------------- Model Validation and Accuracy Calculations----------------")
print("Adaboost Decision Tree Regression")
testing_results=pd.DataFrame(data=X_test, columns=predictors)
testing_results[target_var]=Y_test
testing_results[('Predicted'+target_var)]=np.round(prediction)

testing_results['APE'] = 100 *((abs(testing_results['Price'] - testing_results['PredictedPrice']))/testing_results['Price'])
MAPE = np.mean(testing_results['APE'])
median_MAPE = np.median(testing_results['APE'])

accuracy = 100- MAPE
median_accuracy = 100 - median_MAPE
print("Mean Accuracy on testing data:", accuracy)
print("Median accuracy on testing data:", median_accuracy)

def accuracy_score(orig, pred):
    MAPE = np.mean(100 * (np.abs(orig-pred)/orig))
    print('#########',"Accuracy:", 100-MAPE)
    return (100-MAPE)

custom_scoring = make_scorer(accuracy_score, greater_is_better=True)
accuracy_values = cross_val_score(reg_model, X, Y, cv=10, scoring=custom_scoring)
print("Accuracy values for 10-fold Cross Validation:", accuracy_values)
print("Final Average Accuracy of model:", round(accuracy_values.mean(),2))
mean_accuracy["Adaboost Decision Tree Regression"].append(round(accuracy_values.mean(), 2))

reg_model = XGBRegressor(max_depth=2, learning_rate=0.1, n_estimators=1000,objective='reg:linear',booster='gbtree')
XGB = reg_model.fit(X_train, Y_train)
prediction = XGB.predict(X_test)
print('R2 Value:',metrics.r2_score(Y_train, XGB.predict(X_train)))

feature_importances = pd.Series(XGB.feature_importances_, index=predictors)
feature_importances.nlargest(10).plot(kind='barh')
plt.show()

print("---------------- Model Validation and Accuracy Calculations----------------")
print("XGB Regression")
testing_results=pd.DataFrame(data=X_test, columns=predictors)
testing_results[target_var]=Y_test
testing_results[('Predicted'+target_var)]=np.round(prediction)

testing_results['APE'] = 100 *((abs(testing_results['Price'] - testing_results['PredictedPrice']))/testing_results['Price'])
MAPE = np.mean(testing_results['APE'])
median_MAPE = np.median(testing_results['APE'])

accuracy = 100- MAPE
median_accuracy = 100 - median_MAPE
print("Mean Accuracy on testing data:", accuracy)
print("Median accuracy on testing data:", median_accuracy)

def accuracy_score(orig, pred):
    MAPE = np.mean(100 * (np.abs(orig-pred)/orig))
    print('#########',"Accuracy:", 100-MAPE)
    return (100-MAPE)

custom_scoring = make_scorer(accuracy_score, greater_is_better=True)
accuracy_values = cross_val_score(reg_model, X, Y, cv=10, scoring=custom_scoring)
print("Accuracy values for 10-fold Cross Validation:", accuracy_values)
print("Final Average Accuracy of model:", round(accuracy_values.mean(),2))
mean_accuracy["XGB Regression"].append(round(accuracy_values.mean(), 2))

fig, ax = plt.subplots(figsize=(20, 8))
plot_tree(XGB, num_trees=10, ax=ax)
plt.show()

reg_model = KNeighborsRegressor(n_neighbors=3)
KNN = reg_model.fit(X_train, Y_train)
prediction = KNN.predict(X_test)
print('R2 Value:',metrics.r2_score(Y_train, KNN.predict(X_train)))

print("---------------- Model Validation and Accuracy Calculations----------------")
print("KNeighbors Regression")
testing_results=pd.DataFrame(data=X_test, columns=predictors)
testing_results[target_var]=Y_test
testing_results[('Predicted'+target_var)]=np.round(prediction)

testing_results['APE'] = 100 *((abs(testing_results['Price'] - testing_results['PredictedPrice']))/testing_results['Price'])
MAPE = np.mean(testing_results['APE'])
median_MAPE = np.median(testing_results['APE'])

accuracy = 100- MAPE
median_accuracy = 100 - median_MAPE
print("Mean Accuracy on testing data:", accuracy)
print("Median accuracy on testing data:", median_accuracy)

def accuracy_score(orig, pred):
    MAPE = np.mean(100 * (np.abs(orig-pred)/orig))
    print('#########',"Accuracy:", 100-MAPE)
    return (100-MAPE)

custom_scoring = make_scorer(accuracy_score, greater_is_better=True)
accuracy_values = cross_val_score(reg_model, X, Y, cv=10, scoring=custom_scoring)
print("Accuracy values for 10-fold Cross Validation:", accuracy_values)
print("Final Average Accuracy of model:", round(accuracy_values.mean(),2))
mean_accuracy["KNeighbors Regression"].append(round(accuracy_values.mean(), 2))

reg_model = svm.SVR(C=50, kernel='linear') #had to use linear kernel due to errors
SVM = reg_model.fit(X_train, Y_train)
prediction = SVM.predict(X_test)
print('R2 Value:',metrics.r2_score(Y_train, SVM.predict(X_train)))

feature_importances = pd.Series(SVM.coef_[0], index=predictors)
feature_importance.nlargest(10).plot(kind='barh')
plt.show()

print("---------------- Model Validation and Accuracy Calculations----------------")
print("Support Vector Machine Regression")
testing_results=pd.DataFrame(data=X_test, columns=predictors)
testing_results[target_var]=Y_test
testing_results[('Predicted'+target_var)]=np.round(prediction)

testing_results['APE'] = 100 *((abs(testing_results['Price'] - testing_results['PredictedPrice']))/testing_results['Price'])
MAPE = np.mean(testing_results['APE'])
median_MAPE = np.median(testing_results['APE'])

accuracy = 100- MAPE
median_accuracy = 100 - median_MAPE
print("Mean Accuracy on testing data:", accuracy)
print("Median accuracy on testing data:", median_accuracy)

def accuracy_score(orig, pred):
    MAPE = np.mean(100 * (np.abs(orig-pred)/orig))
    print('#########',"Accuracy:", 100-MAPE)
    return (100-MAPE)

custom_scoring = make_scorer(accuracy_score, greater_is_better=True)
accuracy_values = cross_val_score(reg_model, X, Y, cv=10, scoring=custom_scoring)
print("Accuracy values for 10-fold Cross Validation:", accuracy_values)
print("Final Average Accuracy of model:", round(accuracy_values.mean(),2))
mean_accuracy["Support Vector Machine Regression"].append(round(accuracy_values.mean(), 2))
print("------------------------------End Training--------------------------")
final_value = max(mean_accuracy, key=mean_accuracy.get)

print("The model with the highest accuracy is", final_value, "with an accuracy of:", mean_accuracy.get(final_value))

print("-------------------------------------Starting the Training of the Final Model---------------------------")
print("Linear Regression")
reg_model = LinearRegression()
LREG = reg_model.fit(X_train, Y_train)

def accuracy_score(orig, pred):
    MAPE = np.mean(100 * (np.abs(orig-pred)/orig))
    print('#########',"Accuracy:", 100-MAPE)
    return (100-MAPE)

custom_scoring = make_scorer(accuracy_score, greater_is_better=True)
accuracy_values = cross_val_score(reg_model, X, Y, cv=10, scoring=custom_scoring)
print("Accuracy values for 10-fold Cross Validation:", accuracy_values)
print("Final Average Accuracy of model:", round(accuracy_values.mean(),2))

final_linear_model = reg_model.fit(X,Y)
with open('final_linear_model.pkl', 'wb') as fileWriteStream:
    pickle.dump(final_linear_model, fileWriteStream)
    fileWriteStream.close()

print('pickle file of Predictive Model is saved at Location:',os.getcwd())


def FunctionPredictResult(InputData):
    import pandas as pd
    Num_Inputs=InputData.shape[0]
    DataForML=pd.read_pickle('model_data.pkl')
    InputData = pd.concat([InputData, DataForML], ignore_index=True)

    InputData=pd.get_dummies(InputData)

    predictors = ['Processor_Speed', 'RAM_Size', 'Storage_Capacity']

    X=InputData[predictors].values[0:Num_Inputs]

    X=PredictorScalerfit.transform(X)

    # Loading the Function from pickle file
    import pickle
    with open('final_linear_model.pkl', 'rb') as fileReadStream:
        PredictionModel=pickle.load(fileReadStream)
        fileReadStream.close()

    Prediction=PredictionModel.predict(X)
    PredictionResult=pd.DataFrame(Prediction, columns=['Prediction'])
    print(PredictionResult)
    return(PredictionResult)


NewSampleData=pd.DataFrame(data=[[1.34,8,256],[1.34,16,1000]],columns=['Processor_Speed', 'RAM_Size', 'Storage_Capacity'])

print(NewSampleData)

FunctionPredictResult(InputData=NewSampleData)

print("-------------------------------Predicting With Custom Input-----------------------")

def FunctionGeneratePrediction(inp_Processor , inp_RAM, inp_Storage):


    SampleInputData=pd.DataFrame(
     data=[[inp_Processor , inp_RAM, inp_Storage]],
     columns=['Processor_Speed', 'RAM_Size', 'Storage_Capacity'])

    Predictions=FunctionPredictResult(InputData= SampleInputData)

    return(Predictions.to_json())

FunctionGeneratePrediction(inp_Processor=4, inp_RAM=16, inp_Storage=512)

print("------------------Creating API-----------------------")
app = Flask(__name__)

@app.route('/prediction_api', methods=["GET"])
def prediction_api():
    try:
        Processor_value = float(request.args.get('Processor_Speed'))
        RAM_value=float(request.args.get('RAM_Size'))
        Storage_value=float(request.args.get('Storage_Capacity'))

        prediction_from_api=FunctionGeneratePrediction(inp_Processor=Processor_value,
                                                       inp_RAM=RAM_value,
                                                       inp_Storage=Storage_value)

        return (prediction_from_api)

    except Exception as e:
        return('Something is not right!: '+str(e))
    
print("Use this link to access the API: http://127.0.0.1:9000/prediction_api?Processor_Speed=3&RAM_Size=8&Storage_Capacity=256")
    
if __name__ =="__main__":
    app.run(host='127.0.0.1', port=9000, threaded=True, debug=True, use_reloader=False)

# Copy Paste link below to run the API and cahnge the values next to the equals signs for different values.
#http://127.0.0.1:9000/prediction_api?Processor_Speed=3&RAM_Size=8&Storage_Capacity=256

class LaptopPredictionApp:
    def __init__(self, master):
        self.master = master
        self.master.title('Laptop Price Prediction')
        self.data = pd.read_csv('Laptop_price.csv', usecols=[1,2,3,4,5,6])
        self.sliders = []

        self.X = self.data.drop('Price', axis=1).values
        self.y = self.data['Price'].values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)

        self.create_widgets()

    def create_widgets(self):
        for i, column in enumerate(self.data.columns[:-1]):
            label = tk.Label(self.master, text=column + ': ')
            label.grid(row=i, column=0)
            current_val_label = tk.Label(self.master, text='0.0')
            current_val_label.grid(row=i, column=2)
            slider = ttk.Scale(self.master, from_=self.data[column].min(), to=self.data[column].max(), orient="horizontal",
                               command=lambda val, label=current_val_label: label.config(text=f'{float(val):.2f}'))
            slider.grid(row=i, column=1)
            self.sliders.append((slider, current_val_label))

        predict_button = tk.Button(self.master, text='Predict Price', command=self.predict_price)
        predict_button.grid(row=len(self.data.columns[:-1]), columnspan=3)

    def predict_price(self):
        inputs = [float(slider.get()) for slider, _ in self.sliders]
        price = self.model.predict([inputs])
        messagebox.showinfo('Predicted Price', f'The predicted laptop price is ${price[0]:.2f}')

if __name__ == '__main__':
    root = tk.Tk()
    app = LaptopPredictionApp(root)
    root.mainloop()