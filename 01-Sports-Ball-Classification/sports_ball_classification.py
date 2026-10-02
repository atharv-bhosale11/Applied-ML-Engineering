"""
------------------------------------------------------------
Project Name        : Sports Ball Classification

Dataset Information :

Surface Encoding    :   Rough  = 1
                        Smooth = 0

Ball Type Encoding  :    Tennis Ball  = 1
                        Cricket Ball = 2

Features            :   Weight  -> Weight of Ball (grams)
                        Surface -> Surface Type

Target              :   Ball Type

Machine Learning Information:

Algorithm Used      :   Decision Tree Classifier

Library             :   Scikit-Learn

Problem Type        :   Classification

Author              :   Atharv Tushar Bhosale

Date                :   02/10/2026
------------------------------------------------------------
"""

from sklearn import tree
from sklearn.metrics import accuracy_score

"""
Function Name : DisplayData
Description   : Displays the title of Sports Ball Classification Case Study.
Input         : None
Output        : Prints project header on console.
Author        : Atharv Tushar Bhosale
Date          : 02/10/2026
"""

def DisplayData():
    print("------------------------------------------------------------")
    print("---------- Sports Ball Classification Case Study -----------")
    print("------------------------------------------------------------")

"""
Function Name : LoadDataSet
Description   : Loads sports ball dataset containing features and labels.
Input         : None
Output        : Returns X and Y datasets.
Author        : Atharv Tushar Bhosale
Date          : 02/10/2026
"""

def LoadDataSet():
    X = [[35,1],[47,1],[90,0],[48,1],[90,0],[35,1],
         [92,0],[35,1],[35,1],[35,1],[96,0],
         [43,1],[110,0],[35,1],[95,0]]

    Y = [1,1,2,1,2,1,2,1,1,1,2,1,2,1,2]

    return X,Y

"""
Function Name : PreparedData
Description   : Splits dataset into training and testing datasets.
Input         : None
Output        : Returns Xtrain, Xtest, Ytrain and Ytest.
Author        : Atharv Tushar Bhosale
Date          : 02/10/2026
"""

def PreparedData():
    Xtrain = [[35,1],[47,1],[90,0],[48,1],[90,0],
              [35,1],[92,0],[35,1],[35,1],[35,1],
              [96,0],[43,1],[110,0]]

    Xtest = [[35,1],[95,0]]

    Ytrain = [1,1,2,1,2,1,2,1,1,1,2,1,2]

    Ytest = [1,2]

    return Xtrain,Xtest,Ytrain,Ytest

"""
Function Name : TrainModel
Description   : Trains Decision Tree Classifier using training data.
Input         : Xtrain, Ytrain
Output        : Returns trained machine learning model.
Author        : Atharv Tushar Bhosale
Date          : 02/10/2026
"""

def TrainModel(Xtrain,Ytrain):

    model = tree.DecisionTreeClassifier()

    trainedmodel = model.fit(Xtrain,Ytrain)

    return trainedmodel

"""
Function Name : TestModel
Description   : Tests trained model and calculates prediction accuracy.
Input         : Model, Xtest, Ytest
Output        : Returns accuracy score.
Author        : Atharv Tushar Bhosale
Date          : 02/10/2026
"""

def TestModel(model,Xtest,Ytest):

    prediction = model.predict(Xtest)

    accuracy = accuracy_score(Ytest,prediction)

    return accuracy

"""
Function Name : PredictBall
Description   : Predicts sports ball category using trained model.
Input         : Trained model
Output        : Returns predicted ball type.
Author        : Atharv Tushar Bhosale
Date          : 02/10/2026
"""

def PredictBall(model):

    Result = model.predict([[35,1]])

    return Result

"""
Function Name : DisplayResult
Description   : Displays predicted ball category on console.
Input         : Prediction result
Output        : Prints Tennis Ball or Cricket Ball.
Author        : Atharv Tushar Bhosale
Date          : 02/10/2026
"""

def DisplayResult(Result):

    if Result[0] == 1:
        print("Tennis Ball")
    elif Result[0] == 2:
        print("Cricket Ball")

"""
Function Name : DisplayFooter
Description   : Displays project completion message.
Input         : None
Output        : Prints project footer on console.
Author        : Atharv Tushar Bhosale
Date          : 02/10/2026
"""

def DisplayFooter():
    print("------------------------------------------------------------")
    print("Sports Ball Classification Case Study Completed Successfully")
    print("------------------------------------------------------------")        

"""
Function Name : main
Description   : Controls complete machine learning workflow.
Input         : None
Output        : Displays prediction and model accuracy.
Author        : Atharv Tushar Bhosale
Date          : 02/10/2026
"""

def main():
    
    DisplayData()

    X,Y = LoadDataSet()

    Xtrain, Xtest, Ytrain, Ytest = PreparedData()

    model = TrainModel(Xtrain, Ytrain)

    accuracy = TestModel(model, Xtest, Ytest)

    Result = PredictBall(model)

    DisplayResult(Result)

    print("Accuracy is  : ",accuracy * 100,"%")

    DisplayFooter()

if __name__ == "__main__":
    main()
