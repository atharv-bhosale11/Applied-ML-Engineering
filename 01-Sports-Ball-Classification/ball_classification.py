import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def DisplayData():
    print("------------------------------------------------------------")
    print("---------- Sports Ball Classification Case Study -----------")
    print("------------------------------------------------------------")

def LoadDataSet():
    X = [[35,1],[47,1],[90,0],[48,1],[90,0],[35,1],
         [92,0],[35,1],[35,1],[35,1],[96,0],
         [43,1],[110,0],[35,1],[95,0]]

    Y = [1,1,2,1,2,1,2,1,1,1,2,1,2,1,2]

    return X,Y

def PreparedData():
    Xtrain = [[35,1],[47,1],[90,0],[48,1],[90,0],
              [35,1],[92,0],[35,1],[35,1],[35,1],
              [96,0],[43,1],[110,0]]

    Xtest = [[35,1],[95,0]]

    Ytrain = [1,1,2,1,2,1,2,1,1,1,2,1,2]

    Ytest = [1,2]

    return Xtrain,Xtest,Ytrain,Ytest

def TrainModel(Xtrain,Ytrain):

    model = tree.DecisionTreeClassifier()

    trainedmodel = model.fit(Xtrain,Ytrain)

    return trainedmodel

def TestModel(model,Xtest,Ytest):

    prediction = model.predict(Xtest)

    accuracy = accuracy_score(Ytest,prediction)

    return accuracy

def PredictBall(model):

    Result = model.predict([[35,1]])

    return Result

def DisplayResult(Result):

    if Result[0] == 1:
        print("Tennis Ball")
    elif Result[0] == 2:
        print("Cricket Ball")

def main():
    DisplayData()

    X,Y = LoadDataSet()

    Xtrain, Xtest, Ytrain, Ytest = PreparedData()

    model = TrainModel(Xtrain, Ytrain)

    accuracy = TestModel(model, Xtest, Ytest)

    Result = PredictBall(model)

    DisplayResult(Result)

    print("Accuracy is  : ",accuracy * 100,"%")

if __name__ == "__main__":
    main()
