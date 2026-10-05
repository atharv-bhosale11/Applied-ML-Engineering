import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
 
"""
Function Name : DisplayData
Description   : Displays project header.
Input         : None
Output        : Prints project title.
Author        : Atharv Tushar Bhosale
Date          : 05/10/2026
"""

def DisplayData():

    print("----------------------------------------------------------------------------------------------------")
    print("------------------------------- Iris Flower Classification Case Study ------------------------------")
    print("----------------------------------------------------------------------------------------------------")


"""
Function Name : LoadDataSet
Description   : Loads Iris dataset from CSV file.
Input         : Dataset Path
Output        : Returns DataFrame.
Author        : Atharv Tushar Bhosale
Date          : 05/10/2026
"""

def LoadDataSet(DataPath):

    df = pd.read_csv(DataPath)

    print("Dataset Loaded Successfully")
    print(df.head())

    return df


"""
Function Name : RemoveUnwantedColumns
Description   : Removes unnecessary columns from dataset.
Input         : DataFrame
Output        : Returns cleaned DataFrame.
Author        : Atharv Tushar Bhosale
Date          : 05/10/2026
"""

def RemoveUnwantedColumns(df):

    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    if 'Id' in df.columns:
        df.drop(columns=['Id'], inplace=True)

    print("Dataset Shape After Cleaning :", df.shape)

    return df


"""
Function Name : CheckMissingValues
Description   : Displays missing values from dataset.
Input         : DataFrame
Output        : Missing values information.
Author        : Atharv Tushar Bhosale
Date          : 05/10/2026
"""

def CheckMissingValues(df):

    print("Missing Values")
    print(df.isnull().sum())


"""
Function Name : DisplayStatisticalSummary
Description   : Displays statistical summary.
Input         : DataFrame
Output        : Statistical report.
Author        : Atharv Tushar Bhosale
Date          : 05/10/2026
"""

def DisplayStatisticalSummary(df):

    print(df.describe())


"""
Function Name : DisplayClassDistribution
Description   : Displays class distribution.
Input         : DataFrame
Output        : Species count.
Author        : Atharv Tushar Bhosale
Date          : 05/10/2026
"""

def DisplayClassDistribution(df):

    print(df['species'].value_counts())


"""
Function Name : EncodeTargetVariable
Description   : Encodes target variable.
Input         : DataFrame
Output        : Encoded DataFrame and LabelEncoder object.
Author        : Atharv Tushar Bhosale
Date          : 05/10/2026
"""

def EncodeTargetVariable(df):

    le = LabelEncoder()

    df['Species_Encoded'] = le.fit_transform(df['species'])

    print("Encoded Classes :", list(le.classes_))

    for index, value in enumerate(le.classes_):
        print(index, ":", value)

    return df, le


"""
Function Name : PrepareData
Description   : Creates independent and dependent variables.
Input         : DataFrame
Output        : X and Y.
Author        : Atharv Tushar Bhosale
Date          : 05/10/2026
"""

def PrepareData(df):

    X = df[['sepal length (cm)',
            'sepal width (cm)',
            'petal length (cm)',
            'petal width (cm)']]

    Y = df['Species_Encoded']

    return X, Y


"""
Function Name : SplitData
Description   : Splits dataset into training and testing data.
Input         : X and Y
Output        : Training and Testing datasets.
Author        : Atharv Tushar Bhosale
Date          : 05/10/2026
"""

def SplitData(X, Y):

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, Y_train, Y_test


"""
Function Name : TrainKNNModel
Description   : Trains KNN classifier.
Input         : X_train and Y_train
Output        : Trained KNN model.
Author        : Atharv Tushar Bhosale
Date          : 05/10/2026
"""

def TrainKNNModel(X_train, Y_train):

    knn = KNeighborsClassifier(n_neighbors=5)

    knn.fit(X_train, Y_train)

    return knn


"""
Function Name : TrainDecisionTreeModel
Description   : Trains Decision Tree classifier.
Input         : X_train and Y_train
Output        : Trained Decision Tree model.
Author        : Atharv Tushar Bhosale
Date          : 05/10/2026
"""

def TrainDecisionTreeModel(X_train, Y_train):

    dt = DecisionTreeClassifier(random_state=42)

    dt.fit(X_train, Y_train)

    return dt


"""
Function Name : CompareModels
Description   : Compares KNN and Decision Tree models.
Input         : Models, X_test, Y_test, LabelEncoder
Output        : Displays accuracy and best model.
Author        : Atharv Tushar Bhosale
Date          : 05/10/2026
"""

def CompareModels(KNNModel, DTModel, X_test, Y_test, le):

    KNNPrediction = KNNModel.predict(X_test)
    DTPrediction = DTModel.predict(X_test)

    KNNAccuracy = accuracy_score(Y_test, KNNPrediction)
    DTAccuracy = accuracy_score(Y_test, DTPrediction)

    print("\nKNN Accuracy :", round(KNNAccuracy * 100, 2), "%")
    print("Decision Tree Accuracy :", round(DTAccuracy * 100, 2), "%")

    print("\nClassification Report (KNN)")
    print(classification_report(
        Y_test,
        KNNPrediction,
        target_names=le.classes_
    ))

    print("\nClassification Report (Decision Tree)")
    print(classification_report(
    Y_test,
    DTPrediction,
    target_names=le.classes_
    ))

    if KNNAccuracy > DTAccuracy:
        print("Best Model : KNN")
    elif DTAccuracy > KNNAccuracy:
        print("Best Model : Decision Tree")
    else:
        print("Both Models Perform Equally")


"""
Function Name : DisplayFooter
Description   : Displays project completion message.
Input         : None
Output        : Prints project footer on console.
Author        : Atharv Tushar Bhosale
Date          : 05/10/2026
"""

def DisplayFooter():
   
    print("----------------------------------------------------------------------------------------------------")
    print("------------------ Iris Flower Classification Case Study Completed Successfully --------------------")
    print("----------------------------------------------------------------------------------------------------")


"""
Function Name : main
Description   : Controls complete Iris Classification workflow.
Input         : None
Output        : Displays model accuracy and report.
Author        : Atharv Tushar Bhosale
Date          : 05/10/2026
"""

def main():

    Border = "--"*50

    DataPath = "iris.csv"

    DisplayData()
    
    df = LoadDataSet(DataPath)
    print(Border)

    df = RemoveUnwantedColumns(df)
    print(Border)

    CheckMissingValues(df)
    print(Border)

    DisplayStatisticalSummary(df)
    print(Border)

    DisplayClassDistribution(df)
    print(Border)

    df, le = EncodeTargetVariable(df)

    X, Y = PrepareData(df)

    X_train, X_test, Y_train, Y_test = SplitData(X, Y) 

    KNNModel = TrainKNNModel(X_train, Y_train)
    print(Border)

    DTModel = TrainDecisionTreeModel(X_train, Y_train)
    print(Border)

    CompareModels(
        KNNModel,
        DTModel,
        X_test,
        Y_test,
        le
    )

    print()
    DisplayFooter()

if __name__ == "__main__":
    main()
