# Importing the libraries
import pandas as pd
import numpy as np
import scipy 
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler 
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
    
    
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt   #Data visualisation libraries 
import seaborn as sns
#%matplotlib inline
'exec(%matplotlib inline)'

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def train_and_test(self, df,k, target):
    num_df=df.select_dtypes(include=['integer','float'])
    features=num_df.columns.drop(target)
    model=linear_model.LinearRegression()
    if k==0:
        cut=int(num_df.shape[0]/2)
        train=num_df.iloc[:cut]
        test=num_df.iloc[cut:]
        model.fit(train[feature],train[target])
        prediction=model.predict(test[target])
        mse = mean_squared_error(test[target], predictions)
        rmse = np.sqrt(mse)
        return rmse
    elif k==1:
        # Randomize *all* rows (frac=1) from `df` and return
        shuffled_df = df.sample(frac=1, )
        train = df[:1460]
        test = df[1460:]
        
        model.fit(train[features], train[target])
        predictions_one = model.predict(test[features])        
        
        mse_one = mean_squared_error(test[target], predictions_one)
        rmse_one = np.sqrt(mse_one)
        
        model.fit(test[features], test[target])
        predictions_two = model.predict(train[features])        
        mse_two = mean_squared_error(train[target], predictions_two)
        rmse_two = np.sqrt(mse_two)
        
        avg_rmse = np.mean([rmse_one, rmse_two])
        print(rmse_one)
        print(rmse_two)
        
        return avg_rmse
    else:
        kf = KFold(n_splits=k, shuffle=True)
        rmse_values = []
        test_group = []
        for train_index, test_index, in kf.split(df):
            train = df.iloc[train_index]
            test = df.iloc[test_index]
            model.fit(train[features], train[target])
            predictions = model.predict(test[features])
            test_group.append(str(test[target]))
            mse = mean_squared_error(test[target], predictions)
            rmse = np.sqrt(mse)
            rmse_values.append(rmse)
        print("RMSE values")
        print(rmse_values)
        avg_rmse = np.mean(rmse_values)
        print("average RMSE values")
        print(avg_rmse)
        return avg_rmse


def main():
    print()
    print("*"*120)
    print(color.BLUE  + "Welcome to Machine Learning Prediction Project for Heart Diseases!" + color.END )
    print(color.BLUE  +"This is an Excercise #1 under Big Data Fundamental course, CCBT Fall 2019. Lambton College!" + color.END )
    print(color.CYAN + "Team members:" + color.END)
    features = [ "The Cuong, Bui", "Chiranjeev Singh", "Gurpreet Singh Virdi", "Rosan", "Manoj Moond"]
    
    for i, name in enumerate(features):
        j=i+1
        print(color.CYAN + "%s. %s"% (str(j), name) + color.END)
    missing_value_threshold = 0.05
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
    data_file = "processed.cleveland.data"
    missing_values = ["n/a", "na", "--", "?"]
    features = [ "age", "sex", "cp", "trestbps", "chol","fbs", "restecg",
                   "thalach","exang", "oldpeak","slope", "ca", "thal"]
    target = "num"
    method = 'median'
    test_size = 20
    random_state =  0 # 101
    no_selected_features = 13
    k_fold = 10
    # Importing the datasets
    df = pd.read_csv(url + data_file, names=features + [target], na_values = missing_values)
    
    
    # Understanding data
    print()
    print("*"*120)
    print(color.BOLD + "Understanding Heart Disease data set" + color.END)
    print()
    print(color.UNDERLINE + "Data set features"+ color.END)
    print(features)
    print(color.UNDERLINE +"Data set target: " + color.END +  target)
    print()
    print(color.UNDERLINE +"Shape of Data frame (no of data row, no of attributes)"+ color.END + str(df.shape))
    print()
    print(color.UNDERLINE +"Data Types of data frame"+ color.END)
    print(df.dtypes)
    print()
    print(color.UNDERLINE +"Head of data frame"+ color.END)
    print(df.head(20))
    print() 
    print(color.UNDERLINE +"Descriptive  Data statistics" + color.END)
    print(df.describe())
    print()
    print(color.UNDERLINE + "Classification data distribution based on target"+ color.END)
    print(df.groupby(target).size())
    print()
    #Correlation Between Attributes
    print(color.UNDERLINE +"Correlation Between Attributes" + color.END)
    pd.set_option('display.width', 100)
    pd.set_option('precision', 3)
    print(df.corr(method='pearson'))
    print()
    
    # visualize the relationship between the features and the target
    print(color.UNDERLINE +"Visualize the relationship between each feature and the target" + color.END)
    sns.pairplot(df, x_vars=features, y_vars=[target], height=7, aspect=0.7)
    sns.distplot(df[target])
    print()
    
    # X = df.iloc[:, :-1].values
    # y = df.iloc[:, 13].values
    # # Process missing data
    print('*'*60)
    print(color.UNDERLINE +"Process Missing data"+ color.END)
    total_missing_value = 0;
    for feature in features:
        if df[feature].isnull().any().sum() > 0:
            total_missing_value += df[feature].isnull().any().sum()
    print("Total missing value before processing is %s!" %str(total_missing_value))
    rate_missing_value = total_missing_value
    
    if rate_missing_value == 0:
        print("No NaN or Missing value detect!")
    elif rate_missing_value < 10 or 'drop' in method:
        print("Rate of NaN or Missing value is very low (%s), drop missing value rows!" %str(rate_missing_value))
        df.dropna(inplace=True)
    else:
        imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
        imputer = imputer.fit(X[:, 0:13])
        X[:, 0:13] = imputer.transform(X[:, 0:13])
    # Checking if any missing value after process
    total_missing_value = 0
    for feature in features:
        if df[feature].isnull().any().sum() > 0:
            total_missing_value += df[feature].isnull().any().sum()
    print("Total missing value after processing NaN: %s!" %str(total_missing_value))
    print()
    
    # Process and remove outlier   
    print('*'*120)
    print(color.UNDERLINE +"Process and remove outlier"+ color.END)    
    for feature in features:
        # identify outliers
        data_std = df[feature].std()
        data_mean = df[feature].mean()
        cut_off = data_std * 3
        lower, upper = data_mean - cut_off, data_mean + cut_off
        # identify outliers
        outliers = [x for x in df[feature] if x < lower or x > upper]
        if len(outliers) > 0:
            print('Identified outliers in feature %s: %d' % (feature, len(outliers)))
            # remove outliers
            outliers_removed = [x for x in df[feature] if x >= lower and x <= upper]
            print('Non-outlier observations: %d' % len(outliers_removed))
    print()
    
    
    # #Splitting the X into X_train and X_test
    print("*"*120)
    print(color.BOLD + "Splitting the X into X_train and X_test" + color.END)
    print()
    y = df[target]
    X = df[features]
    
    #Feature scaling
    print("*"*120)
    print(color.BOLD + "Feature Scaling and normalize data" + color.END)
    print()
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    np.set_printoptions(precision=3)
    print(color.UNDERLINE + 'X data after scaling' + color.END)
    print(X)
    print()
    print("*"*120)
    print(color.BOLD + "Feature Extraction using PCA" + color.END)
    print()
    pca = PCA(n_components=no_selected_features)
    X = pca.fit_transform(X)
    
    print("*"*120)
    print(color.UNDERLINE + "The Training data set is split into train set and test set (%d:%d)" %(100-test_size, test_size) + color.END)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)
    
    # Estimating ("Learning") Model Coefficients
    ### STATSMODELS ###

    print("*"*60)
    print(color.BOLD + "STATSMODELS" + color.END)
    print()    
    olsmod = sm.OLS(y, X)
    olsres = olsmod.fit()
    print("Statis model summary")
    print(olsres.summary())
    print("Statis model parameters")
    print(olsres.params)
    ypred = olsres.predict(X)
    print(color.UNDERLINE + "Y Prediction static model" + color.END)
    print()
    print(ypred)
    # print the confidence intervals for the model coefficients
    print("the confidence intervals for the model coefficients")
    print(olsres.conf_int())
    
    
    #lets train our model
    ### SCIKIT-LEARN ###
    print(color.BOLD + "Training and Test Split Machine Learning Model using SCIKIT-LEARN" + color.END)
    print()
    mln_model=linear_model.LinearRegression()
    mln_model.fit(X_train,y_train)
    # print the coefficients
    print(color.UNDERLINE + "Linear model intercept:" + color.END + str(mln_model.intercept_))
    print(color.UNDERLINE + "Feature slope/Coeeficient" + color.END)
    # pair the feature names with the coefficients
    print(list(zip(features, mln_model.coef_)))
    print()
    # variance score: 1 means perfect prediction 
    print(color.UNDERLINE + 'Variance score (1 means perfect prediction: {}'.format(mln_model.score(X_test, y_test)) + color.END) 
    print()
    # coef_df = pd.DataFrame(mln_model.coef_, X.columns, columns=['Coefficient'])  
    # print(color.UNDERLINE + "Feature slope/Coefficient" + color.END)
    # print(coef_df.sort_values('Coefficient'))
    # matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
    # imp_coef.plot(kind = "barh")
    # plt.title("Feature importance")
    
    #Predicting the values of traing set
    y_pred_train = mln_model.predict(X_train)
    y_pred_train = np.round(y_pred_train)
    #Predicting the values of test set
    y_pred_test = mln_model.predict(X_test)
    y_pred_test = np.round(y_pred_test)
    plt.scatter(y_test,y_pred_test)
    #Check the difference between the actual value and predicted value.
    df1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_test})
    print(df1.head(25))
    df1.plot(kind='bar',figsize=(10,8))
    #plot the comparison of Actual and Predicted values
    print(color.UNDERLINE + "plot the comparison of Actual and Predicted values" + color.END)
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()
        
    # Model Evaluation Using Train/Test Split
    print("*"*120)
    print(color.BOLD + "Performance Evaluation using metrics!" + color.END)
    print()
    print(color.UNDERLINE + "Model Evaluation Using F1 Score in Train/Test Split:" + color.END + str(f1_score(y_test, y_pred_test, average='macro')))
    print()
    # RMSE
    ## calculate r-square
    # calculate MAE, MSE, RMSE
    print(color.UNDERLINE + "Model Evaluation Using MAE:" + color.END + str(metrics.mean_absolute_error(y_test, y_pred_test)) + color.END)
    print(color.UNDERLINE + "Model Evaluation Using MSE:" + color.END + str(metrics.mean_squared_error(y_test, y_pred_test)) + color.END)
    print(color.UNDERLINE + "Model Evaluation Using RMSE: " + str(np.sqrt(metrics.mean_squared_error(y_test, y_pred_test))) + color.END)
    print()
   
        
    #Confusion matrix
    #Train
    cm_train = confusion_matrix(y_train,y_pred_train)
    #Test
    cm_test = confusion_matrix(y_test,y_pred_test)

    tot_test = cm_test[0][0]+cm_test[1][1]+cm_test[0][1]+cm_test[1][0]
    tot_train = cm_train[0][0]+cm_train[1][1]+cm_train[0][1]+cm_train[1][0]

    print(color.UNDERLINE +"Accuracy of train set is :" + color.END,((cm_train[0][0]+cm_train[1][1])/tot_train)*100 , "%" )
    print()
    print(color.UNDERLINE +"Accuracy of test set is  : " + color.UNDERLINE  , ((cm_test[0][0]+cm_test[1][1])/tot_test)*100 , " %")
    print()

    print("*"*120)
    print(color.BOLD + "Improved Training model and test  using k-fold Validation" + color.END)
    print()
    #run.create_train_multivalue_linear_model(df, features + [target],0.2)
    #run.train_and_test(df, 10, "num")
    num_df=df.select_dtypes(include=['integer','float'])
    features=num_df.columns.drop(target)
    model=linear_model.LinearRegression()
    if k_fold==0:
        cut=int(num_df.shape[0]/2)
        train=num_df.iloc[:cut]
        test=num_df.iloc[cut:]
        mln_model.fit(train[feature],train[target])
        y_test_pred=mln_model.predict(test[target])
        mse = mean_squared_error(test[target], y_test_pred)
        rmse = np.sqrt(mse)
        return rmse
    elif k_fold==1:
        # Randomize *all* rows (frac=1) from `df` and return
        shuffled_df = df.sample(frac=1, )
        train = df[:1460]
        test = df[1460:]
        mln_model.fit(train[features], train[target])
        y_test_pred = mln_model.predict(test[features])
        mse_test = mean_squared_error(test[target], y_test_pred)
        rmse_test = np.sqrt(mse_test)
        mln_model.fit(test[features], test[target])
        y_train_pred = mln_model.predict(train[features])        
        mse_train = mean_squared_error(train[target], predictions_two)
        rmse_train = np.sqrt(mse_train)
        avg_rmse = np.mean([rmse_test, rmse_train])
        print("Training RMSE: %d"  %rmse_train)
        print("Test RMSE: %d"  %rmse_test)
        print("Average of RMSE: %d"  %avg_rmse)
    else:
        kf = KFold(n_splits=k_fold, shuffle=True)
        rmse_values = []
        test_group = []
        for train_index, test_index, in kf.split(df):
            train = df.iloc[train_index]
            test = df.iloc[test_index]
            mln_model.fit(train[features], train[target])
            predictions = mln_model.predict(test[features])
            #test_group.append(str(test[target]))
            mse = mean_squared_error(test[target], predictions)
            rmse = np.sqrt(mse)
            rmse_values.append(rmse)
        print(color.UNDERLINE + "RMSE values" + color.END)
        print(rmse_values)
        avg_rmse = np.mean(rmse_values)
        print(color.UNDERLINE + "average RMSE values" + color.END)
        print(avg_rmse)

    
main()







