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

import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt   #Data visualisation libraries 
import seaborn as sns
#%matplotlib inline
'exec(%matplotlib inline)'

class HEART_DISEASE_ANALYZER:
      
    def __init__(self):
        self.missing_value_threshold = 0.05
        self.url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
        self.data_file = "processed.cleveland.data"
        self.missing_values = ["n/a", "na", "--", "?"]
        self.features = [ "age", "sex", "cp", "trestbps", "chol","fbs", "restecg",
                       "thalach","exang", "oldpeak","slope", "ca", "thal"]
        self.target = "num"
        self.method = 'median'
        self.test_size = 20
        self.random_state =  0 # 101
    
    def preprocess_df_missing_value(self, df, features, method):
        """ Preprocessing missing data"""
        # Missing data
        total_missing_value = 0;
        for feature in features:
            if df[feature].isnull().any().sum() > 0:
                total_missing_value += df[feature].isnull().any().sum()
        print("Total missing value is %s!" %str(total_missing_value))
        rate_missing_value = total_missing_value/df.shape()[0]
        
        if rate_missing_value == 0:
            print("No NaN or Missing value detect!")
            return df
        elif rate_missing_values < 0.01 or 'drop' in method:
            print("Rate of NaN or Missing value is very low (%s), drop missing value rows!" %str(rate_missing_values))
            df.dropna(inplace=True)
            return df
        for feature in features:
            if df[feature].isnull().any().sum() > 0:
                # calculate summary statistics
                print("Detect number of missing values of feature (" + name + "):" + str(df[name].isnull().any().sum()))
                #Deal with missing values
                if 'ffill'  in method:
                    df = df.fillna(method='ffill')
                    print("Convert missing value using %s")
                elif 'Bfill'  in method:
                    df = df.fillna(method='Bfill')
                    print("Convert missing value using %s")
                else:
                    replace_data = df[name].mean()
                    if 'median' in method:
                        replace_data = df[name].median()
                    elif 'std' in method:
                        replace_data = df[name].std()
                    elif 'std' in method:
                        replace_data = df[name].std() 
                    df[feature].fillna(replace_data, inplace=True)
                    print("Convert missing value using %s value (" + str(replace_data) + ")")
        return df
    
    
    def preprocess_imputer_missing_value(self, X, n, method):
        """ Preprocessing missing data"""
        imputer = Imputer(missing_values = 'NaN', strategy = method, axis = 0)
        imputer = imputer.fit(X[:, 0:n])
        X[:, 0:n] = imputer.transform(X[:, 0:n])
        return X
        
    def preprocessing(df, features):
        # Missing data
        
        # imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
        # imputer = imputer.fit(X[:, 0:13])
        # X[:, 0:13] = imputer.transform(X[:, 0:13])
        for name in names:
            if df[name].isnull().any().sum() > 0:
                # calculate summary statistics
                data_median = df[name].median()
                data_mean = df[name].mean()
                data_std = df[name].std()
                #Deal with missing values
                print("---Detect number of missing value of " + name + ": " + str(df[name].isnull().any().sum()))
                print("---Convert missing value to: mean=" + str(data_median))
                df[name].fillna(data_median, inplace=True)
                #dataset = dataset.fillna(method='ffill')
                print("Number of Missing value after preprocess of " + name + ": " + str(df[name].isnull().any().sum()))
                # identify outliers
                cut_off = data_std * 3
                lower, upper = data_mean - cut_off, data_mean + cut_off
                # identify outliers
                outliers = [x for x in df[name] if x < lower or x > upper]
                print('Identified outliers: %d' % len(outliers))
                # remove outliers
                outliers_removed = [x for x in df[name] if x >= lower and x <= upper]
                print('Non-outlier observations: %d' % len(outliers_removed))
        return df

    def scaling(df, names):
        print("*"*20)
        print("ReScaling and normalize data")
        print()
        # ReScaling and normalize data
        array = df.values
        # separate array into input and output components
        X = array[:,0:13]
        Y = array[:,1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        rescaledX = scaler.fit_transform(X)
        # summarize transformed data
        numpy.set_printoptions(precision=3)
        print(rescaledX[0:13,:])
        return df
    
    def select_features(df, coeff_threshold, uniq_threshold, names, target):
        num_df=df.select_dtypes(include=['integer','float'])
        corrs=num_df.corr()[target].abs()
        #keeping only columns that have correlation with target higher than threshold
        df=df.drop(corrs[corrs<coeff_threshold].index, axis=1)
        
        #check to see if our current dataset still keeps the nominal features
        transform_cat_cols=[]
        for col in names[:-1]:
            if col in df.columns:
                transform_cat_cols.append(col)
        
        #getting rid of nominal columns with too many unique values
        for col in transform_cat_cols:
            len(df[col].unique())>uniq_threshold
            df=df.drop(col, axis=1)
            
        #convert text columns to dummy variables
        # text_cols=df.select_dtypes(include=['object'])
        # for col in text_cols:
            # df[col]=df[col].astype('category')
        
        # df=pandas.concat([df,pandas.get_dummies(df.select_dtypes(include=['category']))],axis=1)
        
        return df

    def feature_selection_pca(df):
        array = df.values
        X = array[:,0:13]
        Y = array[:,8]
        # feature extraction
        pca = PCA(n_components=3)
        fit = pca.fit(X)
        # summarize components
        print("Explained Variance: %s" % fit.explained_variance_ratio_)
        print(fit.components_)
        return fit

    def important_feature_selection(df):
        array = df.values
        X = array[:,0:-1]
        Y = array[:,-1]
        # feature extraction
        model = ExtraTreesClassifier(n_estimators=10)
        model.fit(X, Y)
        print(model.feature_importances_)

    def create_train_linear_model (self, df,names, test_size):
        X = df[names[:-1]]
        y = df[target]
        test_size=0.4
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=101)
        model=linear_model.LinearRegression()
        lm = LinearRegression()
        model.fit(X_train,y_train)
        model.predict(X_test)
        plt.scatter(y_test,predictions)
        #To retrieve the intercept:
        print(model.intercept_)
        #For retrieving the slope:
        print(model.coef_)
        y_pred = regressor.predict(X_test)
        df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
        df1 = df.head(25)
        df1.plot(kind='bar',figsize=(16,10))
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        plt.show()
        plt.scatter(X_test, y_test,  color='gray')
        plt.plot(X_test, y_pred, color='red', linewidth=2)
        plt.show()
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
        print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
        

    def create_train_multivalue_linear_model (self, df,features, test_size):
        #divide the data into “attributes” and “labels”. 
        array = df.values
        X = array[:,0:-1]
        y = array[:,-1]
        target = features[-1]
        X1 = df[features[:-1]]
        y1 = df[target]
        #Let's check the average value of the “quality” column.
        plt.figure(figsize=(15,10))
        plt.tight_layout()
        sns.distplot(df[target])
        
        #  we split 80% of the data to the training set while 20% of the data
        test_size=0.2
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=101)
        #lets train our model
        mln_model=linear_model.LinearRegression()
        mln_model.fit(X_train,y_train)
        #in the case of multivariable linear regression, the regression model 
        #has to find the most optimal coefficients for all the attributes. 
        #To see what coefficients our regression model has chosen:
        #To retrieve the intercept:
        print('Intercept: \n', mln_model.intercept_)
        print('Coefficients: \n', mln_model.coef_)
        
        # with statsmodels
        X = sm.add_constant(X) # adding a constant
        
        ## regression coefficients 
        #print('Coefficients: \n', reg.coef_) 
       # variance score: 1 means perfect prediction 
        #print('Variance score: {}'.format(reg.score(X_test, y_test))) 
        
        #df.corr()
        #coef_df = pandas.Series(mln_model.coef_, index = X1.columns)
        coef_df = pd.DataFrame(mln_model.coef_, X1.columns, columns=['Coefficient'])  
        print(coef_df.sort_values('Coefficient'))
        # matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
        # imp_coef.plot(kind = "barh")
        # plt.title("Feature importance")
         
        # model = sm.OLS(Y, X).fit()
        # predictions = model.predict(X) 
        # print_model = model.summary()
        # print(print_model)
        
        #do prediction on test data
        y_pred = mln_model.predict(X_test)
        # plt.scatter(y_test,y_pred)
        
        #Check the difference between the actual value and predicted value.
        df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        df1 = df.head(25)
        
        # #plot the comparison of Actual and Predicted values
        # df1.plot(kind='bar',figsize=(10,8))
        # plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
        # plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        # plt.show()
        
        #evaluate the performance of the algorithm. 
        #We’ll do this by finding the values for MAE, MSE, and RMSE. Execute the following script:
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
        print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
        
        
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
    run = HEART_DISEASE_ANALYZER()
    features = run.features
    target = run.target
    method = run.method
    test_size = run.test_size
    random_state = run.random_state
    # Importing the datasets
    print(features)
    df = pd.read_csv(run.url + run.data_file, names=features + [target], na_values = run.missing_values)
    
    
    # Understanding data
    print()
    print("*"*20)
    print("Understanding data")
    print(df.head(20))
    print(df.shape)
    print(df.dtypes)
    # visualize the relationship between the features and the response using scatterplots
    # print()
    # print("*"*20)
    # print("visualize the relationship between the features and the target using scatterplots")
    # sns.pairplot(df, x_vars=features, y_vars=[target], size=7, aspect=0.7)
    # sns.distplot(df[target])

    #Descriptive Statistics
    print()
    print("*"*20)
    print("Descriptive Statistics")
    print(df.info())
    print(df.describe())
    print(df.columns)
    
    print()
    print("*"*20)
    print("Classification distribution")
    #Classification distribution 
    print(df.groupby(target).size())
    #Correlation Between Attributes
    pd.set_option('display.width', 100)
    pd.set_option('precision', 3)
    print(df.corr(method='pearson'))
    #Skew attributes
    print(df.skew())

    X = df.iloc[:, :-1].values
    y = df.iloc[:, 13].values
    print(df.head())
    # Missing data
    total_missing_value = 0;
    for feature in features:
        if df[feature].isnull().any().sum() > 0:
            total_missing_value += df[feature].isnull().any().sum()
    print("Total missing value is %s!" %str(total_missing_value))
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
        
       
    
    #Splitting the X into X_tr and X_te 
    
    new_features = ["ca", "cp", "slope", "oldpeak", "thal", "exang", "restecg", "sex"]
    new_features = ["ca", "cp", "slope", "oldpeak", "thal"]
    y = df[target]
    X = df[new_features]
    #X = df.drop([target], axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)
   
    
    #Feature scaling
    print("*"*20)
    print("Feature Scaling and normalize data")
    print()
    # separate array into input and output components
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    # summarize transformed data
    np.set_printoptions(precision=3)
    print(X)
    print(X_train)
    print(X_test)
    
    
    # Estimating ("Learning") Model Coefficients
    ### STATSMODELS ###

    # create a fitted model
    # lm1 = smf.ols(formula='Sales ~ TV', data=data).fit()
    # # print the coefficients
    # lm1.params
    print("*"*20)
    print("STATSMODELS")
    print()    
    olsmod = sm.OLS(y, X)
    olsres = olsmod.fit()
    print("Statis model summary")
    print(olsres.summary())
    print("Statis model parameters")
    print(olsres.params)
    ypred = olsres.predict(X)
    print("Y Prediction static model")
    print()
    print(ypred)
    # print the confidence intervals for the model coefficients
    print("the confidence intervals for the model coefficients")
    print(olsres.conf_int())
    
    
    #lets train our model
    ### SCIKIT-LEARN ###
    
    print("*"*20)
    print("SCIKIT-LEARN")
    print()
    mln_model=linear_model.LinearRegression()
    mln_model.fit(X_train,y_train)
    # print the coefficients
    print("Linear model intercept")
    print(mln_model.intercept_)
    print("Linear model Coeeficient")
    #print(mln_model.coef_)
    # pair the feature names with the coefficients
    print(list(zip(features, mln_model.coef_)))
    
    #Predicting the values
    #Predicting the values of traing set
    y_pred_train = mln_model.predict(X_train)
    y_pred_train = np.round(y_pred_train)
    #Predicting the values of test set
    y_pred_test = mln_model.predict(X_test)
    y_pred_test = np.round(y_pred_test)
    
    #Predicting the test values
    y_pred = mln_model.predict(X_test)
    y_pred = np.round(y_pred)

    # Model Evaluation Using Train/Test Split
    from sklearn.metrics import f1_score
    print()
    print("******F1 Score********")
    print(f1_score(y_test, y_pred, average='macro'))
    # RMSE
    ## calculate r-square 
    #lm1.rsquared
    print()
    print("********RMSE********")
    print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    # calculate MAE, MSE, RMSE
    # print(metrics.mean_absolute_error(y_true, y_pred))
    # print(metrics.mean_squared_error(y_true, y_pred))
    # print(np.sqrt(metrics.mean_squared_error(y_true, y_pred)))
    print()
    print("*"*20)
    
    #Confusion matrix
    from sklearn.metrics import confusion_matrix
    #Train
    cm_train = confusion_matrix(y_train,y_pred_train)
    #Test
    cm_test = confusion_matrix(y_test,y_pred_test)

    tot_test = cm_test[0][0]+cm_test[1][1]+cm_test[0][1]+cm_test[1][0]
    tot_train = cm_train[0][0]+cm_train[1][1]+cm_train[0][1]+cm_train[1][0]

    print("The Training data set is split into train set and test set (2:8)")

    print("Accuracy of train set is :",((cm_train[0][0]+cm_train[1][1])/tot_train)*100 , "%" )

    print("Accuracy of test set is  : " , ((cm_test[0][0]+cm_test[1][1])/tot_test)*100 , " %")

    
    
    print()
    print("*"*20)
    print("Training and test k-fold")
    print()
    #run.create_train_multivalue_linear_model(df, features + [target],0.2)
    run.train_and_test(df, 10, "num")

    
main()







