# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 15:28:03 2020
@author: nirbhay kumar, barclays
"""
#Credit Classification
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np

credit_df = pd.read_csv("barclays credit data.csv")
credit_df.info()
credit_df.iloc[0:5,1:7]
credit_df.iloc[0:5,7:]
credit_df.status.value_counts()
X_features = list( credit_df.columns )
X_features.remove( 'status' )
X_features

#Encoding Categorical Features
encoded_credit_df = pd.get_dummies( credit_df[X_features], drop_first = True )
list(encoded_credit_df.columns)
encoded_credit_df[['checkin_acc_A12','checkin_acc_A13','checkin_acc_A14']].head(5)

import statsmodels.api as sm
Y = credit_df.status
X = sm.add_constant( encoded_credit_df )

#Splitting into Train and Validation Sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.3,random_state = 42)

#Building Logistic Regression Model
logit = sm.Logit(y_train, X_train)
logit_model = logit.fit()

#Printing Model Summary
logit_model.summary2()

#Model Dignostics
def get_significant_vars( lm ):
    var_p_vals_df = pd.DataFrame( lm.pvalues )
    var_p_vals_df['vars'] = var_p_vals_df.index
    var_p_vals_df.columns = ['pvals', 'vars']
    return list( var_p_vals_df[var_p_vals_df.pvals <= 0.05]['vars'] )

significant_vars = get_significant_vars( logit_model )
significant_vars

final_logit = sm.Logit( y_train, sm.add_constant( X_train[significant_vars] ) ).fit()
final_logit.summary2()

#Predicting on Test Data
y_pred_df = pd.DataFrame( { "actual": y_test, "predicted_prob": final_logit.predict(sm.add_constant( X_test[significant_vars]))})
y_pred_df.sample(10, random_state = 42)
y_pred_df['predicted'] = y_pred_df.predicted_prob.map(lambda x: 1 if x > 0.5 else 0)
y_pred_df.sample(10, random_state = 42)

#Creating a Confusion Matrix
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn import metrics
def draw_cm( actual, predicted ):
    ## Cret
    cm = metrics.confusion_matrix( actual, predicted, [1,0] )
    sn.heatmap(cm, annot=True, fmt='.2f',
    xticklabels = ["Bad credit", "Good Credit"] ,
    yticklabels = ["Bad credit", "Good Credit"] )
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
draw_cm( y_pred_df.actual,y_pred_df.predicted )

#Measuring Accuracies
print( metrics.classification_report( y_pred_df.actual,y_pred_df.predicted ))
plt.figure( figsize = (8,6) )
sn.distplot( y_pred_df[y_pred_df.actual == 1]["predicted_prob"],kde=False, color = 'b',label = 'Bad Credit' )
sn.distplot( y_pred_df[y_pred_df.actual == 0]["predicted_prob"],kde=False, color = 'g',label = 'Good Credit' )
plt.legend()
plt.show()

#ROC & AUC
def draw_roc( actual, probs ):
    fpr, \
    tpr, \
    thresholds = metrics.roc_curve( actual,
    probs,
    drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(8, 6))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()
    return fpr, tpr, thresholds
fpr, tpr, thresholds = draw_roc( y_pred_df.actual,y_pred_df.predicted_prob)
auc_score = metrics.roc_auc_score( y_pred_df.actual,y_pred_df.predicted_prob )
round( float( auc_score ), 2 )

#Finding Optimal Cutoff
tpr_fpr = pd.DataFrame( { 'tpr': tpr,'fpr': fpr,'thresholds': thresholds })
tpr_fpr['diff'] = tpr_fpr.tpr - tpr_fpr.fpr
tpr_fpr.sort_values( 'diff', ascending = False )[0:5]

y_pred_df['predicted_new'] = y_pred_df.predicted_prob.map(lambda x: 1 if x > 0.22 else 0)
draw_cm( y_pred_df.actual,y_pred_df.predicted_new)
print(metrics.classification_report( y_pred_df.actual,y_pred_df.predicted_new ))

#Cost Based Approach
def get_total_cost( actual, predicted, cost_FPs, cost_FNs ):
    cm = metrics.confusion_matrix( actual, predicted, [1,0] )
    cm_mat = np.array( cm )
    return cm_mat[0,1] * cost_FNs + cm_mat[1,0] * cost_FPs

cost_df = pd.DataFrame( columns = ['prob', 'cost'])

idx = 0
# iterate cut-off probability values between 0.1 and 0.5
for each_prob in range( 10, 50):
    cost = get_total_cost( y_pred_df.actual, y_pred_df.predicted_prob.map(lambda x: 1 if x > (each_prob/100) else 0), 1, 5 )
    cost_df.loc[idx] = [(each_prob/100), cost]
    idx += 1

cost_df.sort_values( 'cost', ascending = True )[0:5]
y_pred_df['predicted_using_cost'] = y_pred_df.predicted_prob.map(lambda x: 1 if x > 0.14 else 0)
draw_cm( y_pred_df.actual,y_pred_df.predicted_using_cost )

#Gain Chart and Lift Chart
#Loading and Preparing the Dataset
import pandas as pd
bank_df = pd.read_csv( 'bank.csv' )
bank_df.head( 5 )
bank_df.info()
X_features = list( bank_df.columns )
X_features.remove( 'subscribed' )
X_features

encoded_bank_df = pd.get_dummies( bank_df[X_features],drop_first = True )
Y = bank_df.subscribed.map( lambda x: int( x == 'yes') )
X = encoded_bank_df

#Building the Logistic Regression Model
logit_model = sm.Logit( Y, sm.add_constant( X ) ).fit()
logit_model.summary2()

significant_vars = get_significant_vars( logit_model )
significant_vars

X_features = ['current-campaign','previous-campaign','job_retired','marital_married','education_tertiary','housing-loan_yes','personal-loan_yes']
logit_model_2 = sm.Logit( Y, sm.add_constant( X[X_features] ) ).fit()
logit_model_2.summary2()

y_pred_df = pd.DataFrame( { 'actual': Y,'predicted_prob': logit_model_2.predict(sm.add_constant( X[X_features] ) ) } )
sorted_predict_df = y_pred_df[['predicted_prob','actual']].sort_values( 'predicted_prob',ascending = False )
num_per_decile = int( len( sorted_predict_df ) / 10 )
print( "Number of observations per decile: ", num_per_decile)

def get_deciles( df ):
    df['decile'] = 1
    
    idx = 0
    
    for each_d in range( 0, 10 ):
        df.iloc[idx:idx+num_per_decile, df.columns.get_loc('decile')] = each_d
        idx += num_per_decile
    df['decile'] = df['decile'] + 1
    return df

deciles_predict_df = get_deciles( sorted_predict_df )
deciles_predict_df[0:10]

gain_lift_df = pd.DataFrame(deciles_predict_df.groupby('decile')['actual'].sum() ).reset_index()
gain_lift_df.columns = ['decile', 'gain']
gain_lift_df['gain_percentage'] = (100 * gain_lift_df.gain.cumsum()/gain_lift_df.gain.sum())
gain_lift_df

plt.figure( figsize = (8,4))
plt.plot( gain_lift_df['decile'],
gain_lift_df['gain_percentage'], '-' )
plt.show()

gain_lift_df['lift'] = ( gain_lift_df.gain_percentage / ( gain_lift_df.decile * 10) )
plt.figure( figsize = (8,4))
plt.plot( gain_lift_df['decile'], gain_lift_df['lift'], '-' )
plt.show()

#Decision Trees
#Split the dataset
Y = credit_df.status
X = encoded_credit_df
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, Y,test_size = 0.3,random_state = 42)

#Building Decision Tree classifier using Gini Criteria
from sklearn.tree import DecisionTreeClassifier
clf_tree = DecisionTreeClassifier(criterion = 'gini',max_depth = 3 )
clf_tree.fit( X_train, y_train )

#Measuring Test Accuracy
tree_predict = clf_tree.predict( X_test )
metrics.roc_auc_score( y_test, tree_predict )