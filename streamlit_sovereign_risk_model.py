import streamlit as st
import warnings # hides warning messages
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd   
import matplotlib.pyplot as plt

import seaborn as sns
#from imblearn.over_sampling import SMOTE
#import itertools
import math

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
st.set_option('deprecation.showPyplotGlobalUse', False)

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
#########

#st.title("------------ Model Background ----------------")
st.title("Sovereign Risk Model: Calculating the Probability that a Country will Default")
st.title("-------------------------------------------")
st.title("Model Background")
st.write("""
## Scroll down for web app functionality
""")
st.title("-------------------------------------------")
from PIL import Image
img = Image.open("world_map.jpg")
st.image(img, width = 700, caption = "Map of all countries")


st.write(""" #  """)
st.write("""
# Countries have defaulted over the years
""")

country_name = st.sidebar.selectbox("Select Country", ('Afghanistan', 'Albania', 'Algeria', 'American Samoa', 'Andorra',
       'Angola', 'Antigua and Barbuda', 'Argentina', 'Armenia', 'Aruba',
       'Australia', 'Austria', 'Azerbaijan', 'Bahamas The', 'Bahrain',
       'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin',
       'Bermuda', 'Bhutan', 'Bolivia', 'Bosnia and Herzegovina',
       'Botswana', 'Brazil', 'Brunei Darussalam', 'Bulgaria',
       'Burkina Faso', 'Burundi', 'Cambodia', 'Cameroon', 'Canada',
       'Cape Verde', 'Cayman Islands', 'Central African Republic', 'Chad',
       'Chile', 'China', 'Colombia', 'Comoros', 'Congo Dem.Rep.',
       'Congo Rep.', 'Costa Rica', "Cote d'Ivoire", 'Croatia', 'Cuba',
       'Curacao', 'Cyprus', 'Czech Republic', 'Denmark', 'Djibouti',
       'Dominica', 'Dominican Republic', 'Ecuador', 'Egypt Arab Rep.',
       'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia',
       'Ethiopia', 'Fiji', 'Finland', 'France', 'French Polynesia',
       'Gabon', 'Gambia The', 'Georgia', 'Germany', 'Ghana', 'Greece',
       'Greenland', 'Grenada', 'Guam', 'Guatemala', 'Guinea',
       'Guinea-Bissau', 'Guyana', 'Haiti', 'Honduras',
       'Hong Kong SARChina', 'Hungary', 'Iceland', 'India', 'Indonesia',
       'Iran Islamic Rep.', 'Iraq', 'Ireland', 'Isle of Man', 'Israel',
       'Italy', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya',
       'Kiribati', 'Korea Dem.Rep.', 'Korea Rep.', 'Kosovo', 'Kuwait',
       'Kyrgyz Republic', 'Lao PDR', 'Latvia', 'Lebanon', 'Lesotho',
       'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 'Luxembourg',
       'Macao SAR China', 'Macedonia FYR', 'Madagascar', 'Malawi',
       'Malaysia', 'Maldives', 'Mali', 'Malta', 'Marshall Islands',
       'Mauritania', 'Mauritius', 'Mexico', 'Micronesia Fed.Sts.',
       'Moldova', 'Monaco', 'Mongolia', 'Montenegro', 'Morocco',
       'Mozambique', 'Myanmar', 'Namibia', 'Nepal', 'Netherlands',
       'New Caledonia', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria',
       'Northern Mariana Islands', 'Norway', 'Oman', 'Pakistan', 'Palau',
       'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines',
       'Poland', 'Portugal', 'PuertoRico', 'Qatar', 'Romania',
       'Russian Federation', 'Rwanda', 'Samoa', 'SanMarino',
       'Sao Tome and Principe', 'SaudiArabia', 'Senegal', 'Serbia',
       'Seychelles', 'SierraLeone', 'Singapore',
       'Sint Maarten (Dutchpart)', 'Slovak Republic', 'Slovenia',
       'Solomon Islands', 'Somalia', 'South Africa', 'South Sudan',
       'Spain', 'Sri Lanka', 'St.Kitts and Nevis', 'St.Lucia',
       'St.Martin (Frenchpart)', 'St.Vincent and the Grenadines', 'Sudan',
       'Suriname', 'Swaziland', 'Sweden', 'Switzerland',
       'Syrian Arab Republic', 'Tajikistan', 'Tanzania', 'Thailand',
       'Timor-Leste', 'Togo', 'Tonga', 'Trinidad and Tobago', 'Tunisia',
       'Turkey', 'Turkmenistan', 'Turks and Caicos Islands', 'Tuvalu',
       'Uganda', 'Ukraine', 'United Arab Emirates', 'United Kingdom',
       'United States', 'Uruguay', 'Uzbekistan', 'Vanuatu',
       'Venezuela RB', 'Vietnam', 'West Bank and Gaza', 'Yemen Rep.',
       'Zambia', 'Zimbabwe', 'Taiwan'
))


#st.write(country_name)

#classifier_name = st.sidebar.selectbox("Select Classifier",("XGBoost", "Random Forest",  "Logistic Regression"))
classifier_name = "XGBoost"

#classifier_name = st.sidebar.selectbox("Select Classifier",("Logistic Regression", "Random Forest", "XGBoost",  "Naive Bayes", "KNN"))
#classifier_name = st.sidebar.selectbox("Select Classifier",("Logistic Regression", "Random Forest", "XGBoost"))

#classifier_name = "Logistic Regression"
#classifier_name = "Random Forest"

#Slider
#future_year = st.sidebar.slider("Year", 2021, 2025)
##########

#default number per year
h = pd.read_csv("Data_number_of_defaults.csv") 
h.drop(h.filter(regex="Unname"),axis=1, inplace=True)

plt.plot(h['Year'], h['Number_of_Defaults']) #plot of number of defaults per year
plt.xlabel("Year")
plt.ylabel("Number of Defaults")
plt.title("Number of Defaults per Year")

st.pyplot()

###########



#########
data = pd.read_csv("Data_default_and_economics_3.csv") # Merged Dataset
data.drop(data.filter(regex="Unname"),axis=1, inplace=True) # Drop 'Unname' column from dataset
data.head()


st.write("""
# This model is based on macro-economic quantities
""")
st.write(""" #  Model Variables and their Statistics """)
st.write(data.describe().transpose()) # Preference to transpose rows and columns

#####
st.write(""" #  Variable Correlations """)
st.write(data.corr())
############

#data1 = data.drop('Year', axis = 1) # drop year
#data1 = data.drop('default_probability', axis = 1)
#corr = data1.corr()
#corr.style.background_gradient(cmap='coolwarm').set_precision(3)
#st.pyplot()
###########

# histograms:
st.write(""" #  """)
st.write(""" #  Variable Histograms """)
plt.subplots(4, 3, figsize=(20, 12), sharex=True) #9 plots of Features shown. 5x2 matrix


plt.rcParams.update({'font.size': 10})

plt.subplot(4, 3, 1)
plt.hist(data['log_gdp_per_capita']) 
plt.xlabel("log(GDP per Capita)") # X label = feature
plt.ylabel("Count") # Y label = Count for each interval
plt.yscale('log') #Count is logarithmic. 10^n


plt.subplot(4, 3, 2)
plt.hist(data['annual_inflation'])
plt.xlabel("Annual Inflation")
plt.ylabel("Count")
plt.yscale('log')


plt.subplot(4, 3, 3)
plt.hist(data['change_in_volume_of_imports'])
plt.xlabel("Imports Volume Change")
plt.ylabel("Count")
plt.yscale('log')


plt.subplot(4, 3, 4)
plt.hist(data['change_in_volume_of_exports'])
plt.xlabel("Exports Volume Change")
plt.ylabel("Count")
plt.yscale('log')


plt.subplot(4, 3, 5)
plt.hist(data['unemployment_rate'])
plt.xlabel("Unemployment Rate")
plt.ylabel("Count")
plt.yscale('log')

plt.subplot(4, 3, 6)
plt.hist(data['gov_revenue_per_GDP'])
plt.xlabel("Gov. Revenue over GDP")
plt.ylabel("Count")
plt.yscale('log')

plt.subplot(4, 3, 7)
plt.hist(data['log_gov_expenditure_per_GDP'])
plt.xlabel("Log(Gov. Expenditure over GDP)")
plt.ylabel("Count")
plt.yscale('log')

plt.subplot(4, 3, 8)
plt.hist(data['gov_lending_minus_borrowing_per_GDP'])
plt.xlabel("Gov. Lending minus Borrowing")
plt.ylabel("Count")
plt.yscale('log')

plt.subplot(4, 3, 9)
plt.hist(data['gov_net_debt_per_GDP'])
plt.xlabel("Gov. Debt over GDP")
plt.ylabel("Count")
plt.yscale('log')


plt.subplot(4, 3, 10)
plt.hist(data['exports_minus_imports_per_GDP'])
plt.xlabel("Exports minus Imports over GDP")
plt.ylabel("Count")
plt.yscale('log')

plt.subplot(4, 3, 11)
plt.hist(data['log10_share_world_GDP'])
plt.xlabel("Log(share of world GDP)")
plt.ylabel("Count")
plt.yscale('log')


plt.subplots_adjust(top=0.90, bottom=0.02, wspace=0.20, hspace=0.4) # margins
#sns.set()    
#plt.show()
st.pyplot()
###########
# largest debtors by percentage of GDP
st.write(""" #  """)
st.write(""" #  Largest Debtors as Percentage of GDP in 2020 """)
df1 = data[data['Year'] == 2020].sort_values(by=['gov_net_debt_per_GDP'], ascending = False)
df1 = df1[['Country_Name', 'gov_net_debt_per_GDP']]
df1 = df1.reset_index(drop=True)
st.write(df1.head(10))

st.write(""" #  """)
st.write(""" #  Smallest Debtors as Percentage of GDP in 2020 """)
# smallest debtors by percentage of GDP
st.write(df1.tail(10))

##########
st.write(""" #  """)
st.write(""" #  Model Computations using XGBoost """)


X = data[['log_gdp_per_capita','annual_inflation' ,   \
   'change_in_volume_of_imports',  'change_in_volume_of_exports', 'unemployment_rate',  \
   'gov_revenue_per_GDP', 'log_gov_expenditure_per_GDP', 'gov_lending_minus_borrowing_per_GDP',\
          'gov_net_debt_per_GDP', 'exports_minus_imports_per_GDP', 'log10_share_world_GDP']]  




y = data['default_flag']
y=y.astype('int')



X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state = 0) #test_size = 0.30
if (classifier_name == "Logistic Regression"):
 model = LogisticRegression()
elif (classifier_name == "Random Forest"):
 from sklearn.ensemble import RandomForestClassifier
 model = RandomForestClassifier()
elif (classifier_name == "XGBoost"):
 import xgboost as xgb
 from xgboost import XGBClassifier
 model = XGBClassifier()
#elif (classifier_name == "Naive Bayes"):
# from sklearn.naive_bayes import GaussianNB
# model = GaussianNB()
 
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

if (classifier_name == "Logistic Regression"):
 st.write("Logistic Regression Coefficients:" , model.coef_)
 st.write("Logistic Regression Intercept:" , model.intercept_)

st.write("Accuracy:",metrics.accuracy_score(y_test, y_pred))
st.write("Precision:",metrics.precision_score(y_test, y_pred))
st.write("Recall:",metrics.recall_score(y_test, y_pred))
#st.write("Accuracy_Score:",accuracy_score(y_test,y_pred))
st.write("Classification Report", classification_report(y_test,y_pred))



st.write(confusion_matrix(y_test,y_pred))

data['default_probability'] = 100.*model.predict_proba(X)[:, 1]
#data.head()

from sklearn import metrics
y_pred_proba = model.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba) #AuC originally worse because of small default dataset (200 defaults)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc=" + repr(auc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
#plt.rc('font', size = MEDIUM_SIZE)
#plt.rc('legend', size = MEDIUM_SIZE)
plt.legend(loc=4)
#plt.show()
st.pyplot()

st.write("AUC = ", auc)

st.title("-----------------------------------------------")
st.title("Web App Functionality: Calculate Probability of Default. ")
st.write("""
# In the sidebar, user can select inputs:
""")
st.write("""
## - Choose country at the top of the sidebar
""")
st.write("""
## - Adjust percentage (%) change in features
""")
st.write("""
## - Use slider to adjust year to use IMF forecasts
""")
st.title("-----------------------------------------------")

st.write(""" # Past Probability of Default of the Selected Country per Year""")
df1 = data[data['Country_Name'] == country_name ]
#df1.head()



#st.write("Probability of Default for " + country_name)
plt.plot(df1['Year'], df1['default_probability']) #1980s economic crisis in South America, Argentina economic crisis around 2003
plt.title("Probability of Default for " + country_name)
plt.xlabel("Year")
plt.ylabel("Probability of Default")
#plt.xlim(left=1990)
#plt.ylim(top=25)

st.pyplot()


##############################################
st.title("-----------------------------------------------")
#st.write(""" ## Change the variable values to calculate the """)

st.write(""" #  """)
st.write(""" #  Probability of Default of the Selected Country in 2021 Using Input Quantities""")
selected_country = country_name

f1 = st.sidebar.number_input("Log10 GDP per capita change (%)")
f2 = st.sidebar.number_input("Annual inflation (%)")
f3 = st.sidebar.number_input("Change in volume of imports (%)")
f4 = st.sidebar.number_input("Change in volume of_exports (%)")
f5 = st.sidebar.number_input("Unemployment rate (%)")
f6 = st.sidebar.number_input("Gov revenue per GDP (%)")
f7 = st.sidebar.number_input("Log gov expenditure per GDP  (%)")
f8 = st.sidebar.number_input("Gov lending minus borrowing per GDP (%)")
f9 = st.sidebar.number_input("Gov net debt per GDP (%)")
f10 = st.sidebar.number_input("Exports minus imports per GDP (%)")
f11 = st.sidebar.number_input("Log10 share world GDP (%)")

fut = pd.read_csv("Data_default_and_economics_3.csv") 
fut.drop(data.filter(regex="Unname"),axis=1, inplace=True)
fut.head()

fut_df = fut[fut['Year'] == 2020]
fut_df = fut_df[fut_df['Country_Name'] == selected_country]
fut_df = fut_df.reset_index(drop=True)
fut_df.drop(fut_df.filter(regex="Unname"),axis=1, inplace=True)
save_fut_df = fut_df
fut_df = fut_df.drop('Country_Name',  axis = 1)
fut_df = fut_df.drop('Country_code',  axis = 1)
fut_df = fut_df.drop('Year', axis = 1)
fut_df = fut_df.drop('default_flag', axis = 1)
fut_df.tail()



factor_log_gdp_per_capita = f1                
factor_annual_inflation = f2                   
factor_change_in_volume_of_imports = f3        
factor_change_in_volume_of_exports = f4           
factor_unemployment_rate = f5                   
factor_gov_revenue_per_GDP = f6                  
factor_log_gov_expenditure_per_GDP = f7          
factor_gov_lending_minus_borrowing_per_GDP = f8
factor_gov_net_debt_per_GDP   = f9                
factor_exports_minus_imports_per_GDP = f10          
factor_log10_share_world_GDP = f11                


fut_df['log_gdp_per_capita'] = fut_df['log_gdp_per_capita']*(100. + factor_log_gdp_per_capita)/100.
fut_df['annual_inflation'] = fut_df['annual_inflation']*(100. + factor_annual_inflation)/100.
fut_df['change_in_volume_of_imports'] = fut_df['change_in_volume_of_imports']*(100. + factor_change_in_volume_of_imports)/100.
fut_df['change_in_volume_of_exports'] = fut_df['change_in_volume_of_exports']*(100. + factor_change_in_volume_of_exports)/100.
fut_df['unemployment_rate'] = fut_df['unemployment_rate']*(100. + factor_unemployment_rate)/100.
fut_df['gov_revenue_per_GDP'] = fut_df['gov_revenue_per_GDP']*(100. + factor_gov_revenue_per_GDP)/100.
fut_df['log_gov_expenditure_per_GDP'] = fut_df['log_gov_expenditure_per_GDP']*(100. + factor_log_gov_expenditure_per_GDP)/100.
fut_df['gov_lending_minus_borrowing_per_GDP'] = fut_df['gov_lending_minus_borrowing_per_GDP']*(100. + factor_gov_lending_minus_borrowing_per_GDP)/100.
fut_df['gov_net_debt_per_GDP'] = fut_df['gov_net_debt_per_GDP']*(100. + factor_gov_net_debt_per_GDP)/100.
fut_df['exports_minus_imports_per_GDP'] = fut_df['exports_minus_imports_per_GDP']*(100. + factor_exports_minus_imports_per_GDP)/100.
fut_df['log10_share_world_GDP'] = fut_df['log10_share_world_GDP']*(100. + factor_log10_share_world_GDP)/100.


new_fut_df = fut_df.values.reshape(1,11) # creating numpy array
pred_df = 100.*(1. - model.predict_proba(new_fut_df))  # calculating probability of default
new_prob_df = pd.DataFrame({'Default_Probability': pred_df[:, 0]}) # creating dataframe with column name
result = save_fut_df.join(new_prob_df, how='inner') # joining the probability of default with original dataframe
result.drop(data.filter(regex="Unname"),axis=1, inplace=True)
new_result = result.sort_values(by=['Default_Probability'], ascending = False)
new_result = new_result.reset_index(drop=True)
new_result['Year'] = new_result['Year'] + 1

new_result = new_result.drop('log_gdp_per_capita',  axis = 1)
new_result = new_result.drop('annual_inflation',  axis = 1)
new_result = new_result.drop('change_in_volume_of_imports',  axis = 1)
new_result = new_result.drop('change_in_volume_of_exports',  axis = 1)
new_result = new_result.drop('unemployment_rate',  axis = 1)
new_result = new_result.drop('gov_revenue_per_GDP',  axis = 1)
new_result = new_result.drop('log_gov_expenditure_per_GDP',  axis = 1)
new_result = new_result.drop('gov_lending_minus_borrowing_per_GDP',  axis = 1)
new_result = new_result.drop('gov_net_debt_per_GDP',  axis = 1)
new_result = new_result.drop('exports_minus_imports_per_GDP',  axis = 1)
new_result = new_result.drop('log10_share_world_GDP',  axis = 1)
new_result = new_result.drop('default_flag',  axis = 1)

st.write(new_result.head())  

############################################


# predicting probabilities of default in a given year
def predicting_probability_of_default(a_year):
 fut = pd.read_csv("Data_predicted_economic_quantities_3.csv") 
 fut.drop(data.filter(regex="Unname"),axis=1, inplace=True)
 fut_df = fut[fut['Year'] == a_year]
 fut_df = fut_df.reset_index(drop=True)
 fut_df.drop(fut_df.filter(regex="Unname"),axis=1, inplace=True)
 save_fut_df = fut_df
 fut_df = fut_df.drop('Country_Name',  axis = 1)
 fut_df = fut_df.drop('Country_code',  axis = 1)
 fut_df = fut_df.drop('Year', axis = 1)
 new_fut_df = fut_df.values.reshape(212,11) # creating numpy array
 pred_df = 100.*(1. - model.predict_proba(new_fut_df))  # calculating probability of default
 new_prob_df = pd.DataFrame({'Default_Probability': pred_df[:, 0]}) # creating dataframe with column name
 result = save_fut_df.join(new_prob_df, how='inner') # joining the probability of default with original dataframe
 result.drop(data.filter(regex="Unname"),axis=1, inplace=True)
 new_result = result.sort_values(by=['Default_Probability'], ascending = False)
 new_result = new_result.reset_index(drop=True)
 return (new_result)

st.title("-----------------------------------------------")
st.write(""" #  """)
st.write(""" # Predictions Using the Forecasts from the IMF for the Selected Year """)

future_year = st.sidebar.slider("Year", 2021, 2025)
my_year = future_year
st.write(""" ## Countries most likely to default """)
st.write("year = ", my_year)
prob_result = predicting_probability_of_default(my_year)
st.write(prob_result.head(20))

st.write(""" ## Countries least likely to default """)
st.write("year = ", my_year)
st.write(prob_result.tail(20))



st.write(""" ## Reference: This Time is Different, by C. M. Reinhart and K. S. Rogoff, Princeton University Press, 2009. """)
