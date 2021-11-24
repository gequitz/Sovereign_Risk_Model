# Sovereign_Risk_Model
Final project for Metis (Project 5: Passion Project): Sovereign Risk Modeling

By: Gabriel Equitz
____________________________________________________________________________

## Problem Statement
- Governments of countries often borrow money, mainly through issuing bonds. These bonds are sold to investors. 
- Associated with each bond there is an interest rate, which is paid periodically, with the principal paid after a certain number of years.
- However, some countries fail to pay back these loans, and go into Sovereign Default. This can cost bond holders up to billions of dollars.

- Bonds have an interest rate associated with them. The larger the probability of default, the the larger the interest rate is expected to be. 

- This is a model is designed to calculate the probability of sovereign default in the next few years, by analyzing macroeconomic data.

This model tries to do things other Sovereign Risk models have not done. This is an open source model, unlike other models on Sovereign Default. In contrast to other models, this project has a free web app hosted on Heroku, in which users can adjust inputs to see how economic features change probability of default for given countries and years.

## Web App link
[MySovereignRiskApp](https://mysovereignriskapp.herokuapp.com/)


## Data Sources
- Data Sources : Detailed Economic data 1980-2020 (By Country dataset)  https://www.imf.org/en/Publications/WEO/weo-database/2020/October/download-entire-database
- Projections for 2021-2025 available, put in separate CSV
- Sovereign Default, Restructuring and agreements under the auspices of the Paris Club datasets (since 1980): https://sites.google.com/site/christophtrebesch/data


## Methodology
- Binary classification method is used, which allows to calculate the probability of default.
- Select macroeconomic features that show potential for predictive value.
- Partitioned data into training and testing datasets (70/30 split).
- Analyze each of these individual variables and select 11 that had an AUC greater than (>) 0.5 when using XGBoost.
- Combined all the variables that passed the above test to form the set of independent variables of the model.
- Tested several methods to compare with with XGBoost, such as Logistic Regression, Random Forest, and Naive Bayes.


## Files
- 'streamlit_sovereign_risk_model.py' is a file that contains Streamlit web app code
- 'Sovereign_Risk_9.ipynb' contains main project code in a Jupyter Notebook
- 'Project 5 Sovereign Risk Final Presentation Final.pdf' is a PDF of presentation slides
- 'Project 5 Sovereign Risk Final Presentation Final.pptx' is a Powerpoint version of presentation slides
- 'Data_default_and_economics_3' contains detailed macroeconomic data from the IMF from 1980-2020
- 'Data_number_of_defaults' contains data on all Sovereign Defaults, Restructurings, and Paris Club agreements from 1980-2020 (restructurings and Paris Club agreements are counted as defaults in this model)
- 'Data_predicted_economic_quantities_3' contains IMF Macroeconomic Projections for 2021-2025


## Findings
- XGBoost is determined to be the best method, having the highest values for metrics I am interested in: Precision, Recall, AUC (because the data is unbalanced, Accuracy is less important).
- AUC = 0.84, Precision = 0.43, Recall = 0.12
- Debt over GDP, Annual Inflation, and Unemployment Rate are Positively (+) correlated with probability of sovereign default
- GDP per Capita, Change in Imports, Change in Export, Gov Revenue/GDP, Gov Expenditure/ GDP , Gov Lending minus Borrowing, Exports minus Imports , and Share of the World GDP are Negatively (-) correlated with probability of sovereign default.
- The 2 most important features of the model, as measured by the AUC, are GDP per Capita and Net Debt/GDP
- Time series plots show peaks in the probability of default as default approaches.



## Libraries, Packages, and Notebook used
- [Streamlit](https://streamlit.io/)
- [Heroku](https://www.heroku.com/)
- [Numpy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Tableau](https://www.tableau.com/)
- [Jupyter](https://jupyter.org/)

## Blog link
- https://medium.com/@gabrielequitz/modeling-sovereign-risk-with-data-science-free-web-app-available-1d3ac09fa66a
