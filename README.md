# Rossmann Sales Prediction
## 
![csm_rossmann_shop_foto_stralsund_1633a5fb67](https://user-images.githubusercontent.com/68538809/148125358-4b61a4af-bc42-4901-8633-49aa49a984b5.png)

[![python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

# 1. Business Context.

**Dirk Rossmann GmbH is one of the largest drug store chains in Europe with around 56,200 employees and over €10 billion annual revenue** in Germany, Poland, Hungary, the Czech Republic, Turkey, Albania, Kosovo, and Spain. 

The scope of this project is based on a CFO demand: stores will be renovated, and for that, the budget needs to be aligned with each store's sales. Adding to that, it was requested to predict each store's sales for the following 6 weeks.

In this context, **this project intends to provide the management team with a 6-week store sales prediction.**

The solution will be available as a Telegram bot, which can be accessed by clicking below:

[<img alt="Telegram" src="https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white"/>](https://t.me/rossmann_bernardo_bot)
## 1.1 Project Deliverables.

In this business context, as a data scientist, I intend to develop a machine learning model that predicts the 6 weeks sales per store.
 
With this model, the management team (including the CFO) will be able to manage the stores renovations.

A report with the following analysis is the deliverable goals:

**1.** Key findings on stores most relevant attributes.

**2.** Stores revenue prediction for the upcoming 6 weeks.

**3.** Create online bot that automatically generates specific store revenue prediction based on request.

# 2. The Solution

## Solution strategy

The following strategy was used to achieve the goal:

### Step 1. Data Description

The initial dataset has 1.017.209 rows and 18 columns. Follows the features description:

- **Store**: Unique Id for each store   
- **Sales**: Turnover for any given day 
- **Customers**: Number of customers on a given day  
- **Open**: Indicator for whether the store was open: 0 = closed, 1 = open  
- **StateHoliday**: Indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. a = Public holiday, b = Easter holiday, c = Christmas, 0 = None 
- **SchoolHoliday**: Indicates if the (Store, Date) was affected by the closure of public schools
- **StoreType**: Differentiates between 4 different store models: a, b, c, d
- **Assortment**: Describes an assortment level: a = basic, b = extra, c = extended
- **CompetitionDistance**: Distance in meters to the nearest competitor store 
- **CompetitionOpenSince [Month/Year]**: Gives the approximate year and month of the time the nearest competitor was opened 
- **Promo**: Indicates whether a store is running a promo on that day
- **Promo2**: A continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating
- **Promo2Since[Year/Week]**: Describes the year and calendar week when the store started participating in Promo2  
- **PromoInterval**: The consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store

**Statistic analysis:**

***Numerical attributes***

 ![numericalfeatures](https://user-images.githubusercontent.com/68538809/151344296-99fefa06-9864-4941-a367-ef5ab24afd74.png)

***Categorical attributes*** 

![categorical features](https://user-images.githubusercontent.com/68538809/151350651-ec1d4d4a-cd01-4e99-8ea5-2fb668c7a6e4.png)

### Step 2. Feature Engineering

On feature creation, **11 columns were created and 2 were modified**:

**Based on** the attribute **'Date'**, 5 features were derived: 'year', 'month', 'day', 'week_of_year' and 'year_week'

**Based on** the attribute **'competition_open_since_year'** 2 features were derived: 'competition_since' and 'competition_time_month'

**Based on** the attribute **'promo_since'** 3 features were derived: 'promo_since', 'promo_time_week' and 'promo_time_day'

**Based on** the attributed **'promo_interval' and 'month_map'** 1 feature was created: 'is_promo'

**'assortment' and 'state_holiday' were modified**, the values were changed to a descriptive format

### Step 3. Data Filtering

Two types of data filtering were used:

- **Rows Filtering**: I have filtered data based on open days in which sales were made as this is a sales analysis.
- **Columns Selection**: I have dropped 4 columns. 'Customers' as they would not be part of a prediction dataset. 'Open' as this column is no longer useful after filtering based on exclusively open days. 'Promo_interval' and  'Month_map' were useful to derive new features but give no additional inputs to the model.

### Step 4. Exploratory Data Analysis (EDA)

On Exploratory Data Analysis, Univariate, Bivariate, and Multivariates study was performed to help understand the statistical properties of each attribute, correlations, and hypothesis testing.

### Step 5. Data Preparation

In this section, Rescaling (Robust Scaler and MinMax Scaler), Encoding (One Hot, Label and Ordinal Encoding), Variable Transformation (Logarithmic Transformation), and Nature Transformations of the variables were carried out.

### Step 6. Feature Selection

To select the features to be used, two methods were used:

**1.** Application of Boruta using RandomForestRegressor method

**2.** EDA insights.

**From Boruta Feature Selection:** 'store', 'promo', 'store_type', 'assortment', 'competition_distance', 'competition_open_since_month', 'competition_open_since_year', 'promo2', 'promo2_since_week', 'promo2_since_year', 'competition_time_month', 'promo_time_week', 'day_of_week_sin', 'day_of_week_cos', 'month_cos', 'day_sin', 'day_cos', week_of_year_cos'

**From EDA:** Features selected from Boruta go in accordance with EDA

In addition, 16 features have not been selected, either for lack of importance or for being used previously to derive new features.

### Step 7. Machine Learning Modelling

Machine learning models were trained and passed through Cross-Validation to evaluate the "real" performance. 

### Step 8. Hyperparameter Fine Tuning

Based on Machine Learning Modelling results, the best model was chosen and submitted to Hyperparameter Fine Tuning to optimize its performance.

### Step 9. Error Translation and Interpretation

In this step I have transformed the performance of the Machine Learning model into a business result. The prediciton of the overall stores revenue was presented along with the best and worst scenarios. 

### Step 10. Model Deploy

The model was published to a cloud environment (Heroku) so that others can use the results to improve and implement business decisions.

### Step 11. Telegram Bot Creation

A Telegram Bot was created, for a more user-friendly and faster access to the predicitons at any time. Please check https://t.me/rossmann_bernardo_bot

# 3. Machine Learning Model Applied

The following Machine Learning models were tested and cross-validated:

- **Linear Regression**
- **Linear Regression Regularized - LASSO**
- **Random Forest**
- **XGBoost Regressor**

# 4. Machine Learning Model Performance

To evaluate the performance of the models, 3 metrics were used:

- **Mean Absolute Error (MAE)**
- **Mean Absolute Percentage Error (MAPE)**
- **Root Mean Square Error (RMSE)**


The indicators "MAE" and "MAPE" were the ones to assume the major importance as it represents the difference between the predicted and the real values. 

The following table discloses the cross-validated ("real") performance of all the models used:

|       Model Name          |        MAE CV       |     MAPE CV    |      RMSE CV       |
|:-------------------------:|:-------------------:|:--------------:|:------------------:|
| XGBoost Regressor          |  1096.96 +/- 179.03   | 0.15 +/- 0.02	 | 1578.28 +/- 246.71 |
| Linear Regression Regularized Model - Lasso                     |  2117.23 +/- 342.11	   | 2117.23 +/- 342.11 | 3058.18 +/- 504.81 |
| Linear Regression     |  2079.95 +/- 303.16	    | 0.29 +/- 0.01	 | 2958.57 +/- 473.83 |
| Random Forest Regressor              |  843.13 +/- 225.18 | 0.12 +/- 0.02 | 2958.57 +/- 473.83 |

Although Random Forest Regressor holds the best MAE and MAPE result, **"XGBoost Regressor"** will be the choosen model due to it's perfomance, low-memory usage and high fine-tunning possibilities.


# 5. Business Results

## **1.** Key findings on stores most relevant attributes.

### Insight 1. Competition Distance
> Stores with closer competition sell more than stores without close competition. Plots show how significant is the competition distance feature when plotted with sales. 

![insight1](https://user-images.githubusercontent.com/68538809/151353116-74b6fd51-eb6f-482b-8f12-ccaa3283468d.png)

### Insight 2. Holidays
> When comparing Christmas, Easter, and Public Holidays, both Christmas and Easter holidays have a higher impact on sales. Although there is no data for 2015 Christmas, based on 2013 and 2014, easter and Christmas are the holidays that bring on average most revenue each store 

![insight21](https://user-images.githubusercontent.com/68538809/151353757-ebe02fdc-6d40-4014-ba93-45940860a53d.png)
![insight22](https://user-images.githubusercontent.com/68538809/151353783-48399572-4aac-43c5-915f-96793ecbba02.png)

### Insight 3. 10 first days of the Month
> The 10 first days of the month are the ones that on average bring more revenue to the stores, there is an inverse correlation between sales and month days. Although the first day of the month seems to be low on sales, the following 9 are high in sales.

![insight3](https://user-images.githubusercontent.com/68538809/151354957-a4a313bd-6f41-4617-8902-9aa658781ce5.png)

### Insight 4. Assortment
> Based on the analysis of the types of assortment, basic, extended and extra, sales are on average much higher on stores with the assortment extra

![insight41](https://user-images.githubusercontent.com/68538809/151356993-b62ee4b5-b83f-485b-a441-6b64905d9e82.png)
![insight42](https://user-images.githubusercontent.com/68538809/151357016-856b907e-aafd-4dc9-985f-5e22f916abb4.png)
![insight43](https://user-images.githubusercontent.com/68538809/151357126-d98f757d-a5b2-44f5-8a94-9711df083d09.png)

## **2.** Stores revenue prediction for the upcoming 6 weeks.


## **3.** Create online bot that automatically generates specific store revenue prediction based on request.

# 6. Conclusions

# 7. Next Steps to Improve

