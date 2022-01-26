# Rossmann Sales Prediction
## 
![csm_rossmann_shop_foto_stralsund_1633a5fb67](https://user-images.githubusercontent.com/68538809/148125358-4b61a4af-bc42-4901-8633-49aa49a984b5.png)

[![python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

# 1. Business Context.

Dirk Rossmann GmbH is one of the largest drug store chains in Europe with around 56,200 employees and over €10 billion annual revenue in Germany, Poland, Hungary, the Czech Republic, Turkey, Albania, Kosovo and Spain. 

The scope of this project is based on a CFO demand: stores will be renovated, and for that, the budget needs to be aligned with each store's sales. Adding to that, it was requested to predict each store sales for the following 6 weeks.

In this context, this project intends to provide the management team with a 6 week store sales prediciton.

The solution will be available as a Telegram bot, which can be accessed by clicking below:

[![forthebadge](data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMTEuNDciIGhlaWdodD0iMzUiIHZpZXdCb3g9IjAgMCAyMTEuNDcgMzUiPjxyZWN0IGNsYXNzPSJzdmdfX3JlY3QiIHg9IjAiIHk9IjAiIHdpZHRoPSI5My45MSIgaGVpZ2h0PSIzNSIgZmlsbD0iIzMxQzRGMyIvPjxyZWN0IGNsYXNzPSJzdmdfX3JlY3QiIHg9IjkxLjkxIiB5PSIwIiB3aWR0aD0iMTE5LjU2IiBoZWlnaHQ9IjM1IiBmaWxsPSIjMzg5QUQ1Ii8+PHBhdGggY2xhc3M9InN2Z19fdGV4dCIgZD0iTTE3LjMzIDIyTDE0LjIyIDIyTDE0LjIyIDEzLjQ3TDE3LjE0IDEzLjQ3UTE4LjU5IDEzLjQ3IDE5LjM0IDE0LjA1UTIwLjEwIDE0LjYzIDIwLjEwIDE1Ljc4TDIwLjEwIDE1Ljc4UTIwLjEwIDE2LjM2IDE5Ljc4IDE2LjgzUTE5LjQ3IDE3LjMwIDE4Ljg2IDE3LjU2TDE4Ljg2IDE3LjU2UTE5LjU1IDE3Ljc1IDE5LjkzIDE4LjI2UTIwLjMxIDE4Ljc4IDIwLjMxIDE5LjUxTDIwLjMxIDE5LjUxUTIwLjMxIDIwLjcxIDE5LjUzIDIxLjM2UTE4Ljc2IDIyIDE3LjMzIDIyTDE3LjMzIDIyWk0xNS43MCAxOC4xNUwxNS43MCAyMC44MkwxNy4zNSAyMC44MlExOC4wNCAyMC44MiAxOC40NCAyMC40N1ExOC44MyAyMC4xMyAxOC44MyAxOS41MUwxOC44MyAxOS41MVExOC44MyAxOC4xOCAxNy40NyAxOC4xNUwxNy40NyAxOC4xNUwxNS43MCAxOC4xNVpNMTUuNzAgMTQuNjZMMTUuNzAgMTcuMDZMMTcuMTUgMTcuMDZRMTcuODQgMTcuMDYgMTguMjMgMTYuNzVRMTguNjIgMTYuNDMgMTguNjIgMTUuODZMMTguNjIgMTUuODZRMTguNjIgMTUuMjMgMTguMjYgMTQuOTVRMTcuOTAgMTQuNjYgMTcuMTQgMTQuNjZMMTcuMTQgMTQuNjZMMTUuNzAgMTQuNjZaTTI0LjY0IDE5LjE2TDI0LjY0IDE5LjE2TDI0LjY0IDEzLjQ3TDI2LjEyIDEzLjQ3TDI2LjEyIDE5LjE4UTI2LjEyIDIwLjAzIDI2LjU1IDIwLjQ4UTI2Ljk4IDIwLjkzIDI3LjgzIDIwLjkzTDI3LjgzIDIwLjkzUTI5LjU0IDIwLjkzIDI5LjU0IDE5LjEzTDI5LjU0IDE5LjEzTDI5LjU0IDEzLjQ3TDMxLjAyIDEzLjQ3TDMxLjAyIDE5LjE3UTMxLjAyIDIwLjUzIDMwLjE1IDIxLjMyUTI5LjI4IDIyLjEyIDI3LjgzIDIyLjEyTDI3LjgzIDIyLjEyUTI2LjM2IDIyLjEyIDI1LjUwIDIxLjMzUTI0LjY0IDIwLjU1IDI0LjY0IDE5LjE2Wk0zNy4xNSAyMkwzNS42NyAyMkwzNS42NyAxMy40N0wzNy4xNSAxMy40N0wzNy4xNSAyMlpNNDcuMzIgMjJMNDEuOTYgMjJMNDEuOTYgMTMuNDdMNDMuNDQgMTMuNDdMNDMuNDQgMjAuODJMNDcuMzIgMjAuODJMNDcuMzIgMjJaTTUzLjEyIDE0LjY2TDUwLjQ5IDE0LjY2TDUwLjQ5IDEzLjQ3TDU3LjI1IDEzLjQ3TDU3LjI1IDE0LjY2TDU0LjU5IDE0LjY2TDU0LjU5IDIyTDUzLjEyIDIyTDUzLjEyIDE0LjY2Wk02OC41NSAyMkw2Ny4wNyAyMkw2Ny4wNyAxMy40N0w2OC41NSAxMy40N0w2OC41NSAyMlpNNzQuODQgMjJMNzMuMzYgMjJMNzMuMzYgMTMuNDdMNzQuODQgMTMuNDdMNzguNjYgMTkuNTRMNzguNjYgMTMuNDdMODAuMTMgMTMuNDdMODAuMTMgMjJMNzguNjUgMjJMNzQuODQgMTUuOTVMNzQuODQgMjJaIiBmaWxsPSIjRkZGRkZGIi8+PHBhdGggY2xhc3M9InN2Z19fdGV4dCIgZD0iTTEwNy44OSAxNS40OEwxMDUuMzEgMTUuNDhMMTA1LjMxIDEzLjYwTDExMi44MyAxMy42MEwxMTIuODMgMTUuNDhMMTEwLjI2IDE1LjQ4TDExMC4yNiAyMkwxMDcuODkgMjJMMTA3Ljg5IDE1LjQ4Wk0xMjMuOTUgMjJMMTE3LjIwIDIyTDExNy4yMCAxMy42MEwxMjMuNzkgMTMuNjBMMTIzLjc5IDE1LjQ0TDExOS41NiAxNS40NEwxMTkuNTYgMTYuODVMMTIzLjI5IDE2Ljg1TDEyMy4yOSAxOC42M0wxMTkuNTYgMTguNjNMMTE5LjU2IDIwLjE3TDEyMy45NSAyMC4xN0wxMjMuOTUgMjJaTTEzNS4xNCAyMkwxMjguNzUgMjJMMTI4Ljc1IDEzLjYwTDEzMS4xMyAxMy42MEwxMzEuMTMgMjAuMTFMMTM1LjE0IDIwLjExTDEzNS4xNCAyMlpNMTQ2LjMwIDIyTDEzOS41NSAyMkwxMzkuNTUgMTMuNjBMMTQ2LjE1IDEzLjYwTDE0Ni4xNSAxNS40NEwxNDEuOTEgMTUuNDRMMTQxLjkxIDE2Ljg1TDE0NS42NCAxNi44NUwxNDUuNjQgMTguNjNMMTQxLjkxIDE4LjYzTDE0MS45MSAyMC4xN0wxNDYuMzAgMjAuMTdMMTQ2LjMwIDIyWk0xNTAuNjggMTcuODBMMTUwLjY4IDE3LjgwUTE1MC42OCAxNi41NCAxNTEuMjcgMTUuNTRRMTUxLjg3IDE0LjU1IDE1Mi45NCAxMy45OVExNTQuMDEgMTMuNDMgMTU1LjM1IDEzLjQzTDE1NS4zNSAxMy40M1ExNTYuNTMgMTMuNDMgMTU3LjQ3IDEzLjgzUTE1OC40MCAxNC4yMiAxNTkuMDMgMTQuOTdMMTU5LjAzIDE0Ljk3TDE1Ny41MSAxNi4zM1ExNTYuNjcgMTUuNDAgMTU1LjQ5IDE1LjQwTDE1NS40OSAxNS40MFExNTUuNDggMTUuNDAgMTU1LjQ3IDE1LjQwTDE1NS40NyAxNS40MFExNTQuNDAgMTUuNDAgMTUzLjczIDE2LjA2UTE1My4wNyAxNi43MSAxNTMuMDcgMTcuODBMMTUzLjA3IDE3LjgwUTE1My4wNyAxOC41MCAxNTMuMzcgMTkuMDRRMTUzLjY4IDE5LjU5IDE1NC4yMiAxOS44OVExNTQuNzUgMjAuMjAgMTU1LjQ1IDIwLjIwTDE1NS40NSAyMC4yMFExNTYuMTQgMjAuMjAgMTU2LjczIDE5LjkzTDE1Ni43MyAxOS45M0wxNTYuNzMgMTcuNjJMMTU4LjgzIDE3LjYyTDE1OC44MyAyMS4xMFExNTguMTEgMjEuNjEgMTU3LjE4IDIxLjg5UTE1Ni4yNCAyMi4xNyAxNTUuMzAgMjIuMTdMMTU1LjMwIDIyLjE3UTE1My45OSAyMi4xNyAxNTIuOTMgMjEuNjFRMTUxLjg3IDIxLjA1IDE1MS4yNyAyMC4wNVExNTAuNjggMTkuMDYgMTUwLjY4IDE3LjgwWk0xNjYuMjAgMjJMMTYzLjgyIDIyTDE2My44MiAxMy42MEwxNjcuNjcgMTMuNjBRMTY4LjgxIDEzLjYwIDE2OS42NSAxMy45OFExNzAuNDkgMTQuMzUgMTcwLjk0IDE1LjA2UTE3MS40MCAxNS43NiAxNzEuNDAgMTYuNzFMMTcxLjQwIDE2LjcxUTE3MS40MCAxNy42MiAxNzAuOTcgMTguMzBRMTcwLjU1IDE4Ljk4IDE2OS43NSAxOS4zNkwxNjkuNzUgMTkuMzZMMTcxLjU2IDIyTDE2OS4wMiAyMkwxNjcuNTAgMTkuNzdMMTY2LjIwIDE5Ljc3TDE2Ni4yMCAyMlpNMTY2LjIwIDE1LjQ3TDE2Ni4yMCAxNy45M0wxNjcuNTIgMTcuOTNRMTY4LjI1IDE3LjkzIDE2OC42MyAxNy42MVExNjkuMDAgMTcuMjkgMTY5LjAwIDE2LjcxTDE2OS4wMCAxNi43MVExNjkuMDAgMTYuMTIgMTY4LjYzIDE1Ljc5UTE2OC4yNSAxNS40NyAxNjcuNTIgMTUuNDdMMTY3LjUyIDE1LjQ3TDE2Ni4yMCAxNS40N1pNMTc3LjYxIDIyTDE3NS4xOCAyMkwxNzguODkgMTMuNjBMMTgxLjIzIDEzLjYwTDE4NC45NSAyMkwxODIuNDggMjJMMTgxLjgyIDIwLjM3TDE3OC4yNyAyMC4zN0wxNzcuNjEgMjJaTTE4MC4wNCAxNS45M0wxNzguOTYgMTguNjFMMTgxLjEyIDE4LjYxTDE4MC4wNCAxNS45M1pNMTkxLjMwIDIyTDE4OS4xMCAyMkwxODkuMTAgMTMuNjBMMTkxLjA2IDEzLjYwTDE5NC4wMSAxOC40NUwxOTYuODkgMTMuNjBMMTk4Ljg1IDEzLjYwTDE5OC44NyAyMkwxOTYuNjkgMjJMMTk2LjY3IDE3LjU1TDE5NC41MCAyMS4xN0wxOTMuNDUgMjEuMTdMMTkxLjMwIDE3LjY3TDE5MS4zMCAyMloiIGZpbGw9IiNGRkZGRkYiIHg9IjEwNC45MSIvPjwvc3ZnPg==)](https://t.me/rossmann_bernardo_bot)

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

Numerical attributes statistic analysis:

 ![numericalfeatures](https://user-images.githubusercontent.com/68538809/149192046-bbd94f39-49da-49be-a8d7-489191b0324d.JPG)

Categorical attributes statistic analysis:

![categorical features](https://user-images.githubusercontent.com/68538809/149192066-cd8b2b68-3ad2-422a-a274-0b2caa62df45.JPG)

### Step 2. Feature Engineering

On feature creation, **11 columns were created and 2 were modified**:

**Based on the attribute 'Date', 5 features were derived** - 'year', 'month', 'day', 'week_of_year' and 'year_week'

**Based on the attribute 'competition_open_since_year' 2 features were derived** - 'competition_since' and 'competition_time_month'

**Based on the attribute 'promo_since' 3 features were derived** - 'promo_since', 'promo_time_week' and 'promo_time_day'

**Based on the attributed 'promo_interval' and 'month_map' 1 feature was created** - 'is_promo'

**'assortment' and 'state_holiday' were modified**, the values were changed to a descriptive format

### Step 3. Data Filtering

Two types of data filtering were used:

- **Rows filtering**: I have filtered data based on open days in which sales were made as this is a sales analysis 
- **Columns Selection**: I have dropped 4 columns. 'Customers' as they would not be part of a prediction dataset. 'Open' as this column is no longer useful after filtering based on exclusively open days. 'Promo_interval' and  'Month_map' were useful to derive new features but give no additional inputs to the model.

### Step 4. Exploratory Data Analysis (EDA)

On Exploratory Data Analysis, Univariate, Bivariate, and Multivariates study was performed to help understand the statistical properties of each attribute, correlations, and hypothesis testing.

### Step 5. Data Preparation

In this section, Rescaling (Robust Scaler and MinMax Scaler), Encoding (One Hot, Label and Ordinal Encoding), Variable Transformation (Logarithmic Transformation), and Nature Transformations of the variables were carried out.

### Step 6. Feature Selection

To select the features to be used, two methods were used:

1. Application of Boruta using RandomForestRegressor method;
2. EDA insights.

**From Boruta Feature Selection:** 'store', 'promo', 'store_type', 'assortment', 'competition_distance', 'competition_open_since_month', 'competition_open_since_year', 'promo2', 'promo2_since_week', 'promo2_since_year', 'competition_time_month', 'promo_time_week', 'promo_time_day', 'day_of_week_sin', 'day_of_week_cos', 'month_cos', 'day_sin', 'day_cos', week_of_year_cos'

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

**Mean Absolute Error (MAE)**

**Mean Absolute Percentage Error (MAPE)**

**Root Mean Square Error (RMSE)**


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

## **2.** Stores revenue prediction for the upcoming 6 weeks.

## **3.** Create online bot that automatically generates specific store revenue prediction based on request.

# 6. Conclusions

# 7. Next Steps to Improve

