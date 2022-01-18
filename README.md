# Rossmann Sales Prediction
## 
![csm_rossmann_shop_foto_stralsund_1633a5fb67](https://user-images.githubusercontent.com/68538809/148125358-4b61a4af-bc42-4901-8633-49aa49a984b5.png)

[![python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

# 1. Business Context.

Dirk Rossmann GmbH is one of the largest drug store chains in Europe with around 56,200 employees and over €10 billion annual revenue in Germany, Poland, Hungary, the Czech Republic, Turkey, Albania, Kosovo and Spain. 

The scope of this project is based on a CFO demand: stores will be renovated, and for that, the budget needs to be aligned with each store's sales. Adding to that, it was requested to predict each store sales for the following 6 weeks.

In this context, this project intends to provide the management team with a 6 week store sales prediciton.

## 1.1 Project Deliverables.

In this business context, as a data scientist, I intend to develop a machine learning model that predicts the 6 weeks sales per store.
 
With this model, the management team (including the CFO) will be able to manage the stores renovations.

A report with the following analysis is the deliverable goals:


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

# 3. Data Insights

# 4. Machine Learning Model Applied

The following Machine Learning models were tested and cross-validated:

- **Linear Regression**
- **Linear Regression Regularized - LASSO**
- **Random Forest**
- **XGBoost Regressor**

# 5. Machine Learning Model Performance


# 6. Business Results

# 7. Conclusions


# 8. Next Steps to Improve

