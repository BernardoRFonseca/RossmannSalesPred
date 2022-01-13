# Rossmann Sales Prediction
## 
![csm_rossmann_shop_foto_stralsund_1633a5fb67](https://user-images.githubusercontent.com/68538809/148125358-4b61a4af-bc42-4901-8633-49aa49a984b5.png)


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


### Step 5. Data Preparation


### Step 6. Feature Selection


### Step 7. Machine Learning Modelling


### Step 8. Hyperparameter Fine Tuning


### Step 9. Final Model


### Step 10. Performance Evaluation and Interpretation


# 3. Data Insights

# 4. Machine Learning Model Applied


# 5. Machine Learning Model Performance


# 6. Business Results

# 7. Conclusions


# 8. Next Steps to Improve

