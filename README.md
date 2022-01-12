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


### Step 3. Exploratory Data Analysis (EDA)


### Step 4. Data Preparation


### Step 5. Feature Selection


### Step 6. Machine Learning Modelling


### Step 7. Hyperparameter Fine Tuning


### Step 8. Final Model


### Step 9. Performance Evaluation and Interpretation


# 3. Data Insights

# 4. Machine Learning Model Applied


# 5. Machine Learning Model Performance


# 6. Business Results

# 7. Conclusions


# 8. Next Steps to Improve

