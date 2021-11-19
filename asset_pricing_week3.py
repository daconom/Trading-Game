import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize, LinearConstraint, Bounds

#import data
prices=pd.read_excel('/Users/davidwester/VU_Finance/Asset Pricing/Data_game_20211111_NOFS.xlsx', sheet_name='PRICES', index_col= 'Date', header=0) 
pb=pd.read_excel('/Users/davidwester/VU_Finance/Asset Pricing/Data_game_20211111_NOFS.xlsx', sheet_name='PRICE TO BOOK', index_col= 'Date', header=0) 
size=pd.read_excel('/Users/davidwester/VU_Finance/Asset Pricing/Data_game_20211111_NOFS.xlsx', sheet_name='SIZE', index_col= 'Date', header=0) 
turnover=pd.read_excel('/Users/davidwester/VU_Finance/Asset Pricing/Data_game_20211111_NOFS.xlsx', sheet_name='VOLUME', index_col= 'Date', header=0) 
beta=pd.read_excel('/Users/davidwester/VU_Finance/Asset Pricing/beta.xlsx', sheet_name='BETA', index_col= 'Date', header=0) 
eps=pd.read_excel('/Users/davidwester/VU_Finance/Asset Pricing/eps.xlsx', sheet_name='eps', index_col= 'Date', header=0) 
shares_outstanding=pd.read_excel('/Users/davidwester/VU_Finance/Asset Pricing/outstanding.xlsx', sheet_name='outstanding', index_col= 'Date', header=0) 
sales_to_price=pd.read_excel('/Users/davidwester/VU_Finance/Asset Pricing/sales.xlsx', sheet_name='salestoprice', index_col= 'Date', header=0) 
growth_in_assets=pd.read_excel('/Users/davidwester/VU_Finance/Asset Pricing/growth_in_assets.xlsx', sheet_name='growth_in_assets', index_col= 'Date', header=0) 
AAII=pd.read_excel('/Users/davidwester/VU_Finance/Asset Pricing/sentiment.xls', sheet_name='AAII', index_col= 'Date', header=0) 
twitter=pd.read_excel('/Users/davidwester/VU_Finance/Asset Pricing/sentiment.xls', sheet_name='twitter', index_col= 'Date', header=0) 

returns = (prices-prices.shift(1))/prices.shift(1) * 100
stocks = prices.columns

d = {}

#tqdm adds a progress bar
for x in tqdm(stocks):
    print(x)
    #feature construction
    price_x = prices[x]
    #short run momentum (25 trading days)
    momentum1_x = (price_x - price_x.shift(25))/price_x.shift(25) *100
    #Price-to-Book
    pb_x = pb[x].replace("@NA", np.NaN)
    #Size
    size_x = size[x].replace("@NA", np.NaN)
    #Turnover
    turnover_x = turnover[x].replace("@NA", np.NaN)
    #Beta
    beta_x = beta[x].replace("@NA", np.NaN)
    #Beta Squared
    betasq_x = beta[x]**2
    #Sales-to-Price
    sales_to_price_x = sales_to_price[x].replace("@NA", np.NaN)
    #log(Equity) = log(Outstanding shares * Price)
    shares_outstanding_x = shares_outstanding[x].replace("@NA", np.NaN)
    equity_x = price_x * shares_outstanding_x
    equity_x = np.log(equity_x)
    #Earnings-to-Price: EPS/Price
    eps_x = eps[x]
    earnings_to_price_x = eps_x/price_x
    #Growth in total assets
    growth_in_assets_x = growth_in_assets[x].replace("@NA", np.NaN)
    #Returns
    returns_x = returns[x]
    last_return_x = returns_x.shift(1)
    #Sentiment measures
    AAII_x = AAII["Spread_AAII"]
    ## aaii.com - use the spread between bullish and bearish sentiment
    tw_x = twitter["TEU-ENG"]
    ## policyuncertainty.com/twitter_uncert.html  

    ##First: define a dataframe and remove the non-trading days.
    data_x  = {"Price": price_x, 'Returns': returns_x, 'Momentum (1-month)': momentum1_x,  'Last return': last_return_x, 'Price to book': pb_x, 'Size': size_x, 'Turnover': turnover_x, 'Beta': beta_x, 'Betasq': betasq_x, "Sales to Price": sales_to_price_x, "log market equity": equity_x, "Earnings to Price": earnings_to_price_x, "Growth in Assets": growth_in_assets_x, "AAII": AAII_x, "Twitter": tw_x }
    pd_data_x = pd.DataFrame(data_x)
    pd_data_x.dropna(subset = ["Price", "Last return"], inplace=True)
    ##Some variables contain missing values for some stocks and can therefore not be used.
    pd_data_x = pd_data_x[pd_data_x.columns[~pd_data_x.isnull().any()]]
    non_empty = pd_data_x.columns
    
    ##Select all features without missing values
    X = pd_data_x.loc[:, non_empty]
    ##Select dependent variable
    Y = pd_data_x.Returns
    
    #define classifier.
    ##We use a RandomForestClassifier https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    ##n_estimator = number of trees in the forest, max_depth = depth of the tree
    clf = RandomForestRegressor(random_state=5)
    
    #train the forest
    training_res = clf.fit(X, Y)
    #Make predictions 
    d["predicted_return{0}".format(x)] = training_res.predict(X)


#Now optimize over predictions
pd_data_x = pd.DataFrame(d)
print(pd_data_x.head())

#Variance-covariance matrix of log returns
cov = pd_data_x.apply(lambda x: np.log(1+x)).cov().to_numpy()

#Weekly returns. Change to M for monthly 
er = pd_data_x.mean().to_numpy()

def objective_function(x):
    return -(np.dot(x, er.transpose()) - 0.25*np.sqrt((np.dot(np.dot(x.transpose(),cov),x))**(1/2)) - np.exp(np.sum(np.dot(x, np.log(x).transpose()))))

n_assets = len(prices.columns)
x = np.random.random(n_assets)
x = x/np.sum(x)

#sum w <= 1
constraint1 = ({'type': 'ineq', 'fun': lambda x:  1 - sum(x)})
constraint2 = {'type':'ineq', 'fun': lambda x: np.sum(x, 0) - 0.85}  

#Bounds(0, 0.1, keep_feasible=True)
bnds1 = tuple((0.000000000001, 0.1) for i in range(n_assets))

resr = minimize(objective_function , x0 = np.random.random(n_assets), constraints= (constraint1, constraint2) , bounds=bnds1)
print("The total invested capital is:", sum(resr.x))
print("The largest share invested is:", max(resr.x))
print("The value of the grading function is:", -resr.fun)

portf_data  = {'Stocks':prices.columns, 'Weights': resr.x}
portfolios = pd.DataFrame(portf_data)
print(portfolios.head(100))
portfolios.to_csv("portfolios_weights_ml.csv",index=False)
portfolios.to_excel('portfolios_weights_ml.xlsx')  
