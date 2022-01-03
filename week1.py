import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds

#importing the excel file
prices=pd.read_excel('/Users/davidwester/VU_Finance/Asset Pricing/Data_game_20211026_NOFS.xlsx', sheet_name='PRICES', index_col= 'Date', header=0) 
print(prices.head())

#Variance-covariance matrix of log returns
cov = prices.pct_change().apply(lambda x: np.log(1+x)).cov().to_numpy()

#Weekly returns. Change to M for monthly 
er = prices.resample('W').last().pct_change().mean().to_numpy()

# min -(w'E[r] - 0.25 sqrt(w' * sigma * w))
# s.t 0.85<= sum w_i <= 1   
# 0 < w_i < 1


def objective_function(x):
    return -(np.dot(x, er.transpose()) - 0.25*np.sqrt((np.dot(np.dot(x.transpose(),cov),x))**(1/2)) - np.exp(np.dot(x, np.log(x).transpose())))

#- np.sum(np.dot(x, np.log(x).transpose())) 

n_assets = len(prices.columns)
x = np.random.random(n_assets)
x = x/np.sum(x)

test1 = np.exp(np.sum(np.dot(x, np.log(x).transpose())))
print(test1)
test2 = np.exp(np.dot(x, np.log(x).transpose()))
print(test2)


#sum w <= 1
constraint1 = ({'type': 'ineq', 'fun': lambda x:  1 - sum(x)})
constraint2 = {'type':'ineq', 'fun': lambda x: np.sum(x, 0) - 0.85}  

#Bounds(0, 0.1, keep_feasible=True)
bnds1 = tuple((0.00000000001, 0.1) for i in range(n_assets))

resr = minimize(objective_function , x0 = np.random.random(n_assets), constraints= (constraint1, constraint2) , bounds=bnds1)
print("The total invested capital is:", sum(resr.x))
print("The largest share invested is:", max(resr.x))
print("The value of the grading function is:", -resr.fun)

portf_data  = {'Stocks':prices.columns, 'Weights': resr.x}
portfolios = pd.DataFrame(portf_data)
print(portfolios.head(100))
portfolios.to_csv("portfolios_weights.csv",index=False)
portfolios.to_excel('portfolios_weights.xlsx')  
