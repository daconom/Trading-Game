import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

#Note: AllData was changed
AllData=pd.read_excel('/Users/davidwester/VU_Finance/Asset Pricing/Data_game_20211111_NOFS.xlsx', sheet_name='PRICES', index_col= 'Date', header=0) 
prices = AllData.loc['2020-12-31':'2021-11-10']
print(prices.tail())
stocks = prices.columns

momentum = []
reversal = []
for x in tqdm(stocks):
    price_x = prices[x]
    momentum1_x = (price_x - price_x.shift(75))/price_x.shift(75) *100
    reversal_x = (price_x - price_x.shift(25))/price_x.shift(25) *100
    momentum_last = momentum1_x.loc['2021-11-10']
    reversal_last = reversal_x.loc['2021-11-10']
    print("%s:\n Momentum (3 month): %d\n Momentum (1 month): %d" % (x, momentum_last, reversal_last))
    momentum.append(momentum_last)
    reversal.append(reversal_last)

st_data  = {'Stocks':stocks, 'Momentum': momentum, 'Reversal':reversal}
st_df = pd.DataFrame(st_data)
print(st_df)

#Momentum weights
st_df['row_num'] = st_df.reset_index().index
dfER_sorted = st_df.sort_values(by = ['Momentum'], ascending = False)
top180 = dfER_sorted.head(180)
totalReturn = np.sum(top180['Momentum'])
#Invest 85% in Momentum
normalizationFactor = 0.85/totalReturn
top180['Weight'] = normalizationFactor*top180['Momentum']
top180 = top180.sort_values(by = ['row_num'])
Momentum = top180[['Stocks', 'Weight']]
Momentum.to_csv("Momentum_weights.csv",index=False)
print(Momentum)  

#Reversal weights
dfER_sorted2 = st_df.sort_values(by = ['Reversal'], ascending = True)
count_negative = (dfER_sorted2['Reversal'] < 0).sum().sum()
print(count_negative)
topdecline = dfER_sorted2.head(count_negative)
print(topdecline)
totalReturn = np.sum(topdecline['Reversal'])
#Invest 15% in Reversal
normalizationFactor = 0.15/totalReturn
topdecline['Weight'] = normalizationFactor*topdecline['Reversal']
topdecline = topdecline.sort_values(by = ['row_num'])
Reversal = topdecline[['Stocks', 'Weight']]
Reversal.to_csv("Reversal_weights.csv",index=False)
