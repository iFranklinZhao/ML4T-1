"""MC1-P1: Analyze a portfolio."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data
import RTLearner as rtl
import BagLearner as bl
import sys

# This function returns indicators and Y values from price timeseries

def generate_indicators(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,12,31), \
    syms = ['AAPL'], \
    allocs=[1.0], \
    sv=1000000, rfr=0.0, sf=252.0, \
    gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols

    # Get daily portfolio value
    norm_prices = prices/prices.ix[0,:] # normalizing data
    norm_prices = norm_prices*allocs*sv # multiplying by allocation and starting stock value
    port_val = norm_prices.sum(axis=1) # summing to get daily portfolio values

    # Calculate indicators

    # Momentum indicator
    window = int(20)
    momentum = []

    price_array = pd.DataFrame.as_matrix(prices)
    for i in range(prices.shape[0]):
        if  i > window:
            momentum_value = (price_array[i]/price_array[i-window]) - 1.0
            momentum.append(momentum_value)

    pmomentum_value = prices.copy()
    pmomentum_value[:] = 0.0
    norm_momentum = (momentum-np.mean(momentum))/np.std(momentum)
    pmomentum_value.ix[window+1:,:] = norm_momentum
    pmomentum_value.ix[:window+1,:] = norm_momentum[22][0]
    
    # Bollinger Bands    
    rolling_std = pd.rolling_std(prices, window=21)
    rolling_avg = pd.rolling_mean(prices, window=21)
    bb_upper = rolling_avg + 2.0*rolling_std
    bb_lower = rolling_avg - 2.0*rolling_std
    bbp_band = (prices-bb_lower)/(bb_upper-bb_lower)
    normed_bbp = (bbp_band-np.mean(bbp_band))/np.std(bbp_band)

    # SMA/Price Indicator

    sma = rolling_avg/prices
    normed_sma = (sma-np.mean(sma))/np.std(sma)
    normed_sma.ix[:21,:] = pd.DataFrame.as_matrix(normed_sma)[22]

    # Daily returns Indicator
    daily_rets = prices.copy()
    daily_rets[:] = 0.0
    daily_rets = (prices/prices.shift(1)) - 1.0
    norm_daily_rets = (daily_rets-np.mean(daily_rets))/np.std(daily_rets)
    daily_rets.ix[0,:] = 0.0

    norm_price = (price_array-np.mean(price_array))/np.std(price_array)

    #plt.plot(pmomentum_value,'k-', label = 'Momentum', lw=2)
    #plt.plot(norm_price)
    #plt.plot(prices)
    #plt.plot(normed_bb_upper)
    #plt.plot(normed_bb_lower)
    #plt.plot(normed_bbp, 'b-', label = 'Bollinger Band %', lw=2)
    #plt.plot(daily_rets, label = 'Daily Returns', lw=2) 
    #plt.plot(normed_sma, 'r-', label = 'SMA/Price', lw=2)
    #plt.legend()
    #plt.show()

    # Calculate Y values using a 21 day wait period

    YBUY  = 0.001
    YSELL = -0.001
    
    return_21days = prices.copy()
    return_21days = (prices/prices.shift(21)) - 1.0
    return_21days.ix[0:21,:] = 0.0
    indices = return_21days.index
    Y = np.zeros(len(indices))

    for idx in range(len(indices)):
        if return_21days.ix[indices[idx]][0] > YBUY:
            Y[idx] = 1
        elif return_21days.ix[indices[idx]][0] < YSELL:
            Y[idx] = -1
        else:
            Y[idx]  = 0

    learning_data = [prices, normed_bbp, normed_sma, pmomentum_value, daily_rets, Y]
    return learning_data

def test_code():
    # Input parameters
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2009,12,31)
    symbols = ['AAPL']
    allocations = [1.0]
    start_val = 1000000  
    risk_free_rate = 0.0
    sample_freq = 252

    # Get trainig data: Indicators (X) and Y values 
    stock_prices, bbp, sma_price, momen, d_rets, Y = generate_indicators(sd = start_date, ed = end_date,\
        syms = symbols, \
        allocs = allocations,\
        sv = start_val, \
        gen_plot = False)
    trainX = pd.DataFrame.as_matrix(pd.concat([bbp, sma_price, momen], axis=1))
    trainY = Y

    # Use a Bagged (20) Regression Tree with 5 leaves
    learner = bl.BagLearner(learner = rtl.RTLearner, kwargs = {"leaf_size":5, "verbose":False}, bags = 200, boost = False, verbose = False)
    learner.addEvidence(trainX, trainY)
    predY = learner.query(trainX)

    order_dates = stock_prices.index
    orders = []
    order_file = open('ml_orders.csv','w')
    manual_file = open('calc_orders.csv','w')

    orders.append(['Date','Symbol','Order','Shares'])

    stock_holdings = 0
    last_order_date = order_dates[0]

    for j in range(len(predY)):
        if stock_holdings == 0:
            if predY[j] == 1:
                orders.append([order_dates[j].date(), 'AAPL', 'BUY', 200])
                stock_holdings = stock_holdings + 200
                last_order_date = order_dates[j]
            elif predY[j] == -1:
                orders.append([order_dates[j].date(), 'AAPL', 'SELL', 200])
                stock_holdings = stock_holdings - 200
                last_order_date = order_dates[j]
        elif stock_holdings == 200:
            if predY[j] == -1:
                if last_order_date == order_dates[0] or pd.date_range(last_order_date, order_dates[j]).shape[0] > 21:
                    orders.append([order_dates[j].date(), 'AAPL', 'SELL', 200])
                    stock_holdings = stock_holdings - 200
                    last_order_date = order_dates[j]
        elif stock_holdings == -200:
            if predY[j] == 1:
                if last_order_date == order_dates[0] or pd.date_range(last_order_date, order_dates[j]).shape[0] > 21:
                    orders.append([order_dates[j].date(), 'AAPL', 'BUY', 200])
                    stock_holdings = stock_holdings + 200
                    last_order_date = order_dates[j]
        else:
            continue
        if pd.date_range(last_order_date, order_dates[j]).shape[0] > 21:
            if stock_holdings == 200:
                orders.append([order_dates[j].date(), 'AAPL', 'SELL', 200])
                stock_holdings = stock_holdings - 200
            elif stock_holdings == -200:
                orders.append([order_dates[j].date(), 'AAPL', 'BUY', 200])
                stock_holdings = stock_holdings + 200

    for order in orders:
        print >> order_file, ",".join(str(x) for x in order)

    manual_orders = []
    manual_orders.append(['Date','Symbol','Order','Shares'])
    stock_holdings = 0
    last_order_date = order_dates[0]

    for k in range(trainX.shape[0]):
        point = trainX[k]
        if (point[0] > 0.05 and point[1] < 0.1 and point[2] > 0.005):
            if last_order_date == order_dates[0] or pd.date_range(last_order_date, order_dates[k]).shape[0] > 21: 
                if stock_holdings == 0 or stock_holdings == -200: 
                    manual_orders.append([order_dates[k].date(), 'AAPL', 'BUY', 200])
                    stock_holdings = stock_holdings + 200
                    last_order_date = order_dates[k]
        elif (point[0] < 0.05 and point[1] > 0.1 and point[2] < 0.005):
            if last_order_date == order_dates[0] or pd.date_range(last_order_date, order_dates[k]).shape[0] > 21:
                if stock_holdings == 0 or stock_holdings == 200:            
                    manual_orders.append([order_dates[k].date(), 'AAPL', 'SELL', 200])
                    stock_holdings = stock_holdings - 200
                    last_order_date = order_dates[k]
        else:
            continue

    for manual_order in manual_orders:
        print >> manual_file, ",".join(str(x) for x in manual_order)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols

if __name__ == "__main__":
    test_code()
