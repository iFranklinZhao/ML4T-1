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
    sv=100000, rfr=0.0, sf=252.0, \
    gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols

    # Get daily portfolio value
    norm_prices = prices/prices.ix[0,:] # normalizing data
    #norm_prices = norm_prices*allocs*sv # multiplying by allocation and starting stock value
    #port_val = norm_prices.sum(axis=1) # summing to get daily portfolio values

    # Calculate indicators

    # Momentum indicator
    window = int(21)
    momentum = []

    price_array = pd.DataFrame.as_matrix(prices)
    for i in range(prices.shape[0]):
        if  i > window:
            momentum_value = (price_array[i]/price_array[i-window]) - 1.0
            momentum.append(momentum_value)

    pmomentum_value = prices.copy()
    pmomentum_value[:] = 0.0

    momentum_2 = prices.copy()
    momentum_2.ix[window+1:,:] = momentum
    momentum_2.ix[:window,:] = momentum_2.ix[20][0]
    norm_momentum = (momentum-np.mean(momentum))/np.std(momentum)
    pmomentum_value.ix[window+1:,:] = norm_momentum
    pmomentum_value.ix[:window,:] = norm_momentum[20][0]
    
    # Bollinger Bands    
    rolling_std = pd.rolling_std(prices, window=21)
    rolling_avg = pd.rolling_mean(prices, window=21)
    bb_upper = rolling_avg + 2.0*rolling_std
    bb_lower = rolling_avg - 2.0*rolling_std
    bbp_band = (prices-bb_lower)/(bb_upper-bb_lower)

    normed_bb_upper = (bb_upper-np.mean(bb_upper))/np.std(bb_upper)
    normed_bb_upper.ix[:21,:] = pd.DataFrame.as_matrix(normed_bb_upper)[20]
    
    normed_bb_lower = (bb_lower-np.mean(bb_lower))/np.std(bb_lower)
    normed_bb_lower.ix[:21,:] = pd.DataFrame.as_matrix(normed_bb_lower)[20]
    
    normed_bbp = (bbp_band-np.mean(bbp_band))/np.std(bbp_band)
    normed_bbp.ix[:21,:] = pd.DataFrame.as_matrix(normed_bbp)[20]

    # SMA/Price indicator
    sma = rolling_avg/prices
    
    normed_rolling_avg = (rolling_avg-np.mean(rolling_avg))/np.std(rolling_avg)
    normed_rolling_avg.ix[:21,:] = pd.DataFrame.as_matrix(normed_rolling_avg)[20]

    normed_sma = (sma-np.mean(sma))/np.std(sma)
    normed_sma.ix[:21,:] = pd.DataFrame.as_matrix(normed_sma)[20]

    # Daily returns indicator
    daily_rets = prices.copy()
    daily_rets[:] = 0.0
    daily_rets = (prices/prices.shift(1)) - 1.0
    norm_daily_rets = (daily_rets-np.mean(daily_rets))/np.std(daily_rets)
    norm_daily_rets.ix[:21,:] = pd.DataFrame.as_matrix(norm_daily_rets)[20]

    #norm_price = (price_array-np.mean(price_array))/np.std(price_array)

    # Plot indicators
    
    if gen_plot == 'train':
        # Bollinger Bands
        df_temp = pd.concat([normed_bb_upper,normed_rolling_avg,normed_bb_lower,normed_bbp], keys=['BB Upper', 'Rolling Avg.', 'BB Lower', 'BB %'], axis=1)
        df_temp = df_temp/df_temp.ix[0,:]
        pdates = df_temp.index

        plt.clf()
        styles = ['g-','b-', 'r-', 'k-']
        linewidths = [2, 2, 2, 2]
        fig, ax = plt.subplots()
        for col, style, lw in zip(df_temp.columns, styles, linewidths):
            df_temp[col].plot(style=style, lw=lw, ax=ax)
        plt.title('Bollenger Band Percentage Indicator', fontsize=16)
        plt.ylabel('Normalized Indicator', fontsize=16)
        plt.xlabel('Dates', fontsize=16)
        #plt.ylim([0.7,1.5])
        plt.legend(loc="upper left")
        plt.savefig('bollinger_bands.png')

        # SMA/Price
        plt.clf()
        df_temp = pd.concat([norm_prices,normed_rolling_avg,normed_sma], keys=['Price', 'SMA', 'SMA/Price'], axis=1)
        #df_temp = pd.concat([prices,rolling_avg,sma], keys=['Price', 'SMA', 'SMA/Price'], axis=1)
        df_temp = df_temp/df_temp.ix[0,:]
        styles = ['b-', 'r-', 'k-']
        linewidths = [2, 2, 2]
        #print df_temp
        fig, ax = plt.subplots()
        for col, style, lw in zip(df_temp.columns, styles, linewidths):
            df_temp[col].plot(style=style, lw=lw, ax=ax)
        plt.title('Simple Moving Average/Price', fontsize=16)
        plt.ylabel('Normalized Indicator', fontsize=16)
        plt.xlabel('Dates', fontsize=16)
        plt.legend(loc="upper center")
        plt.savefig('sma_price.png')
    
        # Plot momentum
        plt.clf()
        df_temp = pd.concat([norm_prices,pmomentum_value], keys=['Price', 'Momentum'], axis=1)
        df_temp = df_temp/df_temp.ix[0,:]
        styles = ['b-', 'r-']
        linewidths = [2, 2]
        fig, ax = plt.subplots()
        for col, style, lw in zip(df_temp.columns, styles, linewidths):
            df_temp[col].plot(style=style, lw=lw, ax=ax)
        plt.title('Momentum Indicator', fontsize=16)
        plt.ylabel('Normalized Indicator', fontsize=16)
        plt.xlabel('Dates', fontsize=16)
        plt.legend(loc="upper center")
        plt.savefig('momentum_price.png')

    #plt.plot(pmomentum_value,'k-', label = 'Momentum', lw=2)
    #plt.plot(normed_bbp, 'b-', label = 'Bollinger Band %', lw=2)
    #plt.plot(normed_sma, 'r-', label = 'SMA/Price', lw=2)
    #plt.legend()
    #plt.savefig("plotted_indicators.png")

    # Calculate Y values using a 21 day wait period
    YBUY  = 0.005
    YSELL = -0.005
    
    return_21days = prices.copy()
    return_21days = (prices/prices.shift(21)) - 1.0
    return_21days.ix[0:21,:] = 0
    indices = return_21days.index
    Y = np.zeros(len(indices))

    for idx in range(len(indices)):
        if return_21days.ix[indices[idx]][0] > YBUY:
            Y[idx] = 1
        elif return_21days.ix[indices[idx]][0] < YSELL:
            Y[idx] = -1
        else:
            Y[idx]  = 0

    learning_data = [prices, normed_bbp, normed_sma, pmomentum_value, norm_daily_rets, Y]
    return learning_data

def test_code():
    # Input parameters
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2009,12,31)
    symbols = ['AAPL']
    allocations = [1.0]
    start_val = 100000  
    risk_free_rate = 0.0
    sample_freq = 252

    test_start_date = dt.datetime(2010,1,1)
    test_end_date = dt.datetime(2011,12,31)

    # Get trainig data: Indicators (X) and Y values 
    stock_prices, bbp, sma_price, momen, d_rets, Y = generate_indicators(sd = start_date, ed = end_date,\
        syms = symbols, \
        allocs = allocations,\
        sv = start_val, \
        gen_plot = 'train')
    trainX = pd.DataFrame.as_matrix(pd.concat([bbp, sma_price, momen], axis=1))
    trainY = Y

    test_prices, test_bbp, test_sma_price, test_momen, test_d_rets, tY = generate_indicators(sd = test_start_date, ed = test_end_date,\
        syms = symbols, \
        allocs = allocations,\
        sv = start_val, \
        gen_plot = 'test')
    testX = pd.DataFrame.as_matrix(pd.concat([test_bbp, test_sma_price, test_momen], axis=1))
    testY = tY
          
    # Use a Bagged (20) Regression Tree with 5 leaves
    learner = bl.BagLearner(learner = rtl.RTLearner, kwargs = {"leaf_size":15, "verbose":False}, bags = 200, boost = False, verbose = False)
    learner.addEvidence(trainX, trainY)
    predY = learner.query(trainX)

    # Predict test results
    testPrediction = learner.query(testX)

    # Generate order files using ML strategy
    train_order_dates = stock_prices.index
    test_order_dates = test_prices.index

    def generate_orders_ml(_trainX, _trainY, filename, gen_dates):
        orders = []
        order_file = open(filename +'.csv','w')
        #manual_file = open('calc_orders.csv','w')

        orders.append(['Date','Symbol','Order','Shares'])
        order_dates = gen_dates

        stock_holdings = 0
        last_order_date = order_dates[0]
        _predY = _trainY
        for j in range(len(_predY)):
            if stock_holdings == 0:
                if _predY[j] == 1:
                    orders.append([order_dates[j].date(), 'AAPL', 'BUY', 200])
                    stock_holdings = stock_holdings + 200
                    last_order_date = order_dates[j]
                elif _predY[j] == -1:
                    orders.append([order_dates[j].date(), 'AAPL', 'SELL', 200])
                    stock_holdings = stock_holdings - 200
                    last_order_date = order_dates[j]
            elif stock_holdings == 200:
                if _predY[j] == -1:
                    if last_order_date == order_dates[0] or pd.date_range(last_order_date, order_dates[j]).shape[0] > 21:
                        orders.append([order_dates[j].date(), 'AAPL', 'SELL', 200])
                        stock_holdings = stock_holdings - 200
                        last_order_date = order_dates[j]
            elif stock_holdings == -200:
                if _predY[j] == 1:
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

    def generate_orders_manual(trainX, filename, gen_dates):
        # Generate order file using manual strategy
        # order_dates = stock_prices.index
        manual_file = open(filename +'.csv','w')
        manual_orders = []
        manual_orders.append(['Date','Symbol','Order','Shares'])
        stock_holdings = 0
        order_dates = gen_dates
        last_order_date = order_dates[0]
        
        #bbp, sma_price, momen
        for k in range(trainX.shape[0]):
            point = trainX[k]
            if stock_holdings == 0:
                if (point[0] > 0.05 and point[1] < 0.1 and point[2] > 0.005):
                    manual_orders.append([order_dates[k].date(), 'AAPL', 'BUY', 200])
                    stock_holdings = stock_holdings + 200
                    last_order_date = order_dates[k]
                elif (point[0] < 0.05 and point[1] > 0.1 and point[2] < 0.005):
                    manual_orders.append([order_dates[k].date(), 'AAPL', 'SELL', 200])
                    stock_holdings = stock_holdings - 200
                    last_order_date = order_dates[k]
            elif stock_holdings == 200:
                if (point[0] < 0.05 and point[1] > 0.1 and point[2] < 0.005):
                    if last_order_date == order_dates[0] or pd.date_range(last_order_date, order_dates[k]).shape[0] > 21:
                        manual_orders.append([order_dates[k].date(), 'AAPL', 'SELL', 200])
                        stock_holdings = stock_holdings - 200
                        last_order_date = order_dates[k]
            elif stock_holdings == -200:
                if (point[0] > 0.05 and point[1] < 0.1 and point[2] > 0.005):
                    if last_order_date == order_dates[0] or pd.date_range(last_order_date, order_dates[k]).shape[0] > 21:
                        manual_orders.append([order_dates[k].date(), 'AAPL', 'BUY', 200])
                        stock_holdings = stock_holdings + 200
                        last_order_date = order_dates[k]
            else:
                continue
            if pd.date_range(last_order_date, order_dates[k]).shape[0] > 21:
                if stock_holdings == 200:
                    manual_orders.append([order_dates[k].date(), 'AAPL', 'SELL', 200])
                    stock_holdings = stock_holdings - 200
                elif stock_holdings == -200:
                    manual_orders.append([order_dates[k].date(), 'AAPL', 'BUY', 200])
                    stock_holdings = stock_holdings + 200

        for manual_order in manual_orders:
            print >> manual_file, ",".join(str(x) for x in manual_order)

    generate_orders_ml(trainX, predY, 'ml_orders', train_order_dates)
    generate_orders_manual(trainX, 'calc_orders', train_order_dates)    

    generate_orders_ml(testX, testPrediction, 'test_ml_orders', test_order_dates)
    generate_orders_manual(testX, 'test_calc_orders', test_order_dates)


    #Plot scatter plots

    # Training data

    longs = trainX[trainY==1]
    shorts = trainX[trainY==-1]
    do_nothing = trainX[trainY==0]

    #print len(longs)
    #print len(shorts)
    #print len(do_nothing)

    plt.clf()
    plt.plot(longs[:,0],longs[:,2], 'go', label = 'LONGS', ms=8)
    plt.plot(shorts[:,0],shorts[:,2], 'rh', label = 'SHORTS', ms=8)
    plt.plot(do_nothing[:,0],do_nothing[:,2], 'k*', label = 'HOLDS', ms=8)
    plt.title('Training Data for ML Strategy', fontsize=16)
    plt.ylabel('Bollinger Band Percentage', fontsize=16)
    plt.xlabel('Momentum', fontsize=16)
    plt.legend(loc="upper left")
    plt.savefig("scatters_training.png")

    # Response after query

    qlongs = trainX[predY==1]
    qshorts = trainX[predY==-1]
    qdo_nothing = trainX[predY==0]

    #print len(qlongs)
    #print len(qshorts)
    #print len(qdo_nothing)

    plt.clf()
    plt.plot(qlongs[:,0],qlongs[:,2], 'go', label = 'LONGS', ms=8)
    plt.plot(qshorts[:,0],qshorts[:,2], 'rh', label = 'SHORTS', ms=8)
    plt.plot(qdo_nothing[:,0],qdo_nothing[:,2], 'k*', label = 'HOLDS', ms=8)
    #plt.ylim([-1.5,1.5])
    #plt.xlim([-1.5,1.5])
    plt.title('ML Strategy Predictions', fontsize=16)
    plt.ylabel('Bollinger Band Percentage', fontsize=16)
    plt.xlabel('Momentum', fontsize=16)
    plt.legend(loc="upper left")
                 
    plt.savefig("scatters_ml.png")

    # Using Rule-based strategy to classify data

    ruleY = np.zeros(trainX.shape[0])
    for m in range(trainX.shape[0]):
        point = trainX[m]
        if (point[0] > 0.05 and point[1] < 0.1 and point[2] > 0.005):
            ruleY[m] = 1
        elif (point[0] < 0.05 and point[1] > 0.1 and point[2] < 0.005):
            ruleY[m] = -1 
        else:
            ruleY[m] = 0

    rlongs = trainX[ruleY==1]
    rshorts = trainX[ruleY==-1]
    rdo_nothing = trainX[ruleY==0]
                
    plt.clf()
    plt.plot(rlongs[:,0],rlongs[:,2], 'go', label = 'LONGS', ms=8)
    plt.plot(rshorts[:,0],rshorts[:,2], 'rh', label = 'SHORTS', ms=8)
    plt.plot(rdo_nothing[:,0],rdo_nothing[:,2], 'k*', label = 'HOLDS', ms=8)
    plt.title('Rule Based Predictions', fontsize=16)
    plt.ylabel('Bollinger Band Percentage', fontsize=16)
    plt.xlabel('Momentum', fontsize=16)
    plt.legend(loc="upper left")
    plt.savefig("scatters_rulebased.png")

    # Best Strategy
    
    matix_prices = pd.DataFrame.as_matrix(stock_prices)
    best_strategy = []
    best_strategy.append(['Date','Symbol','Order','Shares'])
    best_order_dates = stock_prices.index
    best_strategy_file = open('best_strategy_orders.csv','w')

    print matix_prices
    print len(matix_prices)
    print best_order_dates
    print len(best_order_dates)

    for q in range(0,len(matix_prices)-1):
        if matix_prices[q] < matix_prices[q+1]:
            print best_order_dates[q].date(), best_order_dates[q+1].date()
            best_strategy.append([best_order_dates[q].date(), 'AAPL', 'BUY', 200])         
            #if q < len(matix_prices):
            best_strategy.append([best_order_dates[q+1].date(), 'AAPL', 'SELL', 200])
        if matix_prices[q] > matix_prices[q+1]:
            best_strategy.append([best_order_dates[q].date(), 'AAPL', 'SELL', 200])
            #if q < len(matix_prices):
            best_strategy.append([best_order_dates[q+1].date(), 'AAPL', 'BUY', 200])
    for _best_strategy in best_strategy:
        print >> best_strategy_file, ",".join(str(x) for x in _best_strategy)

    #norm_stock_prices = stock_prices/stock_prices,.ix[0,:] # normalizing data
    #norm_stock_prices = norm_prices*allocs*sv # multiplying by allocation and starting stock value
    #port_val = norm_stock_prices.sum(axis=1) # summing to get daily portfolio values

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols

if __name__ == "__main__":
    test_code()
