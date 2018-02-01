"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data
import matplotlib.pyplot as plt

def author():
        return 'enaziga3'

def compute_portvals(orders_file = "ml_orders.csv", sd = dt.datetime(2010,1,1), ed = dt.datetime(2011,12,31), start_val = 100000):

    over_leveraged = True

    # Read in orders csv file, sort on Date, get date range and symbols
    df_orders = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])
    df_orders = df_orders.sort_index()
    start_date = sd #df_orders.index[0]
    end_date = ed #df_orders.index[-1]
    dates = pd.date_range(start_date, end_date)
    symbols = list(set(df_orders['Symbol']))

    # Get the adjusted close prices for the extracted symbols between start_date and end_date 
    df_prices_all = get_data(symbols, dates, addSPY=True, colname = 'Adj Close')
    df_prices = df_prices_all[symbols]
    start_date = df_prices.index[0]
    end_date = df_prices.index[-1]
    dates = pd.date_range(start_date, end_date)
    df_prices['Cash'] = pd.Series(1.0, index=df_prices.index)

    while over_leveraged:

        df_trades = df_prices.copy()
        df_trades[:] = 0.0

        for row in df_orders.groupby(df_orders.index): 
            rDates  = row[0]
	    drows   = row[1]
            if len(drows) > 1: 
                for drow in drows.itertuples(): 
                    rSymbol = drow[1] 
                    action  = drow[2] 
                    amount  = drow[3] 
                    if action == 'BUY':
                        df_trades['Cash'].ix[rDates] = df_trades['Cash'].ix[rDates] + amount*df_prices[rSymbol].ix[rDates]*-1.0
                        df_trades[rSymbol].ix[rDates] = df_trades[rSymbol].ix[rDates] + amount
                    elif action == 'SELL':
                        df_trades['Cash'].ix[rDates] = df_trades['Cash'].ix[rDates] + amount*df_prices[rSymbol].ix[rDates]
                        df_trades[rSymbol].ix[rDates] = df_trades[rSymbol].ix[rDates] + amount*-1.0
            else:
                rSymbol = row[1]['Symbol'][0] 
                action  = row[1]['Order'][0] 
                amount  = row[1]['Shares'][0] 

                if action == 'BUY':
                    df_trades['Cash'].ix[rDates] = amount*df_prices[rSymbol].ix[rDates]*-1.0
                    df_trades[rSymbol].ix[rDates] = amount
                elif action == 'SELL':
                    df_trades['Cash'].ix[rDates] = amount*df_prices[rSymbol].ix[rDates]
                    df_trades[rSymbol].ix[rDates] = amount*-1.0
        df_holdings = df_trades.copy()
        df_holdings[:] = 0.0
        df_values = df_holdings.copy()

        # Populate holdings dataframe, calculate value of holdings and portfolio value
        df_holdings['Cash'].ix[start_date] = start_val

        df_holdings.ix[start_date] = df_holdings.ix[start_date] + df_trades.ix[start_date] 
        indices = df_trades.index
    
        for i in range(1,len(indices)):
            hDates  = indices[i-1]
            rDates  = indices[i]
            df_holdings.ix[rDates] = df_holdings.ix[hDates] + df_trades.ix[rDates]  

        df_values = df_holdings*df_prices 
        df_port_val = df_values.sum(axis=1)
        df_cash = df_values['Cash']
        df_values1 = df_values.drop('Cash',axis=1)
        df_leverage = (abs(df_values1).sum(axis=1))/((df_values1).sum(axis=1) + df_cash)
        df_leverage = pd.DataFrame(df_leverage,columns=['Leverage'])

        # Check if leverage is below 1.5 and update over_leveraged variable
        
        for lrow in df_leverage.itertuples(): 
            lDates  = lrow[0]
            if lrow[1] > np.inf:
                over_leveraged = True
                df_orders.set_value(lDates, 'Shares', 0.0) 
                break
            else:
                over_leveraged = False
    return df_port_val

def test_code():

    of = "ml_orders.csv"
    of2 = "calc_orders.csv"
    of3 = "benchmark.csv"

    #of = "test_ml_orders.csv"
    #of2 = "test_calc_orders.csv"
    #of3 = "test_benchmark.csv"
    sv = 100000
    rfr=0.0
    sf=252.0
    _start_date = dt.datetime(2008,1,1)
    _end_date = dt.datetime(2009,12,31)
    
    #_start_date = dt.datetime(2010,1,1)
    #_end_date = dt.datetime(2011,12,31)

    #test_start_date = dt.datetime(2010,1,1)
    #test_end_date = dt.datetime(2011,12,31)

    # Process orders
    ml_portvals = compute_portvals(orders_file = of, sd = _start_date, ed = _end_date, start_val = sv)
    manual_portval = compute_portvals(orders_file = of2, sd = _start_date, ed = _end_date, start_val = sv)
    bench_portval = compute_portvals(orders_file = of3, sd = _start_date, ed = _end_date, start_val = sv)

    # Plot ML strategy chart
    morders = pd.read_csv(of, index_col='Date', parse_dates=True, na_values=['nan'])
    morders = morders.sort_index()
    ml_dates = morders.index
    longs = []
    shorts = []

    for w in range(1,len(ml_dates)):
        if len(pd.date_range(ml_dates[w-1], ml_dates[w])) > 21:
            #print morders.ix[ml_dates[w]]['Order']
            if morders.ix[ml_dates[w]]['Order'] == 'BUY':
                longs.append(ml_dates[w])
            else:
                shorts.append(ml_dates[w])

    period_end = ml_portvals.ix[-1]
    commul = ml_portvals.pct_change()
    cum_ret   = (period_end-sv)/sv
    avg_daily_ret  = commul[1:].mean()
    std_daily_ret = commul[1:].std()
    sharpe_ratio   = np.sqrt(sf)*(commul[1:]-rfr).mean()/((commul[1:]-rfr).std())

    df_temp = pd.concat([ml_portvals, manual_portval, bench_portval], keys=['ML Strategy', 'Manual Strategy', 'Benchmark'], axis=1)
    df_temp = df_temp/df_temp.ix[0,:]
    pdates = df_temp.index

    styles = ['g-','b-','k-']
    linewidths = [2, 2, 2]
    fig, ax = plt.subplots()
    for col, style, lw in zip(df_temp.columns, styles, linewidths):
        df_temp[col].plot(style=style, lw=lw, ax=ax)
    ymin, ymax = ax.get_ylim()
    ax.vlines(x=longs, ymin=ymin, ymax=ymax+0.5, color='g', linestyles = 'dashed', linewidth = 1.5)
    ax.vlines(x=shorts, ymin=ymin, ymax=ymax+0.5, color='r', linestyles = 'dotted', linewidth = 1.5)
    plt.title('Comparing Strategies', fontsize=16)
    plt.ylabel('Normalized Portfolio', fontsize=16)
    plt.xlabel('Dates', fontsize=16)
    plt.ylim([0.9,1.5])
    plt.legend(loc="upper center")
    plt.savefig('compare_ml_man_bench_portval.png')
    #plt.show()

    # Plot manual strategy chart

    manuals = pd.read_csv(of2, index_col='Date', parse_dates=True, na_values=['nan'])
    manuals = manuals.sort_index()
    manuals_dates = manuals.index
    manuals_longs = []
    manuals_shorts = []

    for w in range(1,len(manuals_dates)):
        if len(pd.date_range(manuals_dates[w-1], manuals_dates[w])) > 21:
            #print manuals.ix[manuals_dates[w]]['Order']
            if manuals.ix[manuals_dates[w]]['Order'] == 'BUY':
                manuals_longs.append(manuals_dates[w])
            else:
                manuals_shorts.append(manuals_dates[w])

    manuals_df_temp = pd.concat([manual_portval, bench_portval], keys=['Manual Strategy', 'Benchmark'], axis=1)
    manuals_df_temp = manuals_df_temp/manuals_df_temp.ix[0,:]
    manuals_pdates = manuals_df_temp.index

    plt.clf()
    styles = ['b-','k-']
    linewidths = [2, 2]
    fig, ax = plt.subplots()
    for col, style, lw in zip(manuals_df_temp.columns, styles, linewidths):
        manuals_df_temp[col].plot(style=style, lw=lw, ax=ax)
    ymin, ymax = ax.get_ylim()
    ax.vlines(x=manuals_longs, ymin=ymin, ymax=ymax+0.5, color='g', linestyles = 'dashed', linewidth = 1.5)
    ax.vlines(x=manuals_shorts, ymin=ymin, ymax=ymax+0.5, color='r', linestyles = 'dotted', linewidth = 1.5)
    plt.title('Comparing Manual and Benchmark Strategies', fontsize=16)
    plt.ylabel('Normalized Portfolio', fontsize=16)
    plt.xlabel('Dates', fontsize=16)
    plt.ylim([0.9,1.5])
    plt.legend(loc="upper center")
    plt.savefig('manual_v_benchmark_plot.png')


    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(_start_date, _end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print
    print "Final Portfolio Value: {}".format(ml_portvals[-1])

if __name__ == "__main__":
    test_code()
