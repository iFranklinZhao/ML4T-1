"""MC1-P2: Optimize a portfolio."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import scipy.optimize as spo
from util import get_data, plot_data


def assess_portfolio(allocs,prices):
       sv = 1000000.0
       rfr  = 0.0
       sf   = 252.0
       norm_prices = prices*allocs*sv 
       port_val = norm_prices.sum(axis=1) 

       period_end = port_val.ix[-1]
       commul = port_val.pct_change()
       cr   = (period_end-sv)/sv
       adr  = commul[1:].mean()
       sddr = commul[1:].std()
       sr   = np.sqrt(sf)*(commul[1:]-rfr).mean()/((commul[1:]-rfr).std())
       return [cr, adr, sddr, sr, port_val]


def run_assess_portfolio(allocs,prices):
   return -1.0*assess_portfolio(allocs,prices)[3]

def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=['GOOG','AAPL','GLD','XOM'], gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates) 
    prices = prices_all[syms] 
    prices_SPY = prices_all['SPY'] 
    prices = prices/prices.ix[0,:]

    noa = len(syms)
    allocs = noa*[1.0/noa,]
    bnds  =  tuple((0,1)  for x in range(noa))
    cons = ({'type':'eq','fun': lambda x:np.sum(x)-1})
    results = spo.minimize(run_assess_portfolio,allocs,args=(prices,),method='SLSQP',bounds=bnds,constraints=cons)
    cr, adr, sddr, sr, port_val = assess_portfolio(prices,results.x)
    allocs = results.x    

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        df_temp = df_temp/df_temp.ix[0,:]
        ax = df_temp.plot(title='Daily Portfolio Value and SPY')
        ax.set_ylabel('Normalized Prices')
        ax.set_xlabel('Dates')
        plt.savefig('comparison_optimal.png')

    return allocs, cr, adr, sddr, sr

def test_code():

    start_date = dt.datetime(2010,1,1)
    end_date = dt.datetime(2010,12,31)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        gen_plot = False)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr

if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called
    test_code()
