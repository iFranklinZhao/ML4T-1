"""MC1-P1: Analyze a portfolio."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data

# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def generate_indicators(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,12,31), \
    syms = ['AAPL'], \
    allocs=[1.0], \
    sv=1000000, rfr=0.0, sf=252.0, \
    gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # Get daily portfolio value
    norm_prices = prices/prices.ix[0,:] # normalizing data
    norm_prices = norm_prices*allocs*sv # multiplying by allocation and starting stock value
    port_val = norm_prices.sum(axis=1) # summing to get daily portfolio values

    # Get portfolio statistics (note: std_daily_ret = volatility)
    period_end = port_val.ix[-1]
    commul = port_val.pct_change()
    commul1 = commul[1:]
    cr   = (period_end-sv)/sv
    adr  = commul[1:].mean() 
    sddr = commul[1:].std() 
    sr   = np.sqrt(sf)*(commul[1:]-rfr).mean()/((commul[1:]-rfr).std())

    # Calculate indicators

    # Momentum indicator
    window = int(21)
    momentum = []
    simple_moving_avg = []
    bollenger_bands = []

    price_array = pd.DataFrame.as_matrix(prices)
    for i in range(prices.shape[0]):
        if  i > window:
            momentum_value = (price_array[i]/price_array[i-window]) - 1.0
            simple_moving_avg_value = np.mean(price_array[i-window:i])
            bollenger_bands = (price_array[i]/simple_moving_avg_value)/np.std(price_array[:i])
            momentum.append(momentum_value)
            simple_moving_avg.append(simple_moving_avg_value)
    
    momentum = np.array(momentum)
    simple_moving_avg = np.array(simple_moving_avg)
    print bollenger_bands
    norm_price = (price_array-np.mean(price_array))/np.std(price_array)
    norm_momentum = (momentum-np.mean(momentum))/np.std(momentum)
    norm_simple_moving_avg = (simple_moving_avg-np.mean(simple_moving_avg))/np.std(simple_moving_avg)

    print "Mean of Normalized Momentum: ", np.mean(norm_momentum)
    print "Standard Deviation of Normalized Momentum: ", np.std(norm_momentum)
    plt.plot(norm_momentum)
    plt.plot(norm_simple_moving_avg)
    plt.plot(norm_price)
    #plt.show()

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        df_temp = df_temp/df_temp.ix[0,:]
        ax = df_temp.plot(title='Daily Portfolio Value and SPY')
        ax.set_ylabel('Normalized Prices')
        ax.set_xlabel('Dates')
        plt.savefig('plot.png')

    # Add code here to properly compute end value
    ev = period_end

    return cr, adr, sddr, sr, ev

def test_code():
    # This code WILL NOT be tested by the auto grader
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2009,12,31)
    symbols = ['AAPL']
    allocations = [1.0]
    start_val = 1000000  
    risk_free_rate = 0.0
    sample_freq = 252

    # Assess the portfolio
    cr, adr, sddr, sr, ev = generate_indicators(sd = start_date, ed = end_date,\
        syms = symbols, \
        allocs = allocations,\
        sv = start_val, \
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
    test_code()
