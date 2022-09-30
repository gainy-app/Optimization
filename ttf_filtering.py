import pandas as pd
from connect import *
import numpy as np

def FilterTTF(tickers, minMC=100, minVolume=2, minPrice=1, verbatim=False):
    '''
    Filters stocks in a ttf given tickers and the following logic:

    1. MarketCap > 100 mln
    2. Average daily volume > $2mln (note Dollars not Shares)
    3. Price > $1 per share 
    4. Lst available price date == max of all tickers
    
    '''

    df = GetMetrics(tickers)
    df = df[['ticker', 'marketcap', 'avg_vol_mil']]

    lp = GetLastPrice(tickers)

    df = df.merge(lp, on='ticker', how='inner')
    df['vol_doll'] = df.avg_vol_mil * df.adjusted_close

    # Filtering logic
    df['Flag'] = np.NaN

    df.loc[df.marketcap<minMC, 'Flag'] = str(df.loc[df.marketcap<minMC, 'Flag'].values)+ '- MC'
    df.loc[df.adjusted_close<minPrice, 'Flag'] = str(df.loc[df.adjusted_close<minPrice, 'Flag'].values) + '-Price'
    df.loc[df.vol_doll<minVolume, 'Flag'] = str(df.loc[df.vol_doll<minVolume, 'Flag'].values) + '-Volume'

    maxdt = df.max_date.max()
    df.loc[df.max_date<maxdt, 'Flag'] = str(df.loc[df.max_date<maxdt, 'Flag']) + '-Date'

    if verbatim:
        df.Flag = df.Flag.str.replace('\[nan\]-','')
        print(df)


    out = df.loc[df.Flag.isna(), 'ticker']
    return out


if __name__ == '__main__':
    tickers = ['AAPL', 'GOOG', 'TSLA', 'ABNB', 'VFF']

    test = FilterTTF(tickers, verbatim=True)
    print(test)
