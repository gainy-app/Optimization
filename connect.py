# Dotenv to use environnment variables
import os
import dotenv

dotenv.load_dotenv("../.env")


DB_HOST = os.getenv('DB_HOST')
DB_USER = os.getenv('DB_USER')
DB_PASS = os.getenv('DB_PASS')
DB_PORT = os.getenv('DB_PORT')

EOD_KEY = os.getenv('EOD_KEY')


import psycopg2
import pandas as pd
import warnings
import json, requests 
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import scipy.optimize as sco
from datetime import datetime
from tqdm import tqdm
from my_utils import DateFromStr, DateToStr


warnings.filterwarnings('ignore')

def GetQuery(qry): 
    DBconn = psycopg2.connect(
        host=DB_HOST,
        database="gainy_external_access",
        user=DB_USER,
        password=DB_PASS,
        port=DB_PORT,
        options=f'-c search_path=gainy_analytics')
    
    tmp = pd.read_sql_query(qry, DBconn)
    DBconn.close()

    return tmp

def GetRussell():
    eod_key = EOD_KEY

    # Russell 1000
    url = requests.get("https://eodhistoricaldata.com/api/fundamentals/RUI.INDX?api_token="+eod_key)
    text = url.text
    r1000 = json.loads(text)

    # Russell 2000
    url = requests.get("https://eodhistoricaldata.com/api/fundamentals/RUT.INDX?api_token="+eod_key)
    text = url.text
    r2000 = json.loads(text)

    r1000 = pd.DataFrame(r1000['Components']).transpose()
    r2000 = pd.DataFrame(r2000['Components']).transpose()
    eod = pd.concat([r1000, r2000], ignore_index=True)
    eod = eod.drop_duplicates()

    return eod
    
def GetDividends(tickers, dt=None, lookback=36):
    if isinstance(tickers, str):
        tickers = "('"+tickers+"')"
    else:
        tickers = tuple(tickers)
       
        
    # Get start date
    if dt is None:
        start_date = (datetime.today() - relativedelta(months=lookback)).strftime('%Y-%m-%d')
        end_date = datetime.today().strftime('%Y-%m-%d')
    else:
        end_date = dt
        start_date = (DateFromStr(dt) - relativedelta(months=lookback)).strftime('%Y-%m-%d')
    
    qry = f"""SELECT code as ticker, recorddate, paymentdate, period, value, unadjustedvalue
            FROM eod_dividends 
            WHERE code IN {tickers} AND recorddate>='{start_date}' AND recorddate<='{end_date}'
            AND period in ('Quarterly', 'Monthly')
            ORDER BY code, recorddate DESC
            """
    tmp = GetQuery(qry)
    
    tmp['recorddate'] = tmp['recorddate'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        
    
    return tmp

def GetLastPrice(tickers, dt=None):
    if isinstance(tickers, str):
        tickers = "('"+tickers+"')"
    else:
        tickers = tuple(tickers)

    if dt is None:
        dt = datetime.today().strftime('%Y-%m-%d')

    qry = f"""
        SELECT p.code as ticker, d.max_date, p.adjusted_close
        FROM  eod_historical_prices p
        INNER JOIN 
            (SELECT code, MAX(date) as max_date
            FROM eod_historical_prices
            WHERE code IN {tickers} AND date <='{dt}'
            GROUP BY code) d
        ON d.code = p.code and d.max_date = p.date
    """
    
    tmp = GetQuery(qry)
    
    return tmp

def GetMetrics(tickers):
    if isinstance(tickers, str):
        tickers = "('"+tickers+"')"
    else:
        tickers = tuple(tickers)
    
    qry = f"""
        SELECT m.symbol as ticker, m.market_capitalization/1000000 as marketcap, m.net_income_ttm/1000000 as net_income,
        m.avg_volume_90d /1000000 as avg_vol_mil, m.price_change_1y as ret1y, m.dividend_payout_ratio,
        i.gic_group, i.gic_industry, i.gic_sector, i.gic_sub_industry
        FROM ticker_metrics m
        INNER JOIN base_tickers i
        ON m.symbol=i.symbol
        WHERE m.symbol IN {tickers}
        
    """
    
    tmp = GetQuery(qry)
    
    return tmp

def GetPrices(tickers, start, end):
    if isinstance(tickers, str):
        tickers = "('"+tickers+"')"
    else:
        tickers = tuple(tickers)
    
    qry = f"""
        SELECT code as ticker, date, adjusted_close as price
        FROM eod_historical_prices
        WHERE code IN {tickers} AND
        date between '{start}' AND '{end}'
    """
    
    tmp = GetQuery(qry)
    
    tmp = tmp.pivot(index='date', columns='ticker', values='price').rename_axis(None, axis=1) 
    tmp.index = pd.to_datetime(tmp.index)
    
    return tmp