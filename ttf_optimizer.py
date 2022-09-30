

import os
from datetime import datetime
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from connect import GetQuery, GetPrices
from tqdm import tqdm
from optimizer import GainyOptimizer
import sys
import getopt
from ttf_filtering import FilterTTF



def GetTTFComposition_byID(ttf_id):
    '''
    Uses TTF id to download the current TTF composition
    '''

    qry = f"""select symbol as ticker, weight
        from ticker_collections
        where collection_id='{ttf_id}' and 
        "_sdc_extracted_at"  = 
        (select max("_sdc_extracted_at") from ticker_collections
        where collection_id='{ttf_id}')

        """

    tmp = GetQuery(qry)
    tmp = tmp.drop_duplicates()
    
    return tmp

def TTFOptimization(ttf_id, dt_today, params):
    '''
    Runs optimization for a given day
    
    '''

    # Get tickers in the TTF
    tickers = list(GetTTFComposition_byID(ttf_id).ticker.unique())

    # Filter ttfs
    post_filter = FilterTTF(tickers, verbatim=False)
    tickers = [ticker for ticker in post_filter ]


    # Run optimizer
    optimizer = GainyOptimizer(tickers, dt_today, benchmark='SPY', lookback=9)
    opt_res = optimizer.OptimizePortfolioRiskBudget(params=params)

    return opt_res




if __name__=='__main__':
    params = {'bounds': (0.01, 0.3), 'penalties': {'hs': 0.005, 'hi': 0.005, 'b': 0.05}}
    
    
    arg_help = "{0} -i <id> -d <date> -o <output>".format(sys.argv[0])
    arg_help = f"""
    The parameters format is \n
    {arg_help} \n
    - id - TTF id
    - date - date of optimization as int YYYYMMDD
    - output - csv file path to append 
    
    """

    if not sys.argv[1:]:
        print(arg_help)
        sys.exit(2)

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:d:o:", ["help", "id=", 
        "date=", "output="])
    except:
        print(arg_help)
        sys.exit(2)


    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # print the help message
            sys.exit(2)
        elif opt in ("-i", "--input"):
            arg_id = arg
        elif opt in ("-d", "--date"):
            arg_date = arg
        elif opt in ("-o", "--output"):
            arg_output = arg

    print('id:', arg_id)
    print('date:', arg_date)

    ttf_id = arg_id
    dt_today = datetime.strftime(datetime.strptime(arg_date, "%Y%m%d"), "%Y-%m-%d")
    
    
    opt_res = TTFOptimization(ttf_id, dt_today, params)
    opt_res = pd.DataFrame.from_dict(opt_res, orient="index").reset_index()
    opt_res.columns = ['symbol', 'weight']
    opt_res['date'] = dt_today
    opt_res['ttf_id'] = ttf_id
    opt_res['opimized_at'] = datetime.now()

    opt_res.to_csv(arg_output, index=False, mode='a', header=(not os.path.exists(arg_output)))

