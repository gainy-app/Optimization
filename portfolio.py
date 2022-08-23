# Portfolio functions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
import scipy.optimize as sco
from datetime import datetime
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from connect import GetQuery, GetRussell, GetLastPrice, GetPrices
import warnings
import logging

def CalcBeta(ret, benchmark='SPY', truncate = [-1,2]):
    '''
    Calculates market beta for each stock. 
    Inputs:
    ret - dataframe with returns
    benchmark - ticker for the benchmark. Default - SPY
    trancate - low and upper limit for betas
    
    Output:
    Beta dictionary. Keys - tickers
    '''
    
    # Get start end end date
    start_dt = str((ret.index.min() - relativedelta(days=5)).date()) # Go backwards in case of holidays
    end_dt = str(ret.index.max().date())
    
    bm = GetPrices("SPY", start_dt, end_dt).pct_change().loc[ret.index.min():ret.index.max()]
    
    # Regression
    out = dict()
    
    for ticker in ret.columns:
        tmp = ret[[ticker]].merge(bm, left_index=True, right_index=True, how='left').dropna()

        X = tmp.iloc[:,0].values.reshape(-1,1)
        y = tmp.iloc[:,1].values
        
        coef = LinearRegression().fit(X, y).coef_[0]
        coef = truncate[0] if coef<truncate[0] else truncate[1] if coef>truncate[1] else coef
        out[ticker] = coef

    return out

def OptimizePortfolioSharpe(r, cov, industries, betas, penalties, target_beta = 1, bounds = (0.0,1.0), verbatim=False ):
    '''
    Portfolio optimization, Sharpe-like
    Inputs:
    - r - mean returns (or any other numerator) as dictionary
    - cov - covariance matrix
    - industries - industries map indexed by ticker
    - betas - stocks betas indexed by tickers
    - target_beta - target portfolio beta
    - bounds - tuple. Stock minimum and maximum weight
    - verbatim - if True prints out all optimization coefficients and optimization output
    
    Pentalties - penalty coefficients dictionary
        - hs - HHI index penalty for stocks
        - hi - HHI index penalty for industries
        - b - beta over target penalty
    
    Output
    - success 
    - optimal weights dictionary
    '''
    

    
    # Check that all inputs are aligned and keep arrays
    tickers = cov.columns
    r = np.array([r[ticker] for ticker in tickers])
    industries = np.array([industries[ticker] for ticker in tickers])
    betas = np.array([betas[ticker] for ticker in tickers])
    sigma = cov.values
    
    # Define functions for optimization
    def numerator(weights):
        return (np.sum(r * weights)) 
    
    def portfolio_sd(weights):
        return np.sqrt(np.transpose(weights) @ (sigma) @ weights)
    
    def hhi_stock(weights):
        return np.sum(weights**2) * penalties['hs'] # HHI concentration index
    
    def hhi_ind(weights):
        tmp = pd.DataFrame({'W':weights,'Ind':industries}, index=range(len(industries)))
        tmp = tmp.groupby('Ind').W.sum().values
        
        return np.sum(tmp**2) * penalties['hi'] # HHI concentration index
    
    def beta_pen(weights):
        port_beta = np.sum(betas * weights)
        return ((target_beta - port_beta)**2)* penalties['b']
    
    def obj_fun(weights):
        fnc = (numerator(weights) / portfolio_sd(weights)) - hhi_stock(weights) - hhi_ind(weights) - beta_pen(weights)
        return - fnc # Minus to turn into minimization problem
    
    # Constraints
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) # Fully invested
    bounds = tuple((bounds[0], bounds[1]) for x in r) 
    
    opt_res = sco.minimize(
      fun = obj_fun, # Objective
      x0 = np.repeat(1/len(r), len(r)), # Initial guess - equal weighted
      method = 'SLSQP',
      bounds = bounds, 
      constraints = constraints
    )
    out = pd.DataFrame({'Weight':opt_res.x} , index = tickers).sort_values('Weight', ascending=False)
    
    weights = np.repeat(1/len(r), len(r))
    if verbatim:
        print("_"*50,"\n"
            f"Success: {opt_res.success}\n",
            f"Weights: {out}\n",
            "_"*50,"\n",
            f"Objective function components with equal weights:\n"
            f" 1: {(numerator(weights) / portfolio_sd(weights))}\n",
            f"2: {hhi_stock(weights)}\n",
            f"3: {hhi_ind(weights)}\n",
            f"4: {beta_pen(weights)}\n",
            "_"*50,"\n",
            f"Objective function components with optimized weights:\n"
            f" 1: Numerator: {round(numerator(opt_res.x),4)} Denom: {round(portfolio_sd(opt_res.x),4)} Value: {numerator(opt_res.x)/portfolio_sd(opt_res.x)}\n",
            f"2: {hhi_stock(opt_res.x)}\n",
            f"3: {hhi_ind(opt_res.x)}\n",
            f"4: {beta_pen(opt_res.x)}\n"
                
             )
          
    return out.Weight.to_dict()    

def ReturnPortfolio(R, weights="equal", rebalance_on = 'never', geometric=True, force_rebalance=False, return_weights=False):
    '''
    Inputs: 

    R - dataframe with stock returns. Index - date. Only 
    weights - dataframe with index as dates of rebalancing or dictionary {'Ticker':weight}  or 'equal'
    rebalance on: 'never', "months", "quarters". If weights provided with schedule -- ignored
    geometric - geometric returns (default) or arithmetic (False)
    force_rebalance: rebalance weights to 1 when calculating geometric. If false, residual returns are zero
    return_weight: boolean. Whether to return bop and eop weights
    
    Outputs
    ret - portfolio return. Dataframe indexed by dates

    If return_weights = True:
        bop_weights - daily BOP weights
        eop_weights - daily EOP weights

    '''
    
    value=1 # Portfolio value
    
    # Fill na returns
    if (R.isna().sum().sum())>0:
        R[R.isna()]=0
        logging.warning("Returns have missing days. Filled with zeros")
        warnings.warn("Returns have missing days. Filled with zeros")
        
    # If weight is none - equally weighted
    if (isinstance(weights, str)):
        if weights == 'equal':
            weights = {ticker:1/R.shape[1] for ticker in R.columns}
    
    # If weights is not in dataframe, create dataframe using weights as starting weights
    if (isinstance(weights, dict)):
        weights = pd.DataFrame.from_dict({min(R.index.date-relativedelta(days=1)):weights}, orient='index')
    
    
    # Rebalance on frequency
    
    if ((weights.shape[0]==1) & (rebalance_on == 'months')): # Works if schedule is not provided
        w_start = str(weights.index.min())
        w_idx = [w_start] + list(pd.date_range(start=w_start, end= R.index.max(), freq='M'))
        weights = pd.DataFrame(weights, index=pd.to_datetime(w_idx)).ffill()

        
    if ((weights.shape[0]==1) & (rebalance_on == 'quarters')): # Works if schedule is not provided
        w_start = str(weights.index.min())
        w_idx = [w_start] + list(pd.date_range(start=w_start, end= R.index.max(), freq='Q'))
        weights = pd.DataFrame(weights, index=pd.to_datetime(w_idx)).ffill()
        
    
    # Check that we have weights and returns for all
    nR = R.shape[1]
    nw = weights.shape[1]
    R = R[list(set(R.columns) & set(weights.columns))]
    
    if R.shape[1]!=nw:
        raise ValueError('Not all returns are found. Provide returns for each instrument in the portfolio')
        logging.error("Not all returns are found. Provide returns for each instrument in the portfolio")
            
    # Align weights and returns
    weights = weights[R.columns]
    
    # Check that returns dates match
    if (R.index.max()<weights.index.min()):
        raise ValueError("Last date in Returns happens sooner than first date of Weight")
    
    if (weights.index.min()>R.index.min()):
        R = R.loc[R.index.date>weights.index.min(),:]
    
    # Make sure we have the same date format
    R.index = pd.to_datetime(R.index).date
    weights.index = pd.to_datetime(weights.index).date
    
    # Calculate portfolio returns
    # Arithmetic return
    if geometric==False:
        
        # Create placeholder
        R_idx = R.loc[R.index>weights.index.min(),:].index
        bop_weights = pd.DataFrame(0, index = R_idx, columns = R.columns) # Beginning op period weights
        eop_weights = bop_weights.copy()  # End of period weights
        period_contrib = bop_weights.copy() # Contributions
        ret = pd.DataFrame(0, index=R_idx, columns=['Portfolio'])
        
        # Loop through periods
        k=0
        
        for i in range(weights.shape[0]):
            # Find rebalance from and to
            dt_from = weights.index[i] + relativedelta(days=1)
            
            if (i == (weights.shape[0]-1)):
                dt_to = R.index.max()
            else:
                dt_to = weights.index[(i+1)]
            
            returns = R.loc[dt_from:dt_to,]
            
            # Loop within returns
            # Inner counter
            if returns.shape[0]>=1:
            
                for j in range(returns.shape[0]):
                    if j==0:
                        bop_weights.iloc[k,:] = weights.iloc[i,:]
                    else:
                        bop_weights.iloc[k,:] = eop_weights.iloc[k-1,]

                    period_contrib.iloc[k,:] = returns.iloc[j,:] * bop_weights.iloc[k,:]
                    
                    eop_weights.iloc[k,:] =\
                    (period_contrib.iloc[k,:] + bop_weights.iloc[k,:])/\
                    np.append(period_contrib.iloc[k,:], bop_weights.iloc[k,:]).sum()
                    
                    ret.iloc[k,0] = period_contrib.iloc[k,:].sum()
                    
                    k+=1
                   
    # Geometric return               
    if geometric:
        
        # Check that all weights sum up to 1
        if sum(round(weights.sum(axis=1),8)!=1)>0:
            if force_rebalance:
                warnings.warn("Some weights do not add up to 1. Rebalancing to 1...")
                logging.warn("Some weights do not add up to 1. Rebalancing to 1...")
                
                weights = weights.apply(lambda x: x/sum(x), axis=1)
            else:
                warnings.warn("Some weights do not add up to 1. Residual will be treated as having zero returns.")
                logging.warn("Some weights do not add up to 1. Residual will be treated as having zero returns.")
                
                weights['Residual'] = 1-weights.sum(axis=1)
                R['Residual'] = 0.0
        

        # Placeholder
        R_idx = R.loc[R.index>weights.index.min(),:].index
        bop_weights = pd.DataFrame(0, index = R_idx, columns = R.columns) # Beginning of period weights
        eop_weights = bop_weights.copy()  # End of period weights
        bop_value = bop_weights.copy() # Beginning of period  value
        eop_value = bop_weights.copy()  # End of period value
        period_contrib = bop_weights.copy() # Contributions
        
        ret = pd.DataFrame(0, index=R_idx, columns=['Portfolio'])
        eop_value_total = pd.DataFrame(0, index=R_idx, columns=['Portfolio'])
        bop_value_total = pd.DataFrame(0, index=R_idx, columns=['Portfolio'])
        end_value = value
        

        # Loop through periods
        k=0
        
        for i in range(weights.shape[0]):
            # Find rebalance from and to
            dt_from = weights.index[i] + relativedelta(days=1)
            
            if (i == (weights.shape[0]-1)):
                dt_to = R.index.max()
            else:
                dt_to = weights.index[(i+1)]
            
            returns = R.loc[dt_from:dt_to,]
            
            # Loop within returns
            # Inner counter
            if returns.shape[0]>=1:
                jj=0
                for j in range(returns.shape[0]):
                    if jj==0:
                        bop_value.iloc[k,:] = end_value * weights.iloc[i,:]
                    else:
                        bop_value.iloc[k,:] = eop_value.iloc[k-1,:]
                        
                                        
                    bop_value_total.Portfolio.iloc[k] = bop_value.iloc[k,:].sum()
                    
                    # End of period values
                    eop_value.iloc[k,:] = (1+returns.iloc[j,:])*bop_value.iloc[k,:]
                    
                    eop_value_total.Portfolio.iloc[k] = eop_value.iloc[k,:].sum()
                    
                    
                    # Calculate weights
                    bop_weights.iloc[k,:] = bop_value.iloc[k,:] / bop_value_total.Portfolio[k]
                    eop_weights.iloc[k,:] = eop_value.iloc[k,:] / eop_value_total.Portfolio[k]
                    
                    # Calculate contributions
                    period_contrib.iloc[k,:] = returns.iloc[j,:] * bop_value.iloc[k,:] / bop_value_total.Portfolio.iloc[k]
                    
                    # Populate portfolio return    
                    ret.iloc[k,0] = period_contrib.iloc[k,:].sum()
                    ret.columns = ['Portfolio']
                    
                    end_value = eop_value_total.Portfolio[k]
                    
                    
                    k+=1
                    jj+=1

        
        
    if (return_weights):
        return ret, bop_weights, eop_weights 
    else:
        return ret