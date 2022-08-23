import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import scipy.optimize as sco
from sklearn.linear_model import LinearRegression
from connect import GetQuery, GetPrices
import dateutil.parser
import logging


class GainyOptimizer():

    def __init__(self, tickers, date_today, lookback = 9, benchmark='SPY', industry_type='gic_sector') -> None:
        self.tickers = tickers # Tickers list
        self.dt = dateutil.parser.parse(date_today).date() # Date of optimization
        self.start_dt = self.dt - relativedelta(months=lookback)
        self.lookback = lookback
        self.benchmark = benchmark
        self.ind_type = industry_type

    def __str__(self):
        return "Tickers: %s  \nBenchmark: %s\nOptimization Date: %s \nStart date: %s\n" % (self.tickers, self.benchmark, self.dt, self.start_dt)

    def StockReturns(self, tickers=None):
        '''
        Returns stock returns for the selected lookback window
        '''

        if (tickers is None):
            ticker=self.tickers

        tickers = tickers + [self.benchmark]

        rets = GetPrices(tickers, self.start_dt - relativedelta(days=5), self.dt).pct_change()
        rets = rets[str(self.start_dt):str(self.dt):]

        # Check that every ticker has at least 80% of non-nas
        min_obs = rets.shape[0]
        obs = rets.count()

        missing = obs[obs<min_obs]

        if (missing.shape[0]>0):
            logging.warning(f"\nThe following tickers have less than 80% of price observations: {missing.index.values}\nThey will be dropped")
            rets = rets.drop(missing.index, axis=1)

        # Check that we have data for all names
        missing_tickers = list(set(tickers) - set(rets.columns))
        if (len(missing_tickers)>0):
            logging.warning(f"\nWe do not support the following tickers {missing_tickers}.\nThey will be dropped from the optimization\n")
            

        return rets
    
    def StockMetrics(self, override_numerator=None):
        '''
        Get key metrics for optimization and create nested dictionary for optimization
        '''

        if override_numerator is None:
            tickers=self.tickers    

        else:
            tickers = list(override_numerator.keys())
            missing_tickers = list(set(self.tickers) - set(tickers))
            if (len(missing_tickers)>0):
                logging.warning(f"\nYou did not provide numerator for the following tickers{missing_tickers}.\nThey will be dropped from the optimization\n")

        
        rets = self.StockReturns(tickers+[self.benchmark])
        bm = rets[self.benchmark]
        rets = rets.drop(self.benchmark, axis=1)
        
        # Covariance
        cov = rets.cov()*252

        # Record numerator
        if override_numerator is None:
            numerator = (rets.mean()*252).to_dict()
        else:
            numerator = override_numerator

        # Get industries
        qry = f"select symbol as ticker, {self.ind_type} as industry from base_tickers where symbol in {tuple(tickers)} "
        industry = GetQuery(qry).set_index('ticker').industry.to_dict()
        

        # Get betas
        betas = dict()
        for ticker in rets.columns:
            tmp = rets[[ticker]].merge(bm, left_index=True, right_index=True, how='left').dropna()

            X = tmp.iloc[:,1].values.reshape(-1,1)
            y = tmp.iloc[:,0].values
            
            coef = LinearRegression(fit_intercept = True).fit(X, y).coef_[0]
            truncate = (-3,3) # Truncate betas in case of crazy numbers
            coef = truncate[0] if coef<truncate[0] else truncate[1] if coef>truncate[1] else coef
            betas[ticker] = coef
    
        return {'Numerator': numerator, 'Covariance':cov, 'Industry':industry, 'Betas':betas}

    def OptimizeSharpe(self, params=None, override_numerator=None, verbatim=False):
        '''
        Params - a dictionary with:

            Pentalties - penalty coefficients dictionary (default=1)
                - hs - HHI index penalty for stocks
                - hi - HHI index penalty for industries
                - b - beta over target penalty
            
            Bounds - tupple with minimum and maximum stock weight (default = (0,1))

            TargetBeta - float with target portfolio beta (default = 1)

        Output:
            success - boolean of the optimization success
            optimal weights dictionary

        '''
        
        # Get metrics
        stock_metrics = self.StockMetrics(override_numerator)

        r = stock_metrics['Numerator']
        cov = stock_metrics['Covariance']
        industries = stock_metrics['Industry']
        betas = stock_metrics['Betas']



        # Check that all inputs are aligned and keep arrays
        tickers = cov.columns
        r = np.array([r[ticker] for ticker in tickers])
        industries = np.array([industries[ticker] for ticker in tickers])
        betas = np.array([betas[ticker] for ticker in tickers])
        sigma = cov.values


        # Create and/or overwrite optimization parameters
        penalties = {'hs':1.0, 'hi':1.0, 'b':1.0}
        bounds = (0,1)
        target_beta = 1

        # Overwride params
        if (params is not None):
            if('bounds' in params.keys()):
                if (params['bounds'][1]*len(tickers)>1): # To avoid lack of solution for short list 
                    bounds = params['bounds']
                else:
                    bounds = (params['bounds'][0],1)
            if('penalties' in params.keys()):
                tmp_ = params['penalties']
                for key, value in tmp_.items():
                    penalties[key] = value
            if('target beta' in params.keys()):
                target_beta = params['target beta']

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
                f" Sharpe: {(numerator(weights) / portfolio_sd(weights))}\n",
                f"Stock HHI: {hhi_stock(weights)}\n",
                f"Industry HHI: {hhi_ind(weights)}\n",
                f"Beta penalty: {beta_pen(weights)}\n",
                "_"*50,"\n",
                f"Objective function components with optimized weights:\n"
                f" Sharpe: Numerator: {round(numerator(opt_res.x),4)} Denom: {round(portfolio_sd(opt_res.x),4)} Value: {numerator(opt_res.x)/portfolio_sd(opt_res.x)}\n",
                f"Stock HHI: {hhi_stock(opt_res.x)}\n",
                f"Industry HHI: {hhi_ind(opt_res.x)}\n",
                f"Beta penalty: {beta_pen(opt_res.x)}\n"
                    
                )
            
        return out.Weight.to_dict()
    
    def OptimizePortfolioRiskBudget(self, params=None, override_numerator=None, verbatim=False):
        '''
        Portfolio optimization, Sharpe-like
        Inputs:
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
    

        # Get metrics
        stock_metrics = self.StockMetrics(override_numerator)

        cov = stock_metrics['Covariance']
        industries = stock_metrics['Industry']
        betas = stock_metrics['Betas']



        # Check that all inputs are aligned and keep arrays
        tickers = cov.columns
        industries = np.array([industries[ticker] for ticker in tickers])
        betas = np.array([betas[ticker] for ticker in tickers])
        sigma = cov.values


        # Create and/or overwrite optimization parameters
        penalties = {'hs':1.0, 'hi':1.0, 'b':1.0}
        bounds = (0,1)
        target_beta = 1

        # Overwride params
        if (params is not None):
            if('bounds' in params.keys()):
                if (params['bounds'][1]*len(tickers)>1): # To avoid lack of solution for short list 
                    bounds = params['bounds']
                else:
                    bounds = (params['bounds'][0],1)
            if('penalties' in params.keys()):
                tmp_ = params['penalties']
                for key, value in tmp_.items():
                    penalties[key] = value
            if('target beta' in params.keys()):
                target_beta = params['target beta']

        # Equal risk budget
        w_t = [1/len(tickers) for ticker in tickers]
        
        # Define functions for optimization

        
        def portfolio_sd(weights):
            return np.sqrt(np.transpose(weights) @ (sigma) @ weights)
        
        def portfolio_var(weights):
            return np.transpose(weights) @ (sigma) @ weights

        # Risk contribution of assets
        def risk_contribution(weights):
            # function that calculates asset contribution to total risk
            w = np.matrix(weights)
            portvol = portfolio_sd(weights)
            # Marginal Risk Contribution
            MRC = sigma*w.T
            # Risk Contribution
            RC = np.multiply(MRC,w.T)/portvol
            return RC
        
        
        
        def hhi_stock(weights):
            return np.sum(weights**2) * penalties['hs'] # HHI concentration index
        
        def hhi_ind(weights):
            tmp = pd.DataFrame({'W':weights,'Ind':industries}, index=range(len(industries)))
            tmp = tmp.groupby('Ind').W.sum().values
            
            return np.sum(tmp**2) * penalties['hi'] # HHI concentration index
        
        def beta_pen(weights):
            port_beta = np.sum(betas * weights)
            return ((target_beta - port_beta)**2)* penalties['b']
        
        def risk_budget_obj(weights):
            portvol = portfolio_sd(weights)  
            risk_target = np.asmatrix(np.multiply(portvol,w_t))
            asset_RC = risk_contribution(weights)
            RB = sum(np.square(asset_RC-risk_target.T))[0,0] # sum of squared error
            return RB
        
        def obj_fun(weights):
            RB = risk_budget_obj(weights)
            fnc = RB + hhi_stock(weights) + hhi_ind(weights) + beta_pen(weights)
            return fnc 
        
        # Constraints
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) # Fully invested
        bounds = tuple((bounds[0], bounds[1]) for x in tickers) 
        
        opt_res = sco.minimize(
        fun = obj_fun, # Objective
        x0 = np.repeat(1/len(tickers), len(tickers)), # Initial guess - equal weighted
        method = 'SLSQP',
        bounds = bounds, 
        constraints = constraints
        )
        out = pd.DataFrame({'Weight':opt_res.x} , index = tickers).sort_values('Weight', ascending=False)
        
        weights = np.repeat(1/len(tickers), len(tickers))
        if verbatim:
            print("_"*50,"\n"
                f"Success: {opt_res.success}\n",
                f"Weights: {out}\n",
                "_"*50,"\n",
                f"Objective function components with equal weights:\n"
                f" Risk budget: {risk_budget_obj(weights)}\n",
                f"Stock HHI: {hhi_stock(weights)}\n",
                f"Industry HHI: {hhi_ind(weights)}\n",
                f"Beta penalty: {beta_pen(weights)}\n",
                "_"*50,"\n",
                f"Objective function components with optimized weights:\n"
                f" Risk Budget: \nRisk contribution: {risk_contribution(opt_res.x)} \nRisk budget term: {risk_budget_obj(opt_res.x)}\n",
                f"Stock HHI: {hhi_stock(opt_res.x)}\n",
                f"Industry HHI: {hhi_ind(opt_res.x)}\n",
                f"Beta penalty: {beta_pen(opt_res.x)}\n"
                    
                )
            
        return out.Weight.to_dict() 

        



if __name__=='__main__':

    
    test = GainyOptimizer(['AAPL', 'GOOG', 'TSLA', 'ABNB'], "2021-01-10")

    test.OptimizePortfolioRiskBudget(verbatim=True)

    
