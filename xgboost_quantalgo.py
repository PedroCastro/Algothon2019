# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 11:52:02 2019

@author: alexm
"""

from quantopian.pipeline import Pipeline
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import Latest, Returns, RSI, EWMA as ewma, SimpleMovingAverage
from quantopian.pipeline.filters import QTradableStocksUS, Q1500US
from quantopian.pipeline.factors import CustomFactor, AverageDollarVolume, RollingLinearRegressionOfReturns, DailyReturns, AnnualizedVolatility,SimpleBeta
from quantopian.pipeline.data import Fundamentals
from quantopian.pipeline.experimental import risk_loading_pipeline
import quantopian.algorithm as algo
import quantopian.optimize as opt
import math
from quantopian.pipeline.data.quandl import cboe_vix
from quantopian.pipeline.data.user_57668ee56fca2390e600024a import algothon_2,algothon_3
 
from scipy.stats.mstats import zscore, winsorize
 
import numpy as np
import pandas as pd
from sklearn.svm import SVR
 
def util_winsor5(x):
    return (winsorize(x,limits=[0.05, 0.05]))
 
def beta_fitconvert(a):
    z = a/np.nanmax(a)
    z = z-np.nanmin(z)
    return(z)
 
class Sector(CustomFactor):
    
    inputs = [morningstar.asset_classification.morningstar_sector_code]
    window_length = 1
    
    def compute(self, today, assets, out, morningstar_sector_code):
        out[:] = morningstar_sector_code[0]        
        
 
def nanzscore(a):
    z = a                    # initialise array for zscores
    z[~np.isnan(a)] = util_winsor5(z[~np.isnan(z)])
    z[~np.isnan(a)] = zscore(z[~np.isnan(z)])    
    return(z)
 
class Volatility(CustomFactor):
    
    inputs = [AnnualizedVolatility()]
    window_length = 1
    
    def compute(self, today, assets, out, AnnualizedVolatility):  
        out[:] =  ((AnnualizedVolatility))
 
 
 
MAX_EXPO = 10
MAX_GROSS_LEVERAGE = 1
TOTAL_POSITIONS = 200
 
# Here we define the maximum position size that can be held for any
# given stock. If you have a different idea of what these maximum
# sizes should be, feel free to change them. Keep in mind that the
# optimizer needs some leeway in order to operate. Namely, if your
# maximum is too small, the optimizer may be overly-constrained.
MAX_SHORT_POSITION_SIZE = 3.0 / TOTAL_POSITIONS
MAX_LONG_POSITION_SIZE = 3.0 / TOTAL_POSITIONS
 
 
class VIXFactor(CustomFactor):  
    window_length = 3  
    inputs = [cboe_vix.vix_close]
 
    def compute(self, today, assets, out, vix_close):   
        out[:] = np.nanmean(vix_close,axis=0)
 
 
def ls_slope (y):
    x = np.linspace(0, len(y)-1,len(y))
 
    X = x - x.mean()
    Y = y - y.mean()
 
    slope = (X.dot(Y)) / (X.dot(X))
    return(slope)
 
def mrm_c(std):
    return(  (0.95-np.tanh(std/1.3)**2) * (std) )
 
def smooth(x,window_len=252,window='hanning'):
    if x.size < window_len:
        return x
    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    elif window == 'hanning':
        w=np.hanning(window_len)
 
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y
 
def diag(k):
    a = np.zeros((k,k),int)
    (np.fill_diagonal(a,1))
    return(a)
 
def KCorth(Mx):
    K = Mx.shape[1]
    T = Mx.shape[0]
    C = np.cov(np.transpose(Mx))
    Mu = np.apply_along_axis(np.mean,0,Mx)
    Dx = Mx-np.tile(Mu,(T,1))
    M = np.matmul(np.transpose(Dx),Dx)
    O = np.linalg.eig(M)[1]
    Dl = diag(K)*np.transpose((np.sqrt(1/np.linalg.eig(M)[0])))
    S = np.matmul(np.matmul(O,Dl),np.transpose(O))
    sc = diag(K)*np.sqrt(abs(C))
    Ss = np.matmul((S*np.sqrt(T-1)),sc)
    out = np.matmul((Mx),Ss)
    return(out)
 
def nanzscore_gp(a,gp):
    pdf  = pd.DataFrame({'a':[a], 'gp':[gp]})    
    pdfz = pdf.groupby('gp').a.transform(nanzscore)
    return((pdfz))
 
def beta_fitconvert(a):
    z = a/np.nanmax(a)
    z = z-np.nanmin(z)
    return(z)
 
 
        
def initialize(context):
    """
    A core function called automatically once at the beginning of a backtest.
 
    Use this function for initializing state or other bookkeeping.
 
    Parameters
    ----------
    context : AlgorithmContext
        An object that can be used to store state that you want to maintain in 
        your algorithm. context is automatically passed to initialize, 
        before_trading_start, handle_data, and any functions run via schedule_function.
        context provides the portfolio attribute, which can be used to retrieve information 
        about current positions.
    """
    
    algo.attach_pipeline(make_pipeline(), 'long_short_equity_template')
 
    # Attach the pipeline for the risk model factors that we
    # want to neutralize in the optimization step. The 'risk_factors' string is 
    # used to retrieve the output of the pipeline in before_trading_start below.
    algo.attach_pipeline(risk_loading_pipeline(), 'risk_factors')
 
    # Schedule our rebalance function
    algo.schedule_function(func=rebalance,
                           date_rule=algo.date_rules.month_start(days_offset = 3),
                           time_rule=algo.time_rules.market_close(hours=0, minutes=30),
                           half_days=True)
 
    # Record our portfolio variables at the end of day
    algo.schedule_function(func=record_vars,
                           date_rule=algo.date_rules.every_day(),
                           time_rule=algo.time_rules.market_close(),
                           half_days=True)
    
def make_pipeline():
    """
    A function to create our dynamic stock selector (pipeline). Documentation
    on pipeline can be found here:
    https://www.quantopian.com/help#pipeline-title
    """
 
 
    # Factor of yesterday's close price.
 
    pipe = Pipeline(
        columns={
            's_all': algothon_2.s_all.latest,
            's_none': algothon_2.s_non.latest,
            's_risk': algothon_2.s_risk.latest,
            's_momentum': algothon_2.s_mom.latest,
            'nn_all': algothon_3.nn_all.latest,
            's_sc': algothon_2.s_sc.latest,
            'vol' : Volatility(),
            'Sector': Sector()
        },
    screen=algothon_2.s_all.latest.notnull()
    )
 
    return pipe
 
 
def before_trading_start(context, data):
 
    
 
    context.pipeline_data = algo.pipeline_output('long_short_equity_template').dropna()
    context.pipeline_data['custom_all'] = context.pipeline_data['nn_all']
    context.pipeline_data['custom_none'] = context.pipeline_data['s_none']/(context.pipeline_data['vol']*100) 
 
 
    context.risk_loadings = algo.pipeline_output('risk_factors').dropna()
 
 
def rebalance(context, data):
    """
    Execute orders according to our schedule_function() timing.
    """
    """
    A function scheduled to run once every Monday at 10AM ET in order to
    rebalance the longs and shorts lists.
 
    Parameters
    ----------
    context : AlgorithmContext
        See description above.
    data : BarData
        See description above.
    """
    # Retrieve pipeline output
    pipeline_data = context.pipeline_data
    
    
    
    risk_loadings = context.risk_loadings
 
    # Here we define our objective for the Optimize API. We have
    # selected MaximizeAlpha because we believe our combined factor
    # ranking to be proportional to expected returns. This routine
    # will optimize the expected return of our algorithm, going
    # long on the highest expected return and short on the lowest.
    objective = opt.MaximizeAlpha(pipeline_data['s_all'])
 
    # Define the list of constraints
    constraints = []
    # Constrain our maximum gross leverage
    constraints.append(opt.MaxGrossExposure(MAX_GROSS_LEVERAGE))
 
    # Require our algorithm to remain dollar neutral
    constraints.append(opt.DollarNeutral())
 
    # Add the RiskModelExposure constraint to make use of the
    # default risk model constraints
    neutralize_risk_factors = opt.experimental.RiskModelExposure(
        risk_model_loadings=risk_loadings,
        version=opt.Newest,  
        min_momentum=-MAX_EXPO, 
        max_momentum=MAX_EXPO, 
        min_size=-MAX_EXPO, 
        max_size=MAX_EXPO, 
        min_value=-MAX_EXPO, 
        max_value=MAX_EXPO, 
        min_short_term_reversal=-MAX_EXPO, 
        max_short_term_reversal=MAX_EXPO, 
        min_volatility=-MAX_EXPO, 
        max_volatility=MAX_EXPO)
    
    constraints.append(neutralize_risk_factors)
 
    
    #constraints.append(opt.FactorExposure(pd.DataFrame(context.pipeline_data.Beta),{'Beta':-0.25},{'Beta':0.25}))
    
    # With this constraint we enforce that no position can make up
    # greater than MAX_SHORT_POSITION_SIZE on the short side and
    # no greater than MAX_LONG_POSITION_SIZE on the long side. This
    # ensures that we do not overly concentrate our portfolio in
    # one security or a small subset of securities.
    constraints.append(
        opt.PositionConcentration.with_equal_bounds(
            min=-MAX_SHORT_POSITION_SIZE,
            max=MAX_LONG_POSITION_SIZE
        ))
 
    constraints.append(
        opt.NetGroupExposure.with_equal_bounds(
            labels=pipeline_data.Sector,
            min=-0.05,
            max=0.05,
        ))    
    
    # Put together all the pieces we defined above by passing
    # them into the algo.order_optimal_portfolio function. This handles
    # all of our ordering logic, assigning appropriate weights
    # to the securities in our universe to maximize our alpha with
    # respect to the given constraints.
    algo.order_optimal_portfolio(
        objective=objective,
        constraints=constraints
    )
    pass
 
 
def record_vars(context, data):
    """
    Plot variables at the end of each day.
    """
 
                
    pass
 
 
def handle_data(context, data):
    """
    Called every minute.
    """
    pass