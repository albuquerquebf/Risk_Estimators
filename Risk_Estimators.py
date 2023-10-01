import mgarch
import pandas as pd
import numpy as np
import pyRMT 
from pyRMT import optimalShrinkage
import portfoliolab as pl

re = pl.estimators.RiskEstimators()

def get_returns(prices , log = False):
    if not isinstance(prices, pd.DataFrame):
        raise Exception("Prices is not a Pandas DataFrame!")
    
    if log == True:
        
        returns =  np.log(prices/prices.shift(1)).dropna()
        
    else:
        returns = prices.pct_change().dropna()
    
    return returns

def ewma_cov(prices, alpha = .98, method = None, corr = False):
    ret = get_returns(prices , log = True)
    hlf = np.log(.5)/np.log(alpha)
    cov_ewma = ret.ewm(halflife=hlf).cov()[-len(prices.columns):].reset_index(level = 'Dates').drop(['Dates'] , axis =1)
    if ewma_cov == False:
        return cov_ewma
    else:
        ewma_matrix = re.cov_to_corr(cov_ewma)
        return ewma_matrix
    
    
def garch_estimation(prices,per, corr = False):
    # Use qrt estimation
    ret = get_returns(prices , log = True)
    vol = mgarch.mgarch()
    vol.fit(ret)
    vol_predictor = vol.predict(per)['cov']
    if corr == True:
        return vol_predictor
    else:
        return re.cov_to_corr(vol_predictor)


def shrinkage(prices, alpha = .1, corr = False):
    ret = get_returns(prices , log = True)
    
    cov_matrix = ret.cov()
    
    matrix_reduction = (1-alpha)*cov_matrix
    
    ev_shifter = (np.trace(cov_matrix)/cov_matrix.shape[1])*alpha*np.identity(cov_matrix.shape[1])
    
    shrinker = matrix_reduction + ev_shifter
    
    if corr == False:
        return shrinker
    
    else:
        corr_shrink = re.cov_to_corr(shrinker)
        return corr_shrink

def shrinked_ewma(prices, alpha = None, gamma = .98, corr = False):
    ret = get_returns(prices , log = True)

    hlf = np.log(.5)/np.log(gamma)

    cov_ewma = ret.ewm(halflife = hlf).cov()[-len(prices.columns):].reset_index(level = 'Dates').drop(['Dates'] , axis =1)

    cov_matrix = cov_ewma

    matrix_reduction = (1-alpha)*cov_matrix
    
    ev_shifter = (np.trace(cov_matrix)/cov_matrix.shape[1])*alpha*np.identity(cov_matrix.shape[1])
    
    ewma_shrinker = matrix_reduction + ev_shifter
    
    if corr == False:
        return ewma_shrinker
    else:
        return re.cov_to_corr(ewma_shrinker)
    
    
def corrEigenClip(prices, return_covariance = True):
    ret = get_returns(prices , log = True)
    return pyRMT.clipped(ret) 
   
    
def corrRIE(prices, return_covariance = True):
    ret =  ret = get_returns(prices , log = True)
    return optimalShrinkage(ret)


### Intradays Unused ###

def parkinson(prices, id_matrix = False):

    open_prices = prices.filter(regex = "Open").dropna()
    low_prices = prices.filter(regex = "Low").dropna()
    high_prices = prices.filter(regex = "High").dropna()
    close_prices = prices.filter(regex = "Close").dropna()

    
    high_low=((high_prices.values / low_prices).apply(pd.to_numeric)).apply(np.log) 
    
    pk = (1/(4*np.log(2))) * (high_low)**2
    
    pk.columns = "pk_" + pk.columns.str.strip("Low")
    vol_estimator = (np.sum(pk)/pk.shape[0])**.5

    df_returns = np.log((close_prices/close_prices.shift(1))).dropna()
    
    vector = np.diag(vol_estimator)
    
    if id_matrix == True:
        identity = np.identity(vol_estimator.shape[0])
        return np.dot(vector, np.dot(identity, vector))
    
    else:
        return np.dot(vector, np.dot(np.array(df_returns.corr()), vector))


def garman_klass(prices, id_matrix = False):
        
    open_prices = prices.filter(regex = "Open").dropna()
    low_prices = prices.filter(regex = "Low").dropna()
    high_prices = prices.filter(regex = "High").dropna()
    close_prices = prices.filter(regex = "Close").dropna()
    
    high_low = ((high_prices.values / low_prices).apply(pd.to_numeric)).apply(np.log)
    close_open = ((close_prices.values / open_prices).apply(pd.to_numeric)).apply(np.log)

    gk = (0.5*(high_low**2)).values - ((2*np.log(2) - 1)*(close_open**2))
    gk.columns = "gk_" + gk.columns.str.strip("Open")
    vol_estimator = (np.sum(gk)/gk.shape[0])**.5

    df_returns = np.log((close_prices/close_prices.shift(1))).dropna()
    
    vector = np.diag(vol_estimator)
    
    if id_matrix == True:
        identity = np.identity(vol_estimator.shape[0])
        return np.dot(vector, np.dot(identity, vector))
    
    else:
        return np.dot(vector, np.dot(np.array(df_returns.corr()), vector))


def rogers_satchell(prices, id_matrix = False):
    
    open_prices = prices.filter(regex = "Open").dropna()
    low_prices = prices.filter(regex = "Low").dropna()
    high_prices = prices.filter(regex = "High").dropna()
    close_prices = prices.filter(regex = "Close").dropna()
    
    low_open = ((low_prices.values / open_prices).apply(pd.to_numeric)).apply(np.log)
    high_close = ((high_prices.values / close_prices).apply(pd.to_numeric)).apply(np.log)
    high_open = ((high_prices.values / open_prices).apply(pd.to_numeric)).apply(np.log)
    low_close = ((low_prices.values / close_prices).apply(pd.to_numeric)).apply(np.log)
    
    rs = ((high_close.values*high_open) + (low_close.values*low_open))
    rs.columns = "rs_" + rs.columns.str.strip("Open")
    
    vol_estimator = (np.sum(rs)/rs.shape[0])**.5

    df_returns = np.log((close_prices/close_prices.shift(1))).dropna()
    
    vector = np.diag(vol_estimator)
    
    if id_matrix == True:
        identity = np.identity(vol_estimator.shape[0])
        return np.dot(vector, np.dot(identity, vector))
    
    else:
        return np.dot(vector, np.dot(np.array(df_returns.corr()), vector))



def gkyz(prices, id_matrix = False):
    """
    Garman-Klass with Yang-Zhang overnight 
    """
    
    open_prices = prices.filter(regex = "Open").dropna()
    low_prices = prices.filter(regex = "Low").dropna()
    high_prices = prices.filter(regex = "High").dropna()
    close_prices = prices.filter(regex = "Close").dropna()

    
    overnight_jump = ((open_prices.values / close_prices.shift(1)).apply(pd.to_numeric)).apply(np.log).dropna()
    high_low = ((high_prices.iloc[1:, :].values / low_prices.iloc[1:, :]).apply(pd.to_numeric)).apply(np.log)
    close_open = ((close_prices.iloc[1:, :].values / open_prices.iloc[1:, :]).apply(pd.to_numeric)).apply(np.log)
    
    garman_zhang = (0.5 * (overnight_jump**2)) + (0.5*(high_low**2)).values - ((2*np.log(2) - 1)*(close_open**2)).values
    garman_zhang.columns = "gkyz_" + garman_zhang.columns.str.strip("Close")
    vol_estimator = (np.sum(garman_zhang)/garman_zhang.shape[0])**.5

    df_returns = np.log((close_prices/close_prices.shift(1))).dropna()
    
    vector = np.diag(vol_estimator)
    
    if id_matrix == True:
        identity = np.identity(vol_estimator.shape[0])
        return np.dot(vector, np.dot(identity, vector))
    
    else:
        return np.dot(vector, np.dot(np.array(df_returns.corr()), vector))


def yang_zhang(prices, id_matrix = False, alpha = 0.34):
    
    """
    Yang-Zhang (https://portfolioslab.com/tools/yang-zhang)
    
    """
    
    open_prices = prices.filter(regex = "Open").dropna()
    low_prices = prices.filter(regex = "Low").dropna()
    high_prices = prices.filter(regex = "High").dropna()
    close_prices = prices.filter(regex = "Close").dropna()

    overnight_jump = ((open_prices.values / close_prices.shift(1)).apply(pd.to_numeric)).apply(np.log).dropna()
    high_low = ((high_prices.iloc[1:, :].values / low_prices.iloc[1:, :]).apply(pd.to_numeric)).apply(np.log)
    close_open = ((close_prices.iloc[1:, :].values / open_prices.iloc[1:, :]).apply(pd.to_numeric)).apply(np.log)
    low_open = ((low_prices.iloc[1:,:].values / open_prices.iloc[1:,:]).apply(pd.to_numeric)).apply(np.log)
    high_close = ((high_prices.iloc[1:,:].values / close_prices.iloc[1:,:]).apply(pd.to_numeric)).apply(np.log)
    high_open = ((high_prices.iloc[1:,:].values / open_prices.iloc[1:,:]).apply(pd.to_numeric)).apply(np.log)
    low_close = ((low_prices.iloc[1:,:].values / close_prices.iloc[1:,:]).apply(pd.to_numeric)).apply(np.log)
    
    k = (alpha - 1)/(alpha + ((prices.shape[0] + 1)/(prices.shape[0] - 1)))
    
    overnight_jump_norm = (np.sum(overnight_jump - overnight_jump.mean()))/(prices.shape[0] - 1)
    log_co_norm = (np.sum(close_open - close_open.mean()))/(prices.shape[0] - 1)
    
    rs = ((high_close.values*high_open) + (low_close.values*low_open))
    rogers_satchell = (np.sum(rs)/(prices.shape[0] - 1))

    yz = (overnight_jump_norm.values + k*log_co_norm + 
          (1 - k)*rogers_satchell.values).rename({"Open IBOV": "yz_IBOV", "Open SPX": "yz_SPX", "Open BLX": "yz_BLX"})
    
    vol_estimator = yz**.5
    
    df_returns = np.log((close_prices/close_prices.shift(1))).dropna()
    
    vector = np.diag(vol_estimator)
    
    if id_matrix == True:
        identity = np.identity(vol_estimator.shape[0])
        return np.dot(vector, np.dot(identity, vector))
    
    else:
        return np.dot(vector, np.dot(np.array(df_returns.corr()), vector))

    

       

    

    
    
        
