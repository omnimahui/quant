# encoding: UTF-8

from price import DailyPrice, Valuation, Security
from USprice import *
from common import *
import algo.generic
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt
import seaborn as sns 
import statsmodels.tsa.api as smt
from pmdarima.arima import ndiffs, nsdiffs
from datetime import date
from statsmodels.tsa.holtwinters import (ExponentialSmoothing, 
                                         SimpleExpSmoothing, 
                                         Holt)
from statsmodels.tsa.arima_model import ARIMA
import scipy.stats as scs
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
import pmdarima as pm

#from fbprophet import Prophet

#Due to Prophet called deregister_matplotlib_converters, need to register again.
pd.plotting.register_matplotlib_converters()

N_LAGS = 50
SIGNIFICANCE_LEVEL = 0.05    

plt.set_cmap('cubehelix')
sns.set_palette('cubehelix')
COLORS = [plt.cm.cubehelix(x) for x in [0.1, 0.3, 0.5, 0.7]]       

class timeseries(object):
    def __init__(self):
        self.df = pd.DataFrame()
        
    def load(self, security, start_date, end_date):
        if isChinaMkt(security):
            self.df = DailyPrice().load(security=security)
        else:
            self.df = USDailyPrice().load(security=security)
            
        self.df = self.df.loc[
            (self.df.index > start_date)
            & (self.df.index <= end_date)
        ]
        return self.df     
    
    def decompose(self,index="GLD"):
        WINDOW = 10
        self.start_date = "2016-01-01"
        self.end_date = TODAY
    
        if isChinaMkt(index):
            self.df = DailyPrice().load(security=index)
        else:
            self.df = USDailyPrice().load(security=index)    
        df = self.df['close'].to_frame()
        df = df.loc[(df.index > self.start_date) & (df.index <= self.end_date)]   
    
        df = df.resample('W').last()
        df = df.dropna()
        df[str(WINDOW)+' rolling_mean'] = df.close.rolling(window=WINDOW).mean()
        df[str(WINDOW)+' rolling_std'] = df.close.rolling(window=WINDOW).std()
        df.plot(title=index+' Price')
    
        plt.tight_layout()
        #plt.savefig('images/ch3 _im1.png')
        plt.show()        
    
        decomposition_mul = seasonal_decompose(df.close, 
                                               model='multiplicative',
                                                   period = 5
                                                   )
        decomposition_add = seasonal_decompose(df.close, 
                                               model='additive',
                                                   period = 5
                                                   )        
    
        plt.rcParams.update({'figure.figsize': (10,10)})
        decomposition_mul.plot().suptitle('Multiplicative Decompose', fontsize=22)
        decomposition_add.plot().suptitle('Additive Decompose', fontsize=22)
        plt.tight_layout()
        # plt.savefig('images/ch3_im2.png')
        plt.show()     
    
    
    
        df.reset_index(drop=False, inplace=True)
        df.rename(columns={'index': 'ds', 'close': 'y'}, inplace=True)   
        #train_indices = df.ds.apply(lambda x: x.year).values < 2020
        #df_train = df.loc[train_indices].dropna()
        #df_test = df.loc[~train_indices].reset_index(drop=True)    
        df_train = df.loc[(df.ds < "2020-08-21")].dropna()
        df_test = df.loc[(df.ds >= "2020-08-21")].reset_index(drop=True)    
    
    
        model_prophet = Prophet(seasonality_mode='multiplicative')
        model_prophet.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model_prophet.fit(df_train)        
        df_future = model_prophet.make_future_dataframe(periods=365)
        df_pred = model_prophet.predict(df_future)
        model_prophet.plot(df_pred)
        plt.tight_layout()
        #plt.savefig('images/ch3_im3.png')
        plt.show()        
    
        model_prophet.plot_components(df_pred)
        plt.tight_layout()
        #plt.savefig('images/ch3_im4.png')
        plt.show()        
    
    
        selected_columns = ['ds', 'yhat_lower', 'yhat_upper', 'yhat']
    
        df_pred = df_pred.loc[:, selected_columns].reset_index(drop=True)
        #df_test = df_test.merge(df_pred, on=['ds'], how='left')
        df_test = df.merge(df_pred, on=['ds'], how='outer')
        df_test.ds = pd.to_datetime(df_test.ds)
        df_test.set_index('ds', inplace=True)        
        fig, ax = plt.subplots(1, 1)
    
        ax = sns.lineplot(data=df_test[['y', 'yhat_lower', 
                                        'yhat_upper', 'yhat']])
        ax.fill_between(df_test.index,
                        df_test.yhat_lower,
                        df_test.yhat_upper,
                        alpha=0.3)
        ax.set(title=index+' Price - actual vs. predicted',
               xlabel='Date',
               ylabel='Gold Price ($)')
    
        plt.tight_layout()
        #plt.savefig('images/ch3_im5.png')
        plt.show()      
        
        
class stationarity(timeseries):
    def __init__(self):
        self.df = pd.DataFrame()
        
    def test(self):
        df = self.load("GLD", "2016-01-01", TODAY)
        #X = df["close"].dropna().values
        df['rtn'] = df.close/df.close.shift(1)
        df['log_rtn'] = np.log(df.close/df.close.shift(1))
        df['close_diff'] = df.close.diff(1)
        #self.adf(df["close"].dropna())
        #self.adf(df["rtn"].dropna())
        #self.adf(df["log_rtn"].dropna())
        #self.kpss(df["close"].dropna())
        self.adf(df["close_diff"].dropna())
        self.kpss(df["close_diff"].dropna())
        self.acf(df["close_diff"].dropna(),"Price diff")
        self.pacf(df["close_diff"].dropna(), "Price diff")
        #self.acf(df["rtn"].dropna(),"Return")
        #self.acf(df["log_rtn"].dropna(), "Log return")
        #self.pacf(df["rtn"].dropna(),"Return")
        #self.pacf(df["log_rtn"].dropna(), "Log return")
        print(f"Suggested # of differences (ADF): {ndiffs(df.close.dropna(), test='adf')}")
        print(f"Suggested # of differences (KPSS): {ndiffs(df.close.dropna(), test='kpss')}")
        print(f"Suggested # of differences (PP): {ndiffs(df.close.dropna(), test='pp')}") 
        print(f"Suggested # of differences (OSCB): {nsdiffs(df.close.dropna(), m=7,test='ocsb')}")
        print(f"Suggested # of differences (CH): {nsdiffs(df.close.dropna(), m=7, test='ch')}")        
        
        

        
    def adf(self, x):
        indices = ['Test Statistic', 'p-value',
                   '# of Lags Used', '# of Observations Used']        
        #result = adfuller(x, maxlag=1, regression="c", autolag=None)  #autolag="AIC"
        result = adfuller(x, autolag="AIC")  #
        results = pd.Series(result[0:4], index=indices)
    
        for key, value in result[4].items():
            results[f'Critical Value ({key})'] = value        
        
        if result[0] < result[4]["10%"]:
            #null hypothesis is series is not stationary
            print("ADF Statistic: %f" % result[0])
            print("p-value: %f" % result[1])
            print("Critical Values:")
            for key, value in result[4].items():
                print("\t%s: %.3f" % (key, value))
            print ("halflife={}".format(algo.generic.algo().halflife(x)))
        else:
            print ("not stationary")
        return results
            
            
    def acf(self, x, ylabel="Series"):
        fig, ax = plt.subplots(3, 1, figsize=(12, 10))
        
        price_acf = smt.graphics.plot_acf(x, 
                                    lags=N_LAGS, 
                                    alpha=SIGNIFICANCE_LEVEL, ax = ax[0])        
        ax[0].set(title='Autocorrelation Plots',
                  ylabel=ylabel)     
        
        smt.graphics.plot_acf(x ** 2, lags=N_LAGS, 
                              alpha=SIGNIFICANCE_LEVEL, ax = ax[1])
        ax[1].set(title='Autocorrelation Plots',
                  ylabel='Squared '+ylabel)
        
        smt.graphics.plot_acf(np.abs(x), lags=N_LAGS, 
                              alpha=SIGNIFICANCE_LEVEL, ax = ax[2])
        ax[2].set(ylabel='Absolute '+ylabel,
                  xlabel='Lag')
        
        # plt.tight_layout()
        # plt.savefig('images/ch1_im14.png')
        plt.show()                

            
    def pacf(self, x, ylabel="Series"):
        fig, ax = plt.subplots(3, 1, figsize=(12, 10))
        
        smt.graphics.plot_pacf(x,lags=N_LAGS, 
                  alpha=SIGNIFICANCE_LEVEL,  ax=ax[0])    
        ax[0].set(title='Partial Autocorrelation Plots',
                  ylabel=ylabel)      
        
        smt.graphics.plot_acf(x ** 2, lags=N_LAGS, 
                              alpha=SIGNIFICANCE_LEVEL, ax = ax[1])
        ax[1].set(title='Partial Autocorrelation Plots',
                  ylabel='Squared '+ ylabel)
        
        smt.graphics.plot_acf(np.abs(x), lags=N_LAGS, 
                              alpha=SIGNIFICANCE_LEVEL, ax = ax[2])
        ax[2].set(ylabel='Absolute '+ ylabel,
                  xlabel='Lag')    
        plt.show()  
        
    def kpss(self, x, h0_type='c'):
        '''
        Function for performing the Kwiatkowski-Phillips-Schmidt-Shin test for stationarity
    
        Null Hypothesis: time series is stationary
        Alternate Hypothesis: time series is not stationary
    
        Parameters
        ----------
        x: pd.Series / np.array
            The time series to be checked for stationarity
        h0_type: str{'c', 'ct'}
            Indicates the null hypothesis of the KPSS test:
                * 'c': The data is stationary around a constant(default)
                * 'ct': The data is stationary around a trend
        
        Returns
        -------
        results: pd.DataFrame
            A DataFrame with the KPSS test's results
        '''
        
        indices = ['Test Statistic', 'p-value', '# of Lags']
    
        kpss_test = kpss(x, regression=h0_type)
        results = pd.Series(kpss_test[0:3], index=indices)
        
        for key, value in kpss_test[3].items():
            results[f'Critical Value ({key})'] = value
        print("KPSS test:")
        print(results)
        return results
        
        
class expsmooth(timeseries):
    def __init__(self):
 
        pass
    
    def test(self):
        df = self.load("GLD", "2016-01-01", TODAY)
        #X = df["close"].dropna().values
        df['rtn'] = df.close/df.close.shift(1)
        df['log_rtn'] = np.log(df.close/df.close.shift(1))
        df['close_diff'] = df.close.diff(1)
        #self.ses(df.close)
        self.holt(df.close)
        

    
    def ses(self, x):
        x_resample = x.resample('M') \
                 .last() 
                     
        train_indices = x_resample.index.year < 2020
        train = x_resample[train_indices]
        test = x_resample[~train_indices]
        test_length = len(test)        

        ses_1 = SimpleExpSmoothing(train).fit(smoothing_level=0.2)
        ses_forecast_1 = ses_1.forecast(test_length)
        
        ses_2 = SimpleExpSmoothing(train).fit(smoothing_level=0.5)
        ses_forecast_2 = ses_2.forecast(test_length)
        
        ses_3 = SimpleExpSmoothing(train).fit()
        alpha = ses_3.model.params['smoothing_level']
        ses_forecast_3 = ses_3.forecast(test_length)  
        
        x_resample.plot(color=COLORS[0], 
                  title='Simple Exponential Smoothing',
                  label='Actual',
                  legend=True)
        
        ses_forecast_1.plot(color=COLORS[1], legend=True, 
                            label=r'$\alpha=0.2$')
        ses_1.fittedvalues.plot(color=COLORS[1])
        
        ses_forecast_2.plot(color=COLORS[2], legend=True, 
                            label=r'$\alpha=0.5$')
        ses_2.fittedvalues.plot(color=COLORS[2])
        
        ses_forecast_3.plot(color=COLORS[3], legend=True, 
                            label=r'$\alpha={0:.4f}$'.format(alpha))
        ses_3.fittedvalues.plot(color=COLORS[3])
        
        plt.tight_layout()
        #plt.savefig('images/ch3_im15.png')
        plt.show()    
        
    def holt(self, x):
        x_resample = x.resample('M') \
                 .last() 
                     
        train_indices = x_resample.index.year < 2020
        train = x_resample[train_indices]
        test = x_resample[~train_indices]
        test_length = len(test)  
        
        hs_1 = Holt(train).fit()
        hs_forecast_1 = hs_1.forecast(test_length)
        
        # Holt's model with exponential trend
        hs_2 = Holt(train, exponential=True).fit()
        # equivalent to ExponentialSmoothing(train, trend='mul').fit()
        hs_forecast_2 = hs_2.forecast(test_length)
        
        # Holt's model with exponential trend and damping
        hs_3 = Holt(train, exponential=False, 
                    damped=True).fit(damping_slope=0.99)
        hs_forecast_3 = hs_3.forecast(test_length)    
        
        x_resample.plot(color=COLORS[0],
                  title="Holt's Smoothing models",
                  label='Actual',
                  legend=True)
        
        hs_1.fittedvalues.plot(color=COLORS[1])
        hs_forecast_1.plot(color=COLORS[1], legend=True, 
                           label='Linear trend')
        
        hs_2.fittedvalues.plot(color=COLORS[2])
        hs_forecast_2.plot(color=COLORS[2], legend=True, 
                           label='Exponential trend')
        
        hs_3.fittedvalues.plot(color=COLORS[3])
        hs_forecast_3.plot(color=COLORS[3], legend=True, 
                           label='Exponential trend (damped)')
        
        plt.tight_layout()
        #plt.savefig('images/ch3_im16.png')
        plt.show()        
        
        
        
        # Holt-Winter's model with exponential trend
        SEASONAL_PERIODS = 12
        hw_1 = ExponentialSmoothing(train, 
                                    trend='mul', 
                                    seasonal='add', 
                                    seasonal_periods=SEASONAL_PERIODS).fit()
        hw_forecast_1 = hw_1.forecast(test_length)
        
        # Holt-Winter's model with exponential trend and damping
        hw_2 = ExponentialSmoothing(train, 
                                    trend='mul', 
                                    seasonal='add', 
                                    seasonal_periods=SEASONAL_PERIODS, 
                                    damped=True).fit()
        hw_forecast_2 = hw_2.forecast(test_length)        
        x_resample.plot(color=COLORS[0],
                  title="Holt-Winter's Seasonal Smoothing",
                  label='Actual',
                  legend=True)
        
        hw_1.fittedvalues.plot(color=COLORS[1])
        hw_forecast_1.plot(color=COLORS[1], legend=True, 
                           label='Seasonal Smoothing')
        
        phi = hw_2.model.params['damping_slope']
        plot_label = f'Seasonal Smoothing (damped with $\phi={phi:.4f}$)'
        
        hw_2.fittedvalues.plot(color=COLORS[2])
        hw_forecast_2.plot(color=COLORS[2], legend=True, 
                           label=plot_label)
        
        plt.tight_layout()
        #plt.savefig('images/ch3_im17.png')
        plt.show()        
    
class arima(timeseries):
    def __init__(self):
        pass
    
    def test(self):
        df = self.load("GLD", "2016-01-01", TODAY)
        #X = df["close"].dropna().values
        df['rtn'] = df.close/df.close.shift(1)
        df['log_rtn'] = np.log(df.close/df.close.shift(1))
        df['close_diff'] = df.close.diff(1)
        self.arima(df.close)
        #self.auto_arima(df.close)

    def arima(self, x):
        x = x.resample('W').last() 
        x = x.pct_change().dropna() #x.diff(1).dropna()
        x_resample = x.loc[
            (x.index <= "2020-08-18" )
        ]    
        x_test = x.loc[
            (x.index > "2020-08-18" )
            & (x.index <= TODAY)
        ]        
        #x_resample = x_resample.resample('W') \
        #         .last() 
        #x_test = x_test.resample('W') \
        #         .last()         
                    
        fig, ax = plt.subplots(2, sharex=True)
        x_resample.plot(title = "Series", ax=ax[0])
        x_resample.plot(ax=ax[1], title='First Differences')
        
        plt.tight_layout()
        #plt.savefig('images/ch3_im18.png')
        #plt.show()        
        
        #Test autocorrelation
        n_lags=40
        alpha=0.05
        h0_type='c'
        
        adf_results = stationarity().adf(x_resample)
        kpss_results = stationarity().kpss(x_resample, h0_type=h0_type)
    
        print('ADF test statistic: {:.2f} (p-val: {:.2f})'.format(adf_results['Test Statistic'],
                                                                 adf_results['p-value']))
        print('KPSS test statistic: {:.2f} (p-val: {:.2f})'.format(kpss_results['Test Statistic'],
                                                                  kpss_results['p-value']))
    
        fig, ax = plt.subplots(2, figsize=(16, 8))
        smt.graphics.plot_acf(x_resample, ax=ax[0], lags=n_lags, alpha=alpha)
        smt.graphics.plot_pacf(x_resample, ax=ax[1], lags=n_lags, alpha=alpha)  
        plt.show()  
        
        arima = ARIMA(x_resample, order=(1, 0, 0)).fit(disp=0)
        print (arima.summary())
        self.arima_diagnostics(arima.resid, 40)
        #Check if resident is white noise. if not, there still must be information in the time series.
        self.ljungbox_test(arima.resid)
        plt.tight_layout()
        #plt.savefig('images/ch3_im21.png')
        plt.show()  
        predict = pd.DataFrame(arima.forecast(3)[0], index = x_test.index)
        #print (x_test)
        #zero = pd.DataFrame(np.zeros(len(x_test.index)), index=x_test.index)
        plt.figure(figsize=(10,4))
        plt.plot(x_test,label='realValue')
        plt.plot(predict,label='predictValue')
        #plt.plot(zero,label='zero')
        plt.legend(loc=0)
        plt.show()
        

    def ljungbox_test(self, x):
        #null hypothesis is white noise
        ljung_box_results = acorr_ljungbox(x)
        
        fig, ax = plt.subplots(1, figsize=[16, 5])
        sns.scatterplot(x=range(len(ljung_box_results[1])), 
                        y=ljung_box_results[1], 
                        ax=ax)
        ax.axhline(0.05, ls='--', c='r')
        ax.set(title="Ljung-Box test's results",
               xlabel='Lag',
               ylabel='p-value')
        #print(table.set_index('lag'))
        plt.tight_layout()
        #plt.savefig('images/ch3_im22.png')
        plt.show()        
        
    def arima_diagnostics(self,resids, n_lags=40):
        '''
        Function for diagnosing the fit of an ARIMA model by investigating the residuals.
        
        Parameters
        ----------
        resids : np.array
            An array containing the residuals of a fitted model
        n_lags : int
            Number of lags for autocorrelation plot
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            Created figure
        '''
         
        # create placeholder subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    
        r = resids
        resids = (r - np.nanmean(r)) / np.nanstd(r)
        resids_nonmissing = resids[~(np.isnan(resids))]
        
        # residuals over time
        sns.lineplot(x=np.arange(len(resids)), y=resids, ax=ax1)
        ax1.set_title('Standardized residuals')
    
        # distribution of residuals
        x_lim = (-1.96 * 2, 1.96 * 2)
        r_range = np.linspace(x_lim[0], x_lim[1])
        norm_pdf = scs.norm.pdf(r_range)
        
        sns.distplot(resids_nonmissing, hist=True, kde=True, 
                     norm_hist=True, ax=ax2)
        ax2.plot(r_range, norm_pdf, 'g', lw=2, label='N(0,1)')
        ax2.set_title('Distribution of standardized residuals')
        ax2.set_xlim(x_lim)
        ax2.legend()
            
        # Q-Q plot
        qq = sm.qqplot(resids_nonmissing, line='s', ax=ax3)
        ax3.set_title('Q-Q plot')
    
        # ACF plot
        smt.graphics.plot_acf(resids, ax=ax4, lags=n_lags, alpha=0.05)
        ax4.set_title('ACF plot')
        plt.show()
    
        return fig        

    def auto_arima_predict(self, arima_model, test):
        n_forecasts = len(test)        
        auto_arima_pred = arima_model.predict(n_periods=n_forecasts, 
                                             return_conf_int=True, 
                                             alpha=0.05)
        
        auto_arima_pred = [pd.DataFrame(auto_arima_pred[0], 
                                        columns=['prediction']),
                           pd.DataFrame(auto_arima_pred[1], 
                                        columns=['ci_lower', 'ci_upper'])]
        auto_arima_pred = pd.concat(auto_arima_pred, 
                                    axis=1).set_index(test_resample.index)    
        fig, ax = plt.subplots(1)
        
        ax = sns.lineplot(data=test, color=COLORS[0], label='Actual')
        ax.plot(auto_arima_pred.prediction, c=COLORS[2], 
                label='ARIMA(3,1,2)')
        ax.fill_between(auto_arima_pred.index,
                        auto_arima_pred.ci_lower,
                        auto_arima_pred.ci_upper,
                        alpha=0.2, 
                        facecolor=COLORS[2])
        
        ax.set(title="Google's stock price  - actual vs. predicted", 
               xlabel='Date', 
               ylabel='Price ($)')
        ax.legend(loc='upper left')
        
        plt.tight_layout()
        #plt.savefig('images/ch3_im25.png')
        plt.show()        


    
    def auto_arima(self, x):
        x_train = x.loc[
            (x.index <= "2020-07-01" )
        ]    
        x_test = x.loc[
            (x.index > "2020-07-01" )
            & (self.df.index <= TODAY)
        ]          
        x_resample = x.resample('W') \
                 .last().dropna()
        test_resample = x_test.resample('W') \
                 .last().dropna() 

        auto_arima = pm.auto_arima(x_resample,
                                   error_action='ignore',
                                   suppress_warnings=True,
                                   seasonal=False,
                                   stepwise=False,
                                   approximation=False
                                   )
        print (auto_arima.summary())
        self.arima_diagnostics(auto_arima.resid(), 40)
        self.auto_arima_predict(auto_arima, test_resample)
        
        