import quandl
import pandas as pd
import numpy as np
from numpy import *
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly as py
import cufflinks as cf
import datetime as dt
import seaborn as sns
from sklearn import linear_model
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from matplotlib.ticker import PercentFormatter
import scipy
import os	
import pywt
import random
from numpy import hstack, sqrt, asarray, mean, arange
import pdb
import scipy.signal


maturities = asarray([1, 3, 6, 12, 24, 36, 60, 84, 120, 240, 360])

beta_names = ['beta1', 'beta2', 'beta3']
lam_t = .0609
_load2 = lambda x: (1.-exp(-lam_t*x)) / (lam_t*x)
_load3 = lambda x: ((1.-exp(-lam_t*x)) / (lam_t*x)) - \
exp(-lam_t*x)

import numpy as np
import pandas as pd
import datetime as dt
import statsmodels.api as sm

def loadData():
    # Define the lambda parameters for the Nelson-Siegel model
    lam_t = 0.0609
    
    # Define maturities in months (example, you can adjust this as needed)
    # maturities = np.asarray([1, 3, 6, 12, 24, 36, 60, 84, 120, 240, 360])

    # Load the Treasury rates dataset
    file_path = '/Users/evelynli/Documents/Work/ASL/ETF Hedging Data/ETF/processed data/daily-treasury-rates.csv'
    ratedata = pd.read_csv(file_path, index_col=0)

    # Define the Nelson-Siegel factor loading functions
    _load2 = lambda x: (1. - np.exp(-lam_t * x)) / (lam_t * x)
    _load3 = lambda x: ((1. - np.exp(-lam_t * x)) / (lam_t * x)) - np.exp(-lam_t * x)

    # Convert the index to datetime
    ratedata.index = pd.to_datetime(ratedata.index, format='%Y-%m-%d')

    # Filter the dataset from January 2018 to August 2024
    start_idx = ratedata.index.get_loc(dt.datetime.strptime('2006-02-09', '%Y-%m-%d'))
    ratedata = ratedata.iloc[start_idx:]

    # Create the design matrix X
    X = np.zeros((len(maturities), 2))
    X[:, 0] = _load2(maturities)
    X[:, 1] = _load3(maturities)
    X = sm.add_constant(X)

    # Initialize arrays for beta coefficients and residuals
    beta_fits = np.zeros((len(ratedata), 3))
    residuals = np.zeros((len(ratedata), len(maturities)))

    # Fit the maturity curve for each observation
    for i in range(len(ratedata)):
        model = sm.OLS(ratedata.iloc[i], X)
        results = model.fit()
        beta_fits[i, :3] = results.params
        residuals[i, :] = results.resid

    # Convert beta_fits and residuals into DataFrames
    beta_fits = pd.DataFrame(beta_fits, columns=['beta1', 'beta2', 'beta3'], index=ratedata.index)
    residuals = pd.DataFrame(residuals, columns=[str(mat) for mat in maturities], index=ratedata.index)
    #ratedata = ratedata.asfreq('W-FRI')  
    #beta_fits = beta_fits.asfreq('W-FRI') 
    #residuals = residuals.asfreq('W-FRI')  

    beta_fits = beta_fits.replace([np.inf, -np.inf], np.nan).ffill().dropna()
    ratedata = ratedata.replace([np.inf, -np.inf], np.nan).ffill().dropna()
    residuals = residuals.replace([np.inf, -np.inf], np.nan).ffill().dropna()

    return beta_fits, residuals, ratedata



def table2(residuals):
    columns = ['Mean', 'Std', 'Min', 'Max', 'MAE', 'RMSE', 'ACF(1)', 'ACF(12)', 'ACF(30)']
    table2 = pd.DataFrame(index=[str(mat) for mat in maturities], columns=columns)
    
    for mat in maturities:
        mat_str = str(mat)
        residual_series = residuals.loc[:, mat_str]

        # Calculate statistics
        table2.loc[mat_str, 'Mean'] = residual_series.mean()
        table2.loc[mat_str, 'Std'] = residual_series.std()
        table2.loc[mat_str, 'Min'] = residual_series.min()
        table2.loc[mat_str, 'Max'] = residual_series.max()
        table2.loc[mat_str, 'MAE'] = residual_series.abs().mean()  # Mean Absolute Error
        table2.loc[mat_str, 'RMSE'] = np.sqrt((residual_series**2).mean())  # Root Mean Squared Error

        # Calculate Autocorrelation Function values
        acf_values = sm.tsa.stattools.acf(residual_series, nlags=31)
        table2.loc[mat_str, 'ACF(1)'] = acf_values[1]
        table2.loc[mat_str, 'ACF(12)'] = acf_values[12]
        table2.loc[mat_str, 'ACF(30)'] = acf_values[30]

    return table2


def table3(beta_fits):
    table3_columns = ['Mean', 'Std', 'Min', 'Max', 'ACF(1)', 'ACF(12)', 'ACF(30)', 'ADF']
    table3 = pd.DataFrame(np.zeros((len(beta_names), len(table3_columns))), index=beta_names, columns=table3_columns)

    # Fill the DataFrame with statistics for each beta
    for beta in beta_names:
        beta_series = beta_fits.loc[:, beta]

        table3.loc[beta, 'Mean'] = beta_series.mean()
        table3.loc[beta, 'Std'] = beta_series.std()
        table3.loc[beta, 'Min'] = beta_series.min()
        table3.loc[beta, 'Max'] = beta_series.max()

        # Calculate Autocorrelation Function values
        acf_values = sm.tsa.stattools.acf(beta_series, nlags=31)
        table3.loc[beta, 'ACF(1)'] = acf_values[1]
        table3.loc[beta, 'ACF(12)'] = acf_values[12]
        table3.loc[beta, 'ACF(30)'] = acf_values[30]

        # Perform the Augmented Dickey-Fuller test
        adf_statistic = sm.tsa.adfuller(beta_series)[0]
        table3.loc[beta, 'ADF'] = adf_statistic

    return table3


def table4(forecast, actual, naive=False):
    # Define the start and end indices for the date range
    idx_2021 = actual.index.get_loc(dt.datetime.strptime('2021-01-08', '%Y-%m-%d'))
    idx_2024 = actual.index.get_loc(dt.datetime.strptime('2024-08-09', '%Y-%m-%d'))
    
    # Adjust the forecast if not naive
    if not naive:
        forecast = np.roll(forecast.values, -1, axis=0) 
        forecast = pd.DataFrame(forecast, index=actual.index[idx_2021:idx_2024+1], columns=actual.columns)
    
    # Calculate the error
    err = actual.iloc[idx_2021:idx_2024+1, :] - forecast.iloc[:, :]
    
    # Initialize the table DataFrame
    table = pd.DataFrame(np.zeros((11, 5)), 
                         index=['1', '3', '6', '12', '24', '36', '60', '84', '120', '240', '360'], 
                         columns=['mean', 'std. dev.', 'RMSE', 'ACF(1)', 'ACF(12)'])
    
    # Fill the table with statistics for each maturity
    for idx in table.index:
        table.loc[idx, 'mean'] = err.loc[:, idx].mean()
        table.loc[idx, 'std. dev.'] = err.loc[:, idx].std()
        table.loc[idx, 'RMSE'] = np.sqrt(np.mean(err.loc[:, idx]**2))
        
        acf_values = sm.tsa.acf(err.loc[:, idx], nlags=13)
        table.loc[idx, 'ACF(1)'] = acf_values[1]
        table.loc[idx, 'ACF(12)'] = acf_values[12] if len(acf_values) > 12 else np.nan

    return table


def yieldContors(ratedata):
    # Create the contour plot
    data = [
        go.Contour(
            z=ratedata.values,
            x=ratedata.columns, 
            y=ratedata.index,
            colorscale='Viridis',
            contours=dict(
                coloring='heatmap'
            ),
            colorbar=dict(
                title='Yield',
                titleside='right'
            )
        )
    ]

    # Layout for the plot
    layout = go.Layout(
        title='Yields vs. Maturities',
        width=640,
        height=480,
        xaxis=dict(
            title='Maturity (months)',
            titlefont=dict(size=16)
        ),  
        yaxis=dict(
            title='Date',
            titlefont=dict(size=16)
        )
    )

    # Create the figure with the data and layout
    fig = go.Figure(data=data, layout=layout)

    return fig


def exampleYield(ratedata, loc):
    # Ensure loc is a list
    if not isinstance(loc, list):
        raise ValueError('You must input a list')

    # Determine the title based on the length of loc
    if len(loc) == 1:
        tit = f'Fig 3: {ratedata.index[loc[0]]}'
    else:
        tit = "Sample Yield Curves"

    # Define the layout for the plot
    layout = go.Layout(
        width=640,
        height=480,
        title=tit,
        titlefont=dict(size=24),
        xaxis=dict(
            title='Maturity (months)',
            titlefont=dict(size=20)
        ),  
        yaxis=dict(
            title='Yield (percent)',
            titlefont=dict(size=20)
        ),
        legend=dict(font=dict(size=12))
    )

    # Initialize an empty list for traces
    data = []

    # Create traces for each specified location
    for idx in loc:
        if idx < len(ratedata):
            trace = go.Scatter(
                y=ratedata.iloc[idx, :], 
                x=ratedata.columns,
                name=str(ratedata.index[idx])
            )
            data.append(trace)

    # Return the figure
    return go.Figure(data=data, layout=layout)


def beta_resid(residuals):
    # Select specific columns based on maturity periods
    resid_interest = residuals.loc[:, ['1', '3', '6', '12', '24', '36', '60', '84', '120', '240', '360']]
    
    # Define layout for the plot
    layout = go.Layout(
        title='Residuals for Selected Maturity Periods (Months)',
        titlefont=dict(size=18),
        legend=dict(font=dict(size=16)),
        width=800,
        height=1000,
    )
    
    # Create traces for each maturity
    trace1 = go.Scatter(y=resid_interest['1'], x=resid_interest.index, name='1 MO')
    trace2 = go.Scatter(y=resid_interest['3'], x=resid_interest.index, name='3 MO')
    trace3 = go.Scatter(y=resid_interest['6'], x=resid_interest.index, name='6 MO')
    trace4 = go.Scatter(y=resid_interest['12'], x=resid_interest.index, name='12 MO')
    trace5 = go.Scatter(y=resid_interest['24'], x=resid_interest.index, name='24 MO')
    trace6 = go.Scatter(y=resid_interest['36'], x=resid_interest.index, name='36 MO')
    trace7 = go.Scatter(y=resid_interest['60'], x=resid_interest.index, name='60 MO')
    trace8 = go.Scatter(y=resid_interest['84'], x=resid_interest.index, name='84 MO')
    trace9 = go.Scatter(y=resid_interest['120'], x=resid_interest.index, name='120 MO')
    trace10 = go.Scatter(y=resid_interest['240'], x=resid_interest.index, name='240 MO')
    trace11 = go.Scatter(y=resid_interest['360'], x=resid_interest.index, name='360 MO')
    
    # Create a subplot layout with 5 rows and 2 columns
    fig = make_subplots(rows=5, cols=2, print_grid=False)

    # Add each trace to the appropriate subplot
    fig.add_trace(trace1, row=1, col=1)
    fig.add_trace(trace2, row=1, col=2)
    fig.add_trace(trace3, row=2, col=1)
    fig.add_trace(trace4, row=2, col=2)
    fig.add_trace(trace5, row=3, col=1)
    fig.add_trace(trace6, row=3, col=2)
    fig.add_trace(trace7, row=4, col=1)
    fig.add_trace(trace8, row=4, col=2)
    fig.add_trace(trace9, row=5, col=1)
    fig.add_trace(trace11, row=5, col=2)  # trace11 for 360 MO

    # Update the figure's layout
    fig.update_layout(layout)

    return fig

def beta_dist(beta_fits):
	fig, axes = plt.subplots(1,3, figsize=(10,7))
	fig.suptitle('Fitted Parameters Histogram')
	sns.set_theme(font_scale=1)
	sns.histplot(beta_fits.loc[:, 'beta1'], kde=True, ax=axes[0])
	axes[0].set_title('Beta 1')
    
	sns.histplot(beta_fits.loc[:, 'beta2'], kde=True, ax=axes[1])
	axes[1].set_title('Beta 2')
    
	sns.histplot(beta_fits.loc[:, 'beta3'], kde=True, ax=axes[2])
	axes[2].set_title('Beta 3')
    
    # Adjust layout for better spacing
	plt.tight_layout(rect=[0, 0, 1, 0.95])
    
	return fig



def fig7(ratedata, beta_fits):
    # Compute the empirical estimates
    beta1_hat = ratedata.loc[:, '120']
    beta2_hat = ratedata.loc[:, '120'] - ratedata.loc[:, '3']
    beta3_hat = 2 * ratedata.loc[:, '24'] - (ratedata.loc[:, '120'] + ratedata.loc[:, '3'])

    # Define the layout for the plot
    layout = go.Layout(
        title='Residuals for selected maturities (months)',
        titlefont=dict(size=18),
        legend=dict(font=dict(size=16)),
        width=640,
        height=480,
    )

    # Create subplots
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        subplot_titles=('Level', 'Slope', 'Curvature'),
        print_grid=False
    )

    # Create traces for beta1 (Level)
    beta1a = go.Scatter(
        x=beta1_hat.index, 
        y=beta1_hat, 
        name='Empirical Level'
    )

    beta1b = go.Scatter(
        x=beta_fits.index,
        y=beta_fits.loc[:, 'beta1'], 
        name='Fitted Level'
    )

    # Create traces for beta2 (Slope)
    beta2a = go.Scatter(
        x=beta2_hat.index, 
        y=beta2_hat, 
        name='Empirical Slope'
    )

    beta2b = go.Scatter(
        x=beta_fits.index,
        y=-beta_fits.loc[:, 'beta2'], 
        name='Fitted Slope'
    )

    # Create traces for beta3 (Curvature)
    beta3a = go.Scatter(
        x=beta3_hat.index, 
        y=beta3_hat, 
        name='Empirical Curvature'
    )

    beta3b = go.Scatter(
        x=beta_fits.index,
        y=0.3 * beta_fits.loc[:, 'beta3'], 
        name='Fitted Curvature'
    )

    # Add traces to the subplots
    fig.add_trace(beta1a, row=1, col=1)
    fig.add_trace(beta1b, row=1, col=1)

    fig.add_trace(beta2a, row=2, col=1)
    fig.add_trace(beta2b, row=2, col=1)

    fig.add_trace(beta3a, row=3, col=1)
    fig.add_trace(beta3b, row=3, col=1)

    # Update layout
    fig.update_layout(layout)

    return fig



def ACF_beta(beta_fits):
    titlefont = {'fontsize': 14}
    fig = plt.figure(figsize=(20, 12))

    ax1 = fig.add_subplot(321)
    sm.graphics.tsa.plot_acf(beta_fits.iloc[:, 0].values.squeeze(), lags=60, ax=ax1)
    ax1.set_title('ACF for Level', **titlefont)

    ax2 = fig.add_subplot(322)
    sm.graphics.tsa.plot_pacf(beta_fits.iloc[:, 0].values.squeeze(), lags=60, ax=ax2)
    ax2.set_title('PACF for Level', **titlefont)

    ax3 = fig.add_subplot(323)
    sm.graphics.tsa.plot_acf(beta_fits.iloc[:, 1].values.squeeze(), lags=60, ax=ax3)
    ax3.set_title('ACF for Slope', **titlefont)

    ax4 = fig.add_subplot(324)
    sm.graphics.tsa.plot_pacf(beta_fits.iloc[:, 1].values.squeeze(), lags=60, ax=ax4)
    ax4.set_title('PACF for Slope', **titlefont)

    ax5 = fig.add_subplot(325)
    sm.graphics.tsa.plot_acf(beta_fits.iloc[:, 2].values.squeeze(), lags=60, ax=ax5)
    ax5.set_title('ACF for Curvature', **titlefont)

    ax6 = fig.add_subplot(326)
    sm.graphics.tsa.plot_pacf(beta_fits.iloc[:, 2].values.squeeze(), lags=60, ax=ax6)
    ax6.set_title('PACF for Curvature', **titlefont)

    return fig



def ARforecast(ratedata, beta_fits):
    # Convert indexes to DatetimeIndex with weekly frequency
    ratedata.index = pd.to_datetime(ratedata.index)
    beta_fits.index = pd.to_datetime(beta_fits.index)
    
    #ratedata = ratedata.asfreq('W-FRI')  # Assuming data is from the end of each week
    #beta_fits = beta_fits.asfreq('W-FRI')  # Assuming data is from the end of each week

    beta_fits = beta_fits.replace([np.inf, -np.inf], np.nan).ffill().dropna()

    # Clip the data to the relevant date range
    idx_2021 = ratedata.index.get_loc(pd.to_datetime('2021-01-08'))
    idx_2024 = ratedata.index.get_loc(pd.to_datetime('2024-08-09'))

    # Calculate the number of out-of-sample points
    N_out = idx_2024 - idx_2021

    # Initialize DataFrames to store predictions
    beta_predict = pd.DataFrame(np.zeros((N_out, 3)), index=beta_fits.index[idx_2021:idx_2024], columns=beta_fits.columns)
    yield_forecast = pd.DataFrame(np.zeros((N_out, len(ratedata.columns))), index=beta_fits.index[idx_2021:idx_2024], columns=ratedata.columns)
    
    beta_predict_nieve = pd.DataFrame(np.zeros((N_out, 3)), index=beta_fits.index[idx_2021:idx_2024], columns=beta_fits.columns)
    yield_forecast_nieve = pd.DataFrame(np.zeros((N_out, len(ratedata.columns))), index=beta_fits.index[idx_2021:idx_2024], columns=ratedata.columns)
    
    beta_predict_random = pd.DataFrame(np.zeros((N_out, 3)), index=beta_fits.index[idx_2021:idx_2024], columns=beta_fits.columns)
    yield_forecast_random = pd.DataFrame(np.zeros((N_out, len(ratedata.columns))), index=beta_fits.index[idx_2021:idx_2024], columns=ratedata.columns)

    # Function to show progress
    def perDone(i, length, goal):
        if i != 0 and (float(i) / length) * 100 > goal:
            print(f"{goal}% done")
            return 10.0
        else:
            return 0

    # Placeholder to store wavelet coefficients
    saveRuns = []
    d = 10.0
    wavelet = 'db2'
    i =  0

    dates = beta_fits.index[idx_2021:idx_2024+1].to_list()
    dates_whole = beta_fits.index.to_list()

    for date in dates:
        idx = ratedata.index.get_loc(pd.to_datetime(date)) - idx_2021
        d += perDone(idx, N_out, d)
        now = date
        start = dates_whole[i]

        for beta in beta_fits.columns:
            testb = beta_fits.loc[start:now, beta]
            signal_length = len(testb)
            if signal_length % 2 != 0: 
                testb = np.append(testb, 0)
            coeff = pywt.swt(testb, wavelet)
            model = sm.tsa.AutoReg(beta_fits.loc[:now, beta], lags=1).fit()
            
            try:
                beta_predict_nieve.loc[date, beta] = model.predict(len(beta_fits.loc[:now, beta])-1, len(beta_fits.loc[:now, beta])).iloc[-1]
                idx_lag = ratedata.index.get_loc(pd.to_datetime(date))-1
                date_lag = ratedata.index[idx_lag]
                beta_predict_random.loc[date, beta] = (beta_fits.loc[date_lag, beta] + beta_fits.loc[:now, beta].std() * np.random.randn())
    
            except KeyError:
                    pdb.set_trace()

            # Handle wavelet-based predictions
            for level in coeff:
                for detail in range(len(level)):
                    model = sm.tsa.AutoReg(level[detail], lags=1).fit()
                    prediction = model.predict(start=len(level[detail]) - 1, end=len(level[detail]))[-1]
                    level[detail][:] = np.hstack((level[detail][1:], prediction))

        # Forecast the yields at specific maturities
            try:
                pred = []
                N_lvl = np.shape(coeff)[0]  # Number of levels in the wavelet decomposition
                # Loop through each level
                for t in range(0, N_lvl):
                    try:
                        # Calculate the prediction with stochastic shock
                        pred.append((coeff[t][0][-1] + coeff[t][1][-5]) / np.sqrt(2) ** (N_lvl - t))

                    except IndexError:
                        pdb.set_trace()

                # Store the mean of predictions
                beta_predict.loc[date, beta] = np.mean(pred)

            except KeyError:
                pdb.set_trace()

        try:
            # Perform the yield forecast using the beta predictions
            yield_forecast.loc[date, :] = (beta_predict.loc[date, 'beta1'] + beta_predict.loc[date, 'beta2'] * _load2(np.asarray(maturities)) + beta_predict.loc[date, 'beta3'] * _load3(np.asarray(maturities)))

            yield_forecast_nieve.loc[date, :] = (beta_predict_nieve.loc[date, 'beta1'] + beta_predict_nieve.loc[date, 'beta2'] * _load2(np.asarray(maturities)) + beta_predict_nieve.loc[date, 'beta3'] * _load3(np.asarray(maturities)))

            yield_forecast_random.loc[date, :] = (beta_predict_random.loc[date, 'beta1'] + beta_predict_random.loc[date, 'beta2'] * _load2(np.asarray(maturities)) + beta_predict_random.loc[date, 'beta3'] * _load3(np.asarray(maturities)))
            # Save the wavelet coefficients
            saveRuns.append(coeff)

        except TypeError:
            pdb.set_trace()
                
        i += 1

    return yield_forecast_nieve, yield_forecast, yield_forecast_random, saveRuns



def cwtGraph(beta_fits):
    fig = plt.figure(figsize=(15, 7))

    # Plot for Beta1
    ax1 = fig.add_subplot(131)
    cwtmatr1 = scipy.signal.cwt(beta_fits.loc[:, 'beta1'], scipy.signal.ricker, np.arange(1, 100))
    ax1.imshow(cwtmatr1, extent=[0, cwtmatr1.shape[1], 1, cwtmatr1.shape[0]], cmap='PRGn',
               aspect='auto', vmax=abs(cwtmatr1).max(), vmin=-abs(cwtmatr1).max())
    ax1.set_xlabel('Time (months)')
    ax1.set_title('Beta1')
    ax1.set_ylabel('Scale')

    # Plot for Beta2
    ax2 = fig.add_subplot(132)
    cwtmatr2 = scipy.signal.cwt(beta_fits.loc[:, 'beta2'], scipy.signal.ricker, np.arange(1, 100))
    ax2.imshow(cwtmatr2, extent=[0, cwtmatr2.shape[1], 1, cwtmatr2.shape[0]], cmap='PRGn',
               aspect='auto', vmax=abs(cwtmatr2).max(), vmin=-abs(cwtmatr2).max())
    ax2.set_xlabel('Time (months)')
    ax2.set_ylabel('Scale')
    ax2.set_title('Beta2')

    # Plot for Beta3
    ax3 = fig.add_subplot(133)
    cwtmatr3 = scipy.signal.cwt(beta_fits.loc[:, 'beta3'], scipy.signal.ricker, np.arange(1, 100))
    ax3.imshow(cwtmatr3, extent=[0, cwtmatr3.shape[1], 1, cwtmatr3.shape[0]], cmap='PRGn',
               aspect='auto', vmax=abs(cwtmatr3).max(), vmin=-abs(cwtmatr3).max())
    ax3.set_xlabel('Time (months)')
    ax3.set_ylabel('Scale')
    ax3.set_title('Beta3')

    plt.show()


def meanError(actual, naive, wavelet):
    # Find the indices for the given dates
    idx_2021 = actual.index.get_loc(dt.datetime.strptime('2021-01-08', '%Y-%m-%d'))
    idx_2024 = actual.index.get_loc(dt.datetime.strptime('2024-08-09', '%Y-%m-%d'))
    
    # Roll the wavelet data to align correctly, and calculate the error
    err = np.roll(wavelet.values, -1, axis=0)
    
    # Calculate mean absolute errors
    wavelet_error = np.abs((err - actual.iloc[idx_2021:idx_2024+1].values)).mean(axis=0)
    naive_error = np.abs((naive - actual.iloc[idx_2021:idx_2024+1].values)).mean(axis=0)
    
    # Create trace for wavelet-extended
    trace1 = go.Scatter(
        x=maturities, 
        y=wavelet_error,
        name='Wavelet-extended'
    )
    
    # Create trace for naive
    trace2 = go.Scatter(
        x=maturities, 
        y=naive_error,
        name='Naive'
    )

    # Define the layout for the plot
    layout = go.Layout(
        title='Mean Absolute Error',
        titlefont=dict(size=16),
        legend=dict(font=dict(size=18)),
        width=600,
        height=480,
        xaxis=dict(title='Maturity (months)'),
        yaxis=dict(title='Mean Error (yield percentage)')
    )
    
    # Compile the data and return the figure
    data = [trace1, trace2]
    return go.Figure(data=data, layout=layout)