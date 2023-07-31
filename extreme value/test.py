# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
import seaborn as sns
import statsmodels.api as sm

# %%
import warnings
warnings.filterwarnings('ignore')

# %%
filepath = "D:\杨钦\计算机语言与笔记\quant\牧鑫\extreme value\pricing_data_OriginalIndex_2013-07_2023-07.csv"
data = pd.read_csv(filepath)


# %%
def append_CCI(df: pd.DataFrame, P = 20):
    TP = (df['high'] + df['low'] + df['close']) / 3
    MA_TP = TP.rolling(window=P).mean()
    MD = TP.rolling(window=P).apply(lambda x: pd.Series(x - x.mean()).abs().mean())
    df['CCI'] = (TP - MA_TP) / (0.015 * MD)
    
    return df


def append_RSI(df: pd.DataFrame, rsi_period = 14):
    # Calculate the daily price change
    price_change = df['close'].diff()
    # Separate the gains and losses into their own columns
    gain = price_change.where(price_change > 0, 0)
    loss = -price_change.where(price_change < 0, 0)
    # Calculate the average gain and loss over the RSI period
    # Note: The first value is calculated as a simple average
    # Subsequent values are calculated with smoothing
    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()
    # Calculate the relative strength (RS)
    rs = avg_gain / avg_loss
    # Calculate the RSI
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df
            
def append_Bollinger(df: pd.DataFrame, period = 20):
    # Calculate the simple moving average (SMA)
    df['SMA'] = df['close'].rolling(window=period).mean()
    # Calculate the standard deviation
    df['STD'] = df['close'].rolling(window=period).std()
    # Calculate the upper Bollinger Band
    df['Upper_BB'] = df['SMA'] + (df['STD'] * 2)
    # Calculate the lower Bollinger Band
    df['Lower_BB'] = df['SMA'] - (df['STD'] * 2)
    
    return df

def append_MACD(df: pd.DataFrame, short_span=12, long_span=26, signal_span=9):
    # Calculate the short and long term EMAs
    df['EMA_short'] = df['close'].ewm(span=short_span, adjust=False).mean()
    df['EMA_long'] = df['close'].ewm(span=long_span, adjust=False).mean()
    # Calculate the MACD line
    df['MACD_line'] = df['EMA_short'] - df['EMA_long']
    # Calculate the signal line
    df['Signal_line'] = df['MACD_line'].ewm(span=signal_span, adjust=False).mean()

    return df
    
def append_OBV(df: pd.DataFrame): 
    # Calculate the direction of close price
    close_direction = np.sign(df['close'].diff())
    # handle potential 0 cases
    close_direction.replace(to_replace=0, method='ffill', inplace=True)
    # Calculate On Balance Volume
    df['OBV'] = (df['vol'] * close_direction).cumsum()
    
    return df

# %%
def bound_signal(df: pd.DataFrame, target, upper_lim, lower_lim, name):
    # Get positive signals, note that lim can be both scalar or series
    top_signal = np.where(df[target] > upper_lim, 1, 0)
    # Get negative signals
    bottom_signal = np.where(df[target] < lower_lim, -1, 0)
    # combine signals
    df[f'{name}_bound_signal'] = top_signal + bottom_signal
    
    return df

def reverse_signal(df: pd.DataFrame, target, refer, upper_lim, lower_lim, name):
    # Get trends of target column and reference column
    target_trend = np.sign(df[target] - df[target].shift(1)).fillna(0)
    refer_trend = np.sign(df[refer] - df[refer].shift(1)).fillna(0)
    # Get reverse signals, observe if has upper or lower limit requirement
    if upper_lim & lower_lim:
        df[f'{name}_reverse_signal'] = target_trend.where(target_trend != refer_trend, 0).where((df[target] < lower_lim) | (df[target] > upper_lim), 0)
    elif upper_lim: 
        df[f'{name}_reverse_signal'] = target_trend.where(target_trend != refer_trend, 0).where((df[target] > upper_lim), 0)
    elif lower_lim: 
        df[f'{name}_reverse_signal'] = target_trend.where(target_trend != refer_trend, 0).where((df[target] < lower_lim), 0)
    else: 
        df[f'{name}_reverse_signal'] = target_trend.where(target_trend != refer_trend, 0)
    # Reverse the sign of the signal
    df[f'{name}_reverse_signal'] = df[f'{name}_reverse_signal'] * -1
    
    return df

# def crossover_signal(df: pd.DataFrame, target, refer, name, strict=False):
#     # Calculate the differences between target and reference
#     diff = df[target] - df[refer]
#     # Get the signs of the differences (-1, 0, 1)
#     sign = np.sign(diff)
#     # Find points where the sign changed
#     crossover_points = sign.diff().iloc[1:]
#     if strict:
#         # Get the directions of target and reference (1 for up, -1 for down, 0 for unchanged)
#         target_direction = np.sign(df[target].diff())
#         reference_direction = np.sign(df[refer].diff())
#         # Keep only the crossover points where target and reference move in opposite directions
#         crossover_points = crossover_points[(target_direction != reference_direction).iloc[1:]]
    
#     # get signal
#     signal = pd.Series(0, index=df.index)
#     signal[crossover_points != 0] = -1 * np.sign(crossover_points[crossover_points != 0])

#     df[f'{name}_crossover_signal'] = signal
    
#     return df

def crossover_signal(df, target, refer, name, strict=False):
    # compute the differences between two columns
    diff = df[target] - df[refer]

    # get the points where crossover happens
    crossover_points = ((diff > 0) & (diff.shift(1) < 0)) | ((diff < 0) & (diff.shift(1) > 0))
    
    if strict:
        direction_changed = ((df[target] - df[target].shift(1)) * (df[refer] - df[refer].shift(1))) < 0
        crossover_points = crossover_points & direction_changed
    
    # get signal
    signal = pd.Series(0, index=df.index)
    signal[crossover_points] = np.sign((df[target] - df[target].shift(1)) - (df[refer] - df[refer].shift(1)))
    
    df[f'{name}_crossover_signal'] = signal
    
    return df

# %%
def signal_metric(df: pd.DataFrame, signal, window=63, upper_q=95, lower_q=5):
    precisions = []
    recalls = []
    f1_scores = []
    for qscode in df['qscode'].unique():
        cur_df = df[df['qscode'] == qscode]
        cur_df['rolling_high'] = df['close'].rolling(window=window, center=True).apply(lambda x: np.percentile(x, upper_q))
        cur_df['rolling_low'] = df['close'].rolling(window=window, center=True).apply(lambda x: np.percentile(x, lower_q))
        # Create a binary variable for extreme prices
        cur_df['Extreme_High'] = np.where(cur_df['close'] >= cur_df['rolling_high'], 1, 0)
        cur_df['Extreme_Low'] = np.where(cur_df['close'] <= cur_df['rolling_low'], -1, 0)
        cur_df['Extreme'] = cur_df['Extreme_High'] + cur_df['Extreme_Low']
        
        # Calculate the precision
        true_positives = np.sum((cur_df['Extreme'] == cur_df[signal]) & (cur_df[signal] != 0))
        total_predicted_positives = np.sum(cur_df[signal] != 0)
        precision = true_positives / total_predicted_positives
        precisions.append(precision)
        
        # Calculate the recall
        total_actual_positives = np.sum(cur_df['Extreme'] != 0)
        recall = true_positives / total_actual_positives
        recalls.append(recall)
        
        # Calculate the F1 score
        f1_score = 2 * (precision * recall) / (precision + recall)
        f1_scores.append(f1_score)
        
    return np.mean(precisions), np.mean(recalls), np.mean(f1_scores)



# %%
def plot_signal(df: pd.DataFrame, qscode, signal, target, add_signal_line = True):
    plot_df = df[df['qscode'] == qscode].set_index('date')
    plot_df.index = pd.to_datetime(plot_df.index)
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.set_title(f"{target} Plot of {qscode} with {signal}")
    ax.set_xlabel('Date')
    ax.set_ylabel(target)
    ax.plot(plot_df.index, plot_df[target], color='black')
    ax.plot(plot_df.index, plot_df[target].where(plot_df[signal] == 1), color='green', alpha=0.3, linewidth=10)
    ax.plot(plot_df.index, plot_df[target].where(plot_df[signal] == -1), color='orange', alpha=0.3, linewidth=10)
    
    # Iterate over the signal series
    if add_signal_line: 
        for i in range(1, len(plot_df)):
            if plot_df[signal].iloc[i-1] == 0 and plot_df[signal].iloc[i] == 1:
                ax.axvline(x=plot_df.index[i], color='red', linestyle='dotted', alpha=0.4)
            elif plot_df[signal].iloc[i-1] == 0 and plot_df[signal].iloc[i] == -1:
                ax.axvline(x=plot_df.index[i], color='green', linestyle='dotted', alpha=0.4)
    
    ax.grid()
    plt.show()
    
# %%
def extreme_price_scanner(df: pd.DataFrame, window=63, upper_q=95, lower_q=5):
    # Calculate the local top and bottom 5 percentile prices
    df['roll_high'] = df['close'].rolling(window, center=True).apply(lambda x: np.percentile(x, upper_q))
    df['roll_low'] = df['close'].rolling(window, center=True).apply(lambda x: np.percentile(x, lower_q))

    # Create a binary variable for extreme prices
    df['Extreme_High'] = np.where(df['close'] >= df['roll_high'], 1, 0)
    df['Extreme_Low'] = np.where(df['close'] <= df['roll_low'], -1, 0)
    df['Extreme'] = df['Extreme_High'] + df['Extreme_Low']
    
    return df
# %%
df = data.copy()
df = df.groupby('qscode', group_keys=False).apply(lambda x: append_CCI(x))
df = df.groupby('qscode', group_keys=False).apply(lambda x: append_RSI(x))
df = df.groupby('qscode', group_keys=False).apply(lambda x: append_Bollinger(x))
df = df.groupby('qscode', group_keys=False).apply(lambda x: append_MACD(x))
df = df.groupby('qscode', group_keys=False).apply(lambda x: append_OBV(x))
# %%
# CCI Signals
df = df.groupby('qscode', group_keys=False).apply(lambda x: bound_signal(x, "CCI", 100, -100, "CCI"))
df = df.groupby('qscode', group_keys=False).apply(lambda x: reverse_signal(x, "CCI", "close", 100, -100, "CCI"))

# %%
# RSI Signals
df = df.groupby('qscode', group_keys=False).apply(lambda x: bound_signal(x, "RSI", 70, 30, "RSI"))
df = df.groupby('qscode', group_keys=False).apply(lambda x: reverse_signal(x, "RSI", "close", 70, 30, "RSI"))

# %%
# Bollinger Signals
df = df.groupby('qscode', group_keys=False).apply(lambda x: bound_signal(x, "close", x['Upper_BB'], x['Lower_BB'], "Bollinger"))

# %%
# MACD Signals
df = df.groupby('qscode', group_keys=False).apply(lambda x: crossover_signal(x.reset_index(drop=True), "MACD_line", "Signal_line", "MACD").reset_index(drop=True))

# %%
# OBV Signals
df = df.groupby('qscode', group_keys=False).apply(lambda x: reverse_signal(x, "OBV", "close", False, False, "OBV"))

# %%
plot_signal(df, "000001.SH", "CCI_reverse_signal", "close")
# %%
plot_signal(df[df['date'] > '2022-07-14'], "000905.SH", "OBV_reverse_signal", "close")


# %%
df = df.groupby('qscode', group_keys=False).apply(lambda x: extreme_price_scanner(x, window=63, upper_q=95, lower_q=5))
# %%
plot_signal(df, "000300.SH", "Extreme", "close")


# %%
signal_metric(df, "RSI_reverse_signal", window=63, upper_q=95, lower_q=5)



# %%
def extreme_price_scanner(df: pd.DataFrame, window=63, upper_q=95, lower_q=5, change_lim=0.2):
    # Calculate the local top and bottom 5 percentile prices
    df['roll_high'] = df['close'].rolling(window).apply(lambda x: np.percentile(x, upper_q))
    df['roll_low'] = df['close'].rolling(window).apply(lambda x: np.percentile(x, lower_q))
    df['shift_return'] = np.abs(df['close'].pct_change(window))

    # Create a binary variable for extreme prices
    df['Extreme_High'] = np.where((df['close'] >= df['roll_high']) & (df['shift_return'] >= change_lim), 1, 0)
    df['Extreme_Low'] = np.where((df['close'] <= df['roll_low']) & (df['shift_return'] >= change_lim), -1, 0)
    df['Extreme'] = df['Extreme_High'] + df['Extreme_Low']
    
    return df

# %%
test_df = pd.DataFrame({
    'date': ['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04', '2021-01-05'], 
    'close': [0.3, 0.35, 0.5, 0.5, 0.2]
})
# %%
extreme_price_scanner(test_df)