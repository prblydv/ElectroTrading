import pandas as pd
import numpy as np
from scipy.signal import argrelextrema



def getprices(df):
    c1 = df['Close']
    c0 = df['Close'].shift(-1)
    c2 = df['Close'].shift(1)
    
    o1 = df['Open']
    o0 = df['Open'].shift(-1)
    o2 = df['Open'].shift(1)

    h1 = df['High']
    h0 = df['High'].shift(-1)
    h2 = df['High'].shift(1)

    l0 = df['Low'].shift(-1)
    l1 = df['Low']
    l2 = df['Low'].shift(1)

    return c0 , c1, c2, o0, o1, o2, l0, l1, l2, h0, h1, h2





def doji(df):
    doji_criteria = (
        abs(df['Close'] - df['Open']) / (df['High'] - df['Low']) < 0.1) & \
        ((df['High'] - df[['Close', 'Open']].max(axis=1)) > (3 * abs(df['Close'] - df['Open']))) & \
        ((df[['Close', 'Open']].min(axis=1) - df['Low']) > (3 * abs(df['Close'] - df['Open'])))
    
    # Create a new column 'Doji' to indicate presence of a Doji pattern
    df['Doji'] = doji_criteria.astype(int)
    return df

def three_crows(df):
    c0 , c1, c2, o0, o1, o2, l0, l1, l2, h0, h1, h2 = getprices(df)
    
    two_black_crows = (
        (c0 < o0) &  # First candle is black
        (c1 < o1) &  # Second candle is black
        (o1 < o0) &
        (c1 < c0)
    )
    three_black_crows = (
        (c0 < o0) &  # First candle is black
        (c1 < o1) &  # Second candle is black
        (c2 < o2) &  # Third candle is black
        (o2 < o1) &  # Downtrend for opens
        (o1 < o0) &
        (c2 < c1) &  # Downtrend for closes
        (c1 < c0)
    )

    two_white_soldiers = (
        (c0 > o0) &  # First candle is white
        (c1 > o1) &  # Second candle is white
        (o1 > o0) &
        (c1 > c0)
    )
    three_white_soldiers = (
        (c0 > o0) &  # First candle is white
        (c1 > o1) &  # Second candle is white
        (c2 > o2) &  # Third candle is white
        (o2 > o1) &  # Uptrend for opens
        (o1 > o0) &
        (c2 > c1) &  # Uptrend for closes
        (c1 > c0)
    )
    df['Three Crows Pattern'] = 0
    df.loc[three_black_crows, 'Three Crows Pattern'] += -2
    df.loc[three_white_soldiers, 'Three Crows Pattern'] += 2
    df.loc[two_black_crows, 'Three Crows Pattern'] += -1
    df.loc[two_white_soldiers, 'Three Crows Pattern'] += 1

    return df

def get_max_min(df, smoothing=2, window_range=16):
    # Apply smoothing to the 'Close' column and drop NaN values that come from rolling mean
    smooth_prices = df['Close'].rolling(window=smoothing).mean().dropna()

    # Identify local maxima and minima
    local_max = argrelextrema(smooth_prices.values, np.greater)[0]
    local_min = argrelextrema(smooth_prices.values, np.less)[0]

    # Initialize a score series with NaN values
    scores = pd.Series(index=df.index, dtype=float)

    # Assign scores for maxima and minima within the window range
    for i in local_max:
        if (i >= window_range) and (i < len(df) - window_range):
            scores.iloc[i] = 1
    for i in local_min:
        if (i >= window_range) and (i < len(df) - window_range):
            scores.iloc[i] = -1

    # Forward-fill the scores for maxima and minima
    scores.ffill(inplace=True)
    scores.bfill(inplace=True)  # Ensure the series is fully populated

    # Linearly interpolate between the scores to get a gradual transition between maxima and minima
    scores = scores.interpolate(method='linear')

    # Add scores to the original dataframe
    df['Scores'] = scores

    return df



def get_max_min_scores(df, smoothing=2, window_range=16):
    # Smooth the close prices
    smooth_prices = df['Close'].rolling(window=smoothing).mean().dropna()

    # Find local maxima and minima indices
    local_max_indices = argrelextrema(smooth_prices.values, np.greater, order=window_range)[0]
    local_min_indices = argrelextrema(smooth_prices.values, np.less, order=window_range)[0]

    # Initialize an empty scores series to store our results
    scores = pd.Series(index=df.index, dtype=float)

    # Loop through each local max and min to assign scores
    for i in range(len(local_max_indices)):
        if i < len(local_max_indices) - 1:
            # Define the range between the current max and the next min
            max_idx = local_max_indices[i]
            if i < len(local_min_indices) - 1:
                min_idx = local_min_indices[i+1]
            else:
                min_idx = df.index[-1]  # Last available index if no more mins

            # Assign scores for the range
            max_price = smooth_prices.iloc[max_idx]
            min_price = smooth_prices.iloc[min_idx]
            price_range = max_price - min_price

            for idx in range(max_idx, min_idx + 1):
                if idx in df.index:
                    curr_price = smooth_prices.iloc[idx]
                    # Calculate score based on closeness to max or min
                    scores.iloc[idx] = ((curr_price - min_price) / price_range) * 2 - 1

    # Forward fill the remaining NaNs in scores
    scores.ffill(inplace=True)
    scores.bfill(inplace=True)  # Fill NaNs at the beginning

    # Assign scores to original df
    df['Scores'] = scores

    return df

def adxx(high_prices, low_prices, close_prices, period=14):
    """
    Calculate Average Directional Index (ADX)
    
    Parameters:
        high_prices (Series): High prices of the asset.
        low_prices (Series): Low prices of the asset.
        close_prices (Series): Close prices of the asset.
        period (int): Number of periods. Default is 14.
    
    Returns:
        Series: Average Directional Index (ADX) values.
    """
    high_low_diff = high_prices - low_prices
    high_close_diff = (high_prices - close_prices.shift(1)).abs()
    low_close_diff = (low_prices - close_prices.shift(1)).abs()
    
    plus_dm = high_close_diff.where((high_close_diff > low_close_diff) & (high_close_diff > 0), 0)
    minus_dm = low_close_diff.where((low_close_diff > high_close_diff) & (low_close_diff > 0), 0)
    
    tr = high_low_diff.combine(high_close_diff.combine(low_close_diff, max), max)
    
    atr = tr.rolling(window=period).mean()
    plus_di = (plus_dm / atr).rolling(window=period).mean() * 100
    minus_di = (minus_dm / atr).rolling(window=period).mean() * 100
    
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).abs()).rolling(window=period).mean() * 100
    adx = dx.rolling(window=period).mean()
    
    return adx




def adx(forex_data, period):
    forex_data['ADX'] = adxx(forex_data['High'], forex_data['Low'], forex_data['Close'], period)
    return forex_data













def hammer(df):
    # Calculate the candle components
    real_body = abs(df['Close'] - df['Open'])
    candle_range = df['High'] - df['Low']
    upper_shadow = df['High'] - np.maximum(df['Close'], df['Open'])
    lower_shadow = np.minimum(df['Close'], df['Open']) - df['Low']

    # Hammer criteria
    hammer = (
        (real_body / candle_range <= 0.33) &  # Small real body
        (lower_shadow >= 2 * real_body) &  # Long lower shadow
        (upper_shadow <= real_body)  # Little to no upper shadow
    )
    
    # Inverted Hammer criteria
    inverted_hammer = (
        (real_body / candle_range <= 0.33) &  # Small real body
        (upper_shadow >= 2 * real_body) &  # Long upper shadow
        (lower_shadow <= real_body)  # Little to no lower shadow
    )
    
    # Assign +1 for Hammer, -1 for Inverted Hammer, and 0 otherwise
    df['Hammer_Signal'] = np.where(hammer, 1, np.where(inverted_hammer, -1, 0))
    
    return df