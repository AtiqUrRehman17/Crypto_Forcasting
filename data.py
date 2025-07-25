import requests
import pandas as pd

def get_klines(symbol='BTCUSDT', interval='1d', limit=1000, start_time=None, end_time=None):
    """
    Fetch full candlestick (kline) data from Binance API.
    """
    url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }

    if start_time:
        params['startTime'] = start_time
    if end_time:
        params['endTime'] = end_time

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    # Convert to DataFrame with all useful features
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
    ])

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

    # Convert numerical columns to float
    num_cols = ['open', 'high', 'low', 'close', 'volume',
                'quote_asset_volume', 'taker_buy_base_volume', 'taker_buy_quote_volume']
    df[num_cols] = df[num_cols].astype(float)

    # Drop the 'ignore' column (not useful)
    df.drop(columns=['ignore'], inplace=True)

    return df

# Example usage
if __name__ == "__main__":
    symbol = 'BTCUSDT'
    interval = '1d'
    df = get_klines(symbol=symbol, interval=interval)

    # Save full feature set
    df.to_csv(f"{symbol}_{interval}_full_klines.csv", index=False)
    print(f"Saved {len(df)} rows with full features to CSV.")
