import numpy as np
import pandas as pd
import vectorbt as vbt
import quantstats as qs

from binance import Client
from binance.enums import HistoricalKlinesType

def retrieve_data(symbol = 'BTCUSDT', 
                  interval = '15m', 
                  start_str = '2020-01-01', 
                  end_str = '2023-06-30', 
                  klines_type=HistoricalKlinesType.FUTURES) -> pd.DataFrame:
    
    client = Client()
    value = client.get_historical_klines(symbol = symbol, 
                                        interval = interval, 
                                        start_str = start_str, 
                                        end_str = end_str,
                                        klines_type = klines_type)

    columns_name = ['openTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'closeTime', 
                    'quoteVolume', 'numTrade', 'takerBuyVolume', 'takerBuyQuoteVolume', 'ignore']

    df = pd.DataFrame(value)
    df.columns = columns_name
    df['openTime']= pd.to_datetime(df['openTime'], unit='ms')
    df = df.drop(['ignore', 'closeTime'], axis=1)
    df = df.sort_values('openTime', ascending=True)
    df = df.set_index('openTime')
    df = df.astype(float)
    df['takerSellVolume'] = df['Volume'] - df['takerBuyVolume']
    df['takerSellQuoteVolume'] = df['quoteVolume'] - df['takerBuyQuoteVolume']
    df['avgTradeVolume'] = df['quoteVolume'] / df['numTrade']
    df = df[~df.index.duplicated(keep='first')]
    
    return df

class Strategy():

    def rsi_long(close, rsi_window = 32, filter_window = 39):
        
        def create_signal(close, rsi_window = rsi_window, filter_window = filter_window, ma_freq = '1H'):

            rsi_crossover = vbt.RSI.run(close, window=rsi_window).rsi_crossed_above(70).to_numpy()
            rsi_crossunder = vbt.RSI.run(close, window=rsi_window).rsi_crossed_below(30).to_numpy()

            rsi_long = vbt.RSI.run(close, window = filter_window).rsi.to_numpy()
            close = close.to_numpy()

            trend = np.where(rsi_crossover & (rsi_long > 70), 1, 0)
            trend = np.where(rsi_crossunder, -1, trend)

            return trend
        
        strategy = vbt.IndicatorFactory(
            class_name = 'long_strategy',
            short_name = 'long',
            input_names = ['close'],
            param_names = ['rsi_window', 'filter_window', 'ma_freq'],
            output_names = ['signals'] 
        ).from_apply_func(create_signal, keep_pd=True)

        signal = strategy.run(close , rsi_window=32, filter_window=39, ma_freq='1H', param_product = True)

        return signal.signals
    
    def rsi_ma(close, rsi_window = 32, filter_window = 39):
        
        def create_signal(close, rsi_window = rsi_window, filter_window = filter_window, ma_freq = '1H'):

            rsi_crossover = vbt.RSI.run(close, window=rsi_window).rsi_crossed_above(70).to_numpy()
            rsi_crossunder = vbt.RSI.run(close, window=rsi_window).rsi_crossed_below(30).to_numpy()

            ma = vbt.MA.run(close, window = filter_window).ma.to_numpy()
            close = close.to_numpy()

            trend = np.where(rsi_crossover & (close > ma), 1, 0)
            trend = np.where(rsi_crossunder, -1, trend)

            return trend
        
        strategy = vbt.IndicatorFactory(
            class_name = 'long_strategy',
            short_name = 'long',
            input_names = ['close'],
            param_names = ['rsi_window', 'filter_window', 'ma_freq'],
            output_names = ['signals'] 
        ).from_apply_func(create_signal, keep_pd=True)

        signal = strategy.run(close, rsi_window=rsi_window, filter_window=filter_window, ma_freq='1H', param_product = True)

        return signal.signals
    
    