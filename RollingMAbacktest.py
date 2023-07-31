import numpy as np
import pandas as pd
import vectorbt as vbt

from binance import Client
from binance.enums import HistoricalKlinesType 

#get binance data
client = Client()
value = client.get_historical_klines(symbol = 'BTCUSDT', 
                                      interval = '1d', 
                                      start_str = '2018-01-01', 
                                      end_str = '2023-07-01',
                                      klines_type=HistoricalKlinesType.FUTURES)

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


#backtest
price = df['Close']

(in_sample_prices, in_sample_dates), (out_sample_prices, out_sample_dates) = price.vbt.rolling_split(n=20, window_len=365 * 2, set_lens=(180, ), left_to_right=False)

windows = np.arange(10, 100)
fastMA, slowMA = vbt.MA.run_combs(in_sample_prices, window=windows)
 
entries = fastMA.ma_crossed_above(slowMA)
exits = fastMA.ma_crossed_below(slowMA)

portfolio = vbt.Portfolio.from_signals(in_sample_prices, entries, exits, freq='1d', direction='both')
performance = portfolio.sharpe_ratio()
print(performance.sort_values())
print(performance[performance.groupby('split_idx').idxmax()].index)

'''
#testing
test_data = vbt.BinanceData.download("BTCUSDT", start='2022-01-01', end='2023-01-01', interval='1d')

#test_price = test_data.get('Close')
test_price = df['Close'][730:]

fastMA = vbt.MA.run(test_price, 31)
slowMA = vbt.MA.run(test_price, 36)
#filter_ = vbt.MA.run(test_price, 40)

entries = fastMA.ma_crossed_above(slowMA)# & filter_.close_crossed_above(filter_)
exits = fastMA.ma_crossed_below(slowMA)

portfolio = vbt.Portfolio.from_signals(test_price, entries, exits, freq='1d', direction='both')
portfolio.plot().show()

print(portfolio.sharpe_ratio())
'''