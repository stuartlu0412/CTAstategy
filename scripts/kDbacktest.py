import numpy as np
import pandas as pd
import vectorbt as vbt

from binance import Client
from binance.enums import HistoricalKlinesType 

#get binance data
client = Client()
value = client.get_historical_klines(symbol = 'BTCUSDT', 
                                      interval = '1h', 
                                      start_str = '2020-01-01', 
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
price = df[['High', 'Low', 'Close']][:17520]

k_win = np.arange(3, 50)
percent_k, percent_d = vbt.STOCH.run_combs(price['High'], price['Low'], price['Close'], k_window=k_win)

entries = percent_k.percent_k_crossed_above(70)
exits = percent_k.percent_k_crossed_above(30)

portfolio = vbt.Portfolio.from_signals(price['Close'], entries, exits, freq='1h', direction='both')

print(portfolio.sharpe_ratio().sort_values())


#testing
#test_data = vbt.BinanceData.download("BTCUSDT", start='2022-01-01', end='2023-01-01', interval='1d')

#test_price = test_data.get('Close')

test_price = df[['High', 'Low', 'Close']][17520:]

k_win = np.arange(3, 50)
percent_k, percent_d = vbt.STOCH.run(test_price['High'], test_price['Low'], test_price['Close'], k_window=42)
#filter_ = vbt.MA.run(test_price, 40)

entries = percent_k.percent_k_crossed_above(70)
exits = percent_k.percent_k_crossed_above(30)

portfolio = vbt.Portfolio.from_signals(test_price['Close'], entries, exits, freq='1h', direction='both')
portfolio.plot().show()

print(portfolio.sharpe_ratio())