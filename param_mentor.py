class Signal1_l_entries(Signal):
    
    # 預設範圍
    # fast_period_rng = FAST_RANGE
    # slow_period_rng = SLOW_RANGE
    
    def __init__(self, 
                 FAST_RANGE: Union[int, Iterable[int]]=FAST_RANGE,
                 SLOW_RANGE: Union[int, Iterable[int]]=SLOW_RANGE,
                 direction: str='l',
                 in_out: str='in',
                 reverse=False) -> None:
        super().__init__(direction=direction, in_out=in_out)
        
        self.params = {
            "FAST_RANGE": iterate(FAST_RANGE),
            "SLOW_RANGE": iterate(SLOW_RANGE)
            }
        self.reverse=reverse
        
    # 回傳實際entries的dataframe
    def value(self, data: pd.DataFrame) -> pd.DataFrame:
        comb = np.array(list(product(*(self.params.values())))).T
        # list(product([data], *(self.params.values())))
        fast_ma = vbt.MA.run(data['close'], comb[0], short_name=f'FAST_RANGE_{self.direction}_{self.in_out}')
        slow_ma = vbt.MA.run(data['close'], comb[1], short_name=f'SLOW_RANGE_{self.direction}_{self.in_out}')
        signal = fast_ma.ma_crossed_above(slow_ma)
        cols = pd.MultiIndex.from_product([*(self.params.values()) , set(data.columns.get_level_values(1))])
        cols.set_names([*(self.param_names()), None], inplace=True)  #給定multi columns 的name
        signal.columns = cols
        entries = signal
        return entries if not self.reverse else ~entries