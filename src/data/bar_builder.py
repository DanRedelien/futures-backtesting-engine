import pandas as pd
import numpy as np

class BarBuilder:
    """
    Transforms base 'time' bars (e.g. 5m) into alternative representations
    such as Volume Bars, Range Bars, or Heikin-Ashi before feeding to the strategy.
    """
    
    @staticmethod
    def build(df: pd.DataFrame, bar_type: str, bar_size: float = 0.0, tick_size: float = 1.0) -> pd.DataFrame:
        """
        Main entry point for generating bars.
        
        Args:
            df: The raw time-based dataframe (must have open, high, low, close, volume).
            bar_type: 'time', 'volume', 'range', 'heikin_ashi'
            bar_size: Threshold parameter for Volume and Range bars.
            tick_size: Minimum tick increment for the asset (needed for Range Bars).
            
        Returns:
            A new aggregated DataFrame.
        """
        if df.empty:
            return df
            
        btype = bar_type.lower()
        
        if btype == 'time':
            return df.copy()
            
        elif btype == 'heikin_ashi':
            return BarBuilder._build_heikin_ashi(df)
            
        elif btype == 'volume':
            if bar_size <= 0:
                print("[BarBuilder] WARNING: Volume bars require bar_size > 0. Returning time bars.")
                return df.copy()
            return BarBuilder._build_volume_bars(df, bar_size)
            
        elif btype == 'range':
            if bar_size <= 0:
                print("[BarBuilder] WARNING: Range bars require bar_size (in ticks) > 0. Returning time bars.")
                return df.copy()
            return BarBuilder._build_range_bars(df, bar_size, tick_size)
            
        else:
            print(f"[BarBuilder] WARNING: Unknown bar_type '{bar_type}'. Returning time bars.")
            return df.copy()
            
    @staticmethod
    def _build_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Heikin-Ashi formulas:
        HA_Close = (Open + High + Low + Close) / 4
        HA_Open = (Previous HA_Open + Previous HA_Close) / 2
        HA_High = Max(High, HA_Open, HA_Close)
        HA_Low = Min(Low, HA_Open, HA_Close)
        """
        print("[BarBuilder] Generating Heikin-Ashi bars...")
        ha_df = df.copy()
        
        ha_close = (ha_df['open'] + ha_df['high'] + ha_df['low'] + ha_df['close']) / 4
        ha_df['close'] = ha_close
        
        ha_open = np.zeros(len(ha_df))
        ha_open[0] = ha_df['open'].iloc[0]
        
        for i in range(1, len(ha_df)):
            ha_open[i] = (ha_open[i-1] + ha_close.iloc[i-1]) / 2
            
        ha_df['open'] = ha_open
        ha_df['high'] = ha_df[['high', 'open', 'close']].max(axis=1)
        ha_df['low'] = ha_df[['low', 'open', 'close']].min(axis=1)
        
        return ha_df

    @staticmethod
    def _build_volume_bars(df: pd.DataFrame, volume_threshold: float) -> pd.DataFrame:
        """
        Aggregates time bars into Volume bars holding approximately `volume_threshold` volume each.
        We group standard bars together until the cumulative volume exceeds the threshold.
        """
        print(f"[BarBuilder] Generating Volume bars (Threshold: {volume_threshold})...")
        
        # We can use cumulative sum modulo to group bars
        # This is a vectorized approximation, much faster than a loop
        vol_cumsum = df['volume'].cumsum()
        
        # Each integer step represents a new bucket of `volume_threshold` size
        group_id = (vol_cumsum // volume_threshold)
        
        # Group by the bucket IDs
        grouped = df.groupby(group_id)
        
        # Aggregate logic
        vol_bars = grouped.agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Use the timestamp of the LAST bar in the bucket
        last_times = grouped.apply(lambda x: x.index[-1])
        vol_bars.index = last_times
        
        return vol_bars
        
    @staticmethod
    def _build_range_bars(df: pd.DataFrame, range_ticks: float, tick_size: float) -> pd.DataFrame:
        """
        Aggregates time bars into Range bars.
        A bar forms when the price moves `range_ticks` outside the open of the current bar.
        This requires a loop because range calculation is strictly path-dependent.
        """
        print(f"[BarBuilder] Generating Range bars (Size: {range_ticks} ticks)...")
        
        range_price = range_ticks * tick_size
        
        times = []
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []
        
        # Initialization
        c_open = df['open'].iloc[0]
        c_high = df['high'].iloc[0]
        c_low = df['low'].iloc[0]
        c_vol = 0
        
        for date, row in df.iterrows():
            c_high = max(c_high, row['high'])
            c_low = min(c_low, row['low'])
            c_vol += row['volume']
            
            # Check if Range is exceeded
            if (c_high - c_low) >= range_price:
                # Close the bar
                # The exact closing price depends on which side was pierced. 
                # For a rough approximation using base bars, use the current bar's close
                times.append(date)
                opens.append(c_open)
                highs.append(c_high)
                lows.append(c_low)
                closes.append(row['close'])
                volumes.append(c_vol)
                
                # Reset for next bar
                c_open = row['close']
                c_high = c_open
                c_low = c_open
                c_vol = 0
                
        # Handle the last unclosed bar
        if c_vol > 0 and len(times) > 0 and date != times[-1]:
             times.append(date)
             opens.append(c_open)
             highs.append(c_high)
             lows.append(c_low)
             closes.append(row['close'])
             volumes.append(c_vol)
             
        range_df = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        }, index=times)
        
        return range_df
