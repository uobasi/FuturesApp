# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 23:56:48 2025

@author: uobas
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 18:30:37 2025

@author: uobas
"""
import csv
import io
from datetime import datetime, timedelta, date, time
import pandas as pd 
import numpy as np
import math
from google.cloud.storage import Blob
from google.cloud import storage
import plotly.graph_objects as go
from plotly.subplots import make_subplots
np.seterr(divide='ignore', invalid='ignore')
pd.options.mode.chained_assignment = None
#from scipy.signal import argrelextrema
from scipy import signal
from scipy.misc import derivative
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=RuntimeWarning)
import plotly.io as pio
pio.renderers.default='browser'
import bisect


from dataclasses import dataclass
from typing import List, Dict, Literal, Optional, Any
import math
import numpy as np

Side = Literal["B", "A"]  # B = buyer aggressor, A = seller aggressor


@dataclass
class Trade:
    price: float
    size: float
    side: Side
    time: Any  # string, datetime, or int – whatever you use
    raw: Any = None  # optional: store original row if you want


@dataclass
class Candle:
    open: float
    high: float
    low: float
    close: float
    start_time: Any
    end_time: Any
    volume: float = 0.0    

@dataclass    
class NodeProfile:
    hist: Dict[float, float]                # volume at price
    buy_vol: Dict[float, float]            # buy volume at price
    sell_vol: Dict[float, float]           # sell volume at price

    poc: float                             # Point of Control
    val: float                             # Value Area Low
    vah: float                             # Value Area High
    pov: float                             # Point of Void (LVN inside VA)

    total_vol: float
    node_strength: float                   # poc_vol / total_vol

    profile_buy_pct: float                 # overall profile buy %
    poc_buy_pct: float                     # buy % at POC

    vwap: Optional[float] = None
    poc_vwap_distance: Optional[float] = None  # |poc - vwap| / vwap

    # for debugging / tooling
    meta: Dict[str, Any] = None
    
    
SignalType = Literal[
    "poc_bull_reject",
    "poc_bear_reject",
    "pov_bull_reject",
    "pov_bear_reject",
]


@dataclass
class BreachSignal:
    signal_type: SignalType
    level_price: float     # POC or PoV price
    time: Any              # candle end_time
    candle_index: int
    candle: Candle

    node_profile: NodeProfile
    extra: Dict[str, Any]
    
    
class NodeBreachEngine:
    def __init__(
        self,
        bin_size: float = 0.25,
        value_area_pct: float = 0.68,
        compute_vwap: bool = True,
    ):
        self.bin_size = bin_size
        self.value_area_pct = value_area_pct
        self.compute_vwap_flag = compute_vwap

    # ---------- helpers ----------

    def _round_to_bin(self, price: float, p_min: float) -> float:
        """
        Round price to nearest bin starting from p_min.
        """
        steps = round((price - p_min) / self.bin_size)
        return p_min + steps * self.bin_size

    # ---------- profile construction ----------

    def build_profile(self, trades: List[Trade]) -> NodeProfile:
        """
        Build a single volume profile from a list of trades
        (e.g., a full session or pivot-to-pivot window).
        """
        if not trades:
            raise ValueError("No trades provided to build profile.")

        prices = np.array([t.price for t in trades], dtype=float)
        p_min, p_max = prices.min(), prices.max()

        # define bins from p_min to p_max
        num_bins = int(math.floor((p_max - p_min) / self.bin_size)) + 1
        bins = [p_min + i * self.bin_size for i in range(num_bins + 1)]

        hist = {b: 0.0 for b in bins}
        buy_vol = {b: 0.0 for b in bins}
        sell_vol = {b: 0.0 for b in bins}

        total_vol = 0.0
        total_buy = 0.0

        for t in trades:
            b = self._round_to_bin(t.price, p_min)
            hist[b] = hist.get(b, 0.0) + t.size
            total_vol += t.size

            if t.side == "B":
                buy_vol[b] = buy_vol.get(b, 0.0) + t.size
                total_buy += t.size
            elif t.side == "A":
                sell_vol[b] = sell_vol.get(b, 0.0) + t.size

        if total_vol <= 0:
            raise ValueError("Total volume is zero – cannot build profile.")

        # POC
        poc = max(hist, key=hist.get)
        poc_vol = hist[poc]

        # Value Area
        val, vah = self._compute_value_area(hist)

        # PoV (LVN inside VA)
        pov = self._compute_pov(hist, val, vah)

        # Node Strength
        node_strength = poc_vol / total_vol

        # Buy % overall / at POC
        profile_buy_pct = total_buy / total_vol
        poc_buy = buy_vol.get(poc, 0.0)
        poc_buy_pct = poc_buy / poc_vol if poc_vol > 0 else 0.0

        # VWAP (optional)
        vwap = None
        poc_vwap_distance = None
        if self.compute_vwap_flag:
            vwap = (
                sum(t.price * t.size for t in trades)
                / sum(t.size for t in trades)
            )
            poc_vwap_distance = abs(poc - vwap) / vwap if vwap != 0 else None

        return NodeProfile(
            hist=hist,
            buy_vol=buy_vol,
            sell_vol=sell_vol,
            poc=poc,
            val=val,
            vah=vah,
            pov=pov,
            total_vol=total_vol,
            node_strength=node_strength,
            profile_buy_pct=profile_buy_pct,
            poc_buy_pct=poc_buy_pct,
            vwap=vwap,
            poc_vwap_distance=poc_vwap_distance,
            meta={"p_min": p_min, "p_max": p_max},
        )

    def _compute_value_area(self, hist: Dict[float, float]) -> (float, float):
        """
        Classic volume-based value area:
        sort by volume desc, accumulate until value_area_pct of total volume.
        """
        total_volume = sum(hist.values())
        target = self.value_area_pct * total_volume

        rows = sorted(hist.items(), key=lambda x: x[1], reverse=True)

        cum = 0.0
        va_prices = []
        for price, vol in rows:
            va_prices.append(price)
            cum += vol
            if cum >= target:
                break

        val = min(va_prices)
        vah = max(va_prices)
        return val, vah

    def _compute_pov(self, hist: Dict[float, float], val: float, vah: float) -> float:
        """
        PoV = LVN inside the Value Area (lowest-volume bin between VAL & VAH).
        """
        va_hist = {p: v for p, v in hist.items() if val <= p <= vah and v > 0}
        if not va_hist:
            # fallback: just use VAL
            return val
        pov = min(va_hist, key=va_hist.get)
        return pov
    
    
def compute_atr(candles: List[Candle], period: int = 14) -> List[Optional[float]]:
    """
    Simple ATR implementation (Wilder smoothing would be more exact,
    but this is fine for a guard filter).
    Returns list of ATR values (same length as candles).
    """
    if len(candles) < 2:
        return [None] * len(candles)

    trs = []
    for i in range(len(candles)):
        if i == 0:
            trs.append(candles[i].high - candles[i].low)
        else:
            high = candles[i].high
            low = candles[i].low
            prev_close = candles[i - 1].close
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close),
            )
            trs.append(tr)

    atr_values: List[Optional[float]] = [None] * len(candles)
    for i in range(len(candles)):
        if i < period:
            atr_values[i] = None
        else:
            window = trs[i - period + 1 : i + 1]
            atr_values[i] = sum(window) / period

    return atr_values



class NodeBreachEngine(NodeBreachEngine):  # extending the class above

    def detect_breaches(
        self,
        candles: List[Candle],
        profile: NodeProfile,
        atr_values: Optional[List[Optional[float]]] = None,
        atr_mult: float = 0.0,          # set >0 to enforce ATR guard
        wick_threshold: float = 0.0,    # 0–1 fraction of candle range as wick
        margin_pct: float = 0.0,        # margin around POC / PoV
    ) -> List[BreachSignal]:
        """
        Scan candles for POC/PoV breaches & rejections.

        Returns list of BreachSignal objects.
        """
        signals: List[BreachSignal] = []

        poc = profile.poc
        pov = profile.pov

        for i, c in enumerate(candles):
            atr = atr_values[i] if atr_values is not None else None

            # Skip if ATR guard is enabled and we don't have ATR yet
            if atr_mult > 0 and (atr is None or atr <= 0):
                continue

            # Adjust for margin band
            poc_low = poc * (1.0 - margin_pct)
            poc_high = poc * (1.0 + margin_pct)

            pov_low = pov * (1.0 - margin_pct)
            pov_high = pov * (1.0 + margin_pct)

            rng = c.high - c.low if c.high != c.low else 1e-9

            # Real body and wick sizes
            body = abs(c.close - c.open)
            upper_wick = c.high - max(c.close, c.open)
            lower_wick = min(c.close, c.open) - c.low

            # Optional wick filter: require wick to be at least wick_threshold * range
            def wick_ok_bearish() -> bool:
                if wick_threshold <= 0:
                    return True
                return upper_wick >= wick_threshold * rng

            def wick_ok_bullish() -> bool:
                if wick_threshold <= 0:
                    return True
                return lower_wick >= wick_threshold * rng

            # ATR momentum guard: require body >= atr_mult * atr
            def atr_ok() -> bool:
                if atr_mult <= 0 or atr is None:
                    return True
                return body >= atr_mult * atr

            # ---- POC Rejections ----

            # Bearish POC rejection: price tags POC, closes below
            if (
                c.high >= poc_low
                and c.low <= poc_high
                and c.close < poc
                and c.close < c.open  # bearish candle
                and wick_ok_bearish()
                and atr_ok()
            ):
                signals.append(
                    BreachSignal(
                        signal_type="poc_bear_reject",
                        level_price=poc,
                        time=c.end_time,
                        candle_index=i,
                        candle=c,
                        node_profile=profile,
                        extra={
                            "atr": atr,
                            "poc": poc,
                            "pov": pov,
                            "val": profile.val,
                            "vah": profile.vah,
                            "node_strength": profile.node_strength,
                            "poc_buy_pct": profile.poc_buy_pct,
                            "profile_buy_pct": profile.profile_buy_pct,
                        },
                    )
                )

            # Bullish POC rejection: price tags POC, closes above
            if (
                c.low <= poc_high
                and c.high >= poc_low
                and c.close > poc
                and c.close > c.open  # bullish candle
                and wick_ok_bullish()
                and atr_ok()
            ):
                signals.append(
                    BreachSignal(
                        signal_type="poc_bull_reject",
                        level_price=poc,
                        time=c.end_time,
                        candle_index=i,
                        candle=c,
                        node_profile=profile,
                        extra={
                            "atr": atr,
                            "poc": poc,
                            "pov": pov,
                            "val": profile.val,
                            "vah": profile.vah,
                            "node_strength": profile.node_strength,
                            "poc_buy_pct": profile.poc_buy_pct,
                            "profile_buy_pct": profile.profile_buy_pct,
                        },
                    )
                )

            # ---- PoV Rejections ----

            # Bearish PoV rejection: tag PoV, close below
            if (
                c.high >= pov_low
                and c.low <= pov_high
                and c.close < pov
                and c.close < c.open
                and wick_ok_bearish()
                and atr_ok()
            ):
                signals.append(
                    BreachSignal(
                        signal_type="pov_bear_reject",
                        level_price=pov,
                        time=c.end_time,
                        candle_index=i,
                        candle=c,
                        node_profile=profile,
                        extra={"atr": atr},
                    )
                )

            # Bullish PoV rejection: tag PoV, close above
            if (
                c.low <= pov_high
                and c.high >= pov_low
                and c.close > pov
                and c.close > c.open
                and wick_ok_bullish()
                and atr_ok()
            ):
                signals.append(
                    BreachSignal(
                        signal_type="pov_bull_reject",
                        level_price=pov,
                        time=c.end_time,
                        candle_index=i,
                        candle=c,
                        node_profile=profile,
                        extra={"atr": atr},
                    )
                )

        return signals
    
    
    

def parse_raw_trades(raw_rows: List[list]) -> List[Trade]:
    """
    Assumes: [price, size, trade_id, ?, seq, side, time_str]
    Customize indices if your schema differs.
    """
    trades: List[Trade] = []
    for row in raw_rows:
        price = float(row[0])
        size = float(row[1])
        side = row[4]  # 'B' or 'A'
        time_str = row[5]
        trades.append(Trade(price=price, size=size, side=side, time=time_str, raw=row))
    return trades


def df_to_candles(df: pd.DataFrame, interval: str = "5min"):
    """
    Convert your OHLCV dataframe into Candle objects.
    Timestamps are in nanoseconds (UTC) and converted to EST.
    """
    candles = []

    for _, row in df.iterrows():

        # Convert nanosecond timestamp (UTC) → EST
        start_ts = (
            pd.to_datetime(row["timestamp"], unit='ns')
            .tz_localize('UTC')
            .tz_convert('America/New_York')
        )

        # Add interval to get end time
        end_ts = start_ts + pd.to_timedelta(interval)

        candle = Candle(
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            start_time=start_ts,
            end_time=end_ts,
            volume=float(row["volume"]),
        )

        candles.append(candle)

    return candles



def plot_profile(profile: NodeProfile, title="Volume Profile"):
    # Extract prices and volumes
    prices = list(profile.hist.keys())
    volumes = list(profile.hist.values())
    
    # Sort by price ascending
    prices, volumes = zip(*sorted(zip(prices, volumes)))
    
    fig = go.Figure()
    
    # ---------------------------
    # 1. Histogram bars (horizontal)
    # ---------------------------
    fig.add_trace(go.Bar(
        x=volumes,
        y=prices,
        orientation='h',
        marker=dict(color='black'),
        name='Volume'
    ))
    
    # ---------------------------
    # 2. Highlight POC
    # ---------------------------
    fig.add_trace(go.Scatter(
        x=[max(volumes)*1.02],
        y=[profile.poc],
        mode="markers+text",
        text=["POC"],
        textposition="middle left",
        marker=dict(color="red", size=10),
        name="POC"
    ))
    
    # ---------------------------
    # 3. VAL / VAH lines
    # ---------------------------
    fig.add_trace(go.Scatter(
        x=[0, max(volumes)],
        y=[profile.val, profile.val],
        mode="lines",
        line=dict(color="blue", width=2, dash="dot"),
        name="VAL"
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, max(volumes)],
        y=[profile.vah, profile.vah],
        mode="lines",
        line=dict(color="blue", width=2, dash="dot"),
        name="VAH"
    ))
    
    # ---------------------------
    # 4. PoV (LVN inside VA)
    # ---------------------------
    fig.add_trace(go.Scatter(
        x=[max(volumes)*0.8],
        y=[profile.pov],
        mode="markers+text",
        text=["PoV"],
        textposition="middle right",
        marker=dict(color="green", size=9),
        name="PoV"
    ))
    
    # ---------------------------
    # 5. Value Area shading (FIXED)
    # ---------------------------
    fig.add_vrect(
        x0=0,
        x1=max(volumes),
        y0=profile.val,
        y1=profile.vah,
        fillcolor="LightBlue",
        opacity=0.15,
        line_width=0
    )
    
    # ---------------------------
    # Layout
    # ---------------------------
    fig.update_layout(
        title=title,
        xaxis_title="Volume",
        yaxis_title="Price",
        height=1000,
        bargap=0.1,
        showlegend=True
    )
    
    fig.show()


def plot_nbe_combined(df, profile: NodeProfile, poc_path, pov_path, vah_path, val_path, 
                      title="NBE — Candles + Developing Profile"):
    
    # Sort profile histogram for plotting
    prices = list(profile.hist.keys())
    volumes = list(profile.hist.values())
    prices, volumes = zip(*sorted(zip(prices, volumes)))

    max_vol = max(volumes)

    # ---------------------------
    # Create subplot layout
    # ---------------------------
    fig = make_subplots(rows=1, cols=2, shared_xaxes=True, shared_yaxes=True,
                        specs=[[{}, {}]], #[{"colspan": 1},{},][{}, {}, ]'+ '<br>' +' ( Put:'+str(putDecHalf)+'('+str(NumPutHalf)+') | '+'Call:'+str(CallDecHalf)+'('+str(NumCallHalf)+') ' (Sell:'+str(sum(sells))+') (Buy:'+str(sum(buys))+') 
                         horizontal_spacing=0.01, vertical_spacing=0.00, 
                         column_widths=[0.85, 0.15] ) #,row_width=[0.30, 0.70,]

    
    
    # fig = make_subplots(
    #     rows=1, cols=2,
    #     column_widths=[0.25, 0.75],    # left panel narrow (profile), right wide (candles)
    #     horizontal_spacing=0.03,
    #     specs=[[{"type": "xy"}, {"type": "xy"}]]
    # )

    # ======================================================================
    # LEFT PANEL — VOLUME PROFILE
    # ======================================================================
    fig.add_trace(go.Bar(
        x=volumes,
        y=prices,
        orientation='h',
        marker=dict(color='white'),
        name='Volume Profile'
    ), row=1, col=2)

    # POC
    fig.add_trace(go.Scatter(
        x=[max_vol*1.02],
        y=[profile.poc],
        mode="markers+text",
        text=["POC"],
        textposition="middle left",
        marker=dict(color="red", size=12),
        name="POC"
    ), row=1, col=2)

    # VAL line
    fig.add_trace(go.Scatter(
        x=[0, max_vol],
        y=[profile.val, profile.val],
        mode="lines",
        line=dict(color="blue", width=2, dash="dot"),
        name="VAL"
    ), row=1, col=2)

    # VAH line
    fig.add_trace(go.Scatter(
        x=[0, max_vol],
        y=[profile.vah, profile.vah],
        mode="lines",
        line=dict(color="blue", width=2, dash="dot"),
        name="VAH"
    ), row=1, col=2)

    # PoV marker
    fig.add_trace(go.Scatter(
        x=[max_vol*0.8],
        y=[profile.pov],
        mode="markers+text",
        text=["PoV"],
        textposition="middle right",
        marker=dict(color="green", size=10),
        name="PoV"
    ), row=1, col=2)

    # Value Area shading
    fig.add_vrect(
        x0=0,
        x1=max_vol,
        y0=profile.val,
        y1=profile.vah,
        fillcolor="LightBlue",
        opacity=0.35,
        line_width=0,
        row=1,
        col=2
    )

    # ======================================================================
    # RIGHT PANEL — CANDLESTICKS + DEVELOPING POC/POV
    # ======================================================================
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name="Candles"
    ), row=1, col=1)

    # Developing POC
    fig.add_trace(go.Scatter(
        x=df.index, y=poc_path,
        mode="lines",
        line=dict(color="red", width=2),
        name="Dev POC",
        connectgaps=True
    ), row=1, col=1)

    # Developing PoV
    fig.add_trace(go.Scatter(
        x=df.index, y=pov_path,
        mode="lines",
        line=dict(color="green", width=2, dash="dot"),
        name="Dev PoV",
        connectgaps=True
    ), row=1, col=1)

    # Developing VAH
    fig.add_trace(go.Scatter(
        x=df.index, y=vah_path,
        mode="lines",
        line=dict(color="white", width=2, dash="dot"),
        name="Dev VAH",
        connectgaps=True
    ), row=1, col=1)

    # Developing VAL
    fig.add_trace(go.Scatter(
        x=df.index, y=val_path,
        mode="lines",
        line=dict(color="white", width=2, dash="dot"),
        name="Dev VAL",
        connectgaps=True
    ), row=1, col=1)
    
    
    # try:
    #     #pass
    #     fig.add_trace(go.Scatter(x=df.index, y=df['PreviousDayPOC'], mode='lines', name='PreviousDayPOC'))
    #     fig.add_trace(go.Scatter(x=df.index, y=df['PreviousDayHVA'], mode='lines', name='PreviousDayHVA'))
    #     fig.add_trace(go.Scatter(x=df.index, y=df['PreviousDayLVA'], mode='lines', name='PreviousDayLVA'))
    # except(KeyError):
    #     pass

    # ======================================================================
    # Final layout
    # ======================================================================
    fig.update_layout(
        title=title,
        height=900,
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_dark",
        xaxis_rangeslider_visible=False, showlegend=False,
    )
    fig.update_xaxes(showticklabels=False, row=1, col=1)


    return fig 

from scipy.signal import argrelextrema

def detect_swings(df, order=5):
    """
    Detect swing highs/lows using price CLOSE.
    Returns two lists: swing_high_indices, swing_low_indices
    """
    prices = df['close'].values

    swing_highs = argrelextrema(prices, np.greater, order=order)[0]
    swing_lows  = argrelextrema(prices, np.less, order=order)[0]

    return swing_highs.tolist(), swing_lows.tolist()

def build_developing_profiles(engine, trades, make):
    developing_profiles = []

    for i in range(len(make) - 1):
        end_idx = make[i+1][2]   # IMPORTANT FIX

        if end_idx == 0:
            developing_profiles.append(None)
            continue

        slice_trades = trades[:end_idx]  # cumulative
        profile_i = engine.build_profile(slice_trades)
        developing_profiles.append(profile_i)

    developing_profiles.append(developing_profiles[-1])
    return developing_profiles


from concurrent.futures import ThreadPoolExecutor    
def download_data(bucket_name, blob_name):
    blob = Blob(blob_name, bucket_name)
    return blob.download_as_text()   




# def build_swing_profiles(engine, trades, df, swing_highs, swing_lows):
#     pivots = sorted(swing_highs + swing_lows)
#     swing_profiles = []

#     for i in range(len(pivots) - 1):

#         start_idx = pivots[i]
#         end_idx   = pivots[i + 1]

#         t_start = df['timestamp'].iloc[start_idx]
#         t_end   = df['timestamp'].iloc[end_idx]

#         swing_low  = df['low'].iloc[start_idx:end_idx+1].min()
#         swing_high = df['high'].iloc[start_idx:end_idx+1].max()

#         # Filter trades for this swing
#         swing_trades = [
#             t for t in trades
#             if (t.raw[2] >= t_start and t.raw[2] <= t_end)
#             and (swing_low <= t.price <= swing_high)
#         ]

#         if not swing_trades:
#             swing_profiles.append(None)
#             continue

#         # ----------------------------------------------------------
#         # Use SAME binning logic as engine.build_profile
#         # ----------------------------------------------------------
#         swing_prices = [t.price for t in swing_trades]
#         p_min = min(swing_prices)

#         hist = {}
#         swing_poc_path = []
#         current_poc = None
#         current_poc_vol = 0.0

#         for t in swing_trades:
#             b = engine._round_to_bin(t.price, p_min)   # <<< key change

#             hist[b] = hist.get(b, 0.0) + t.size

#             # update POC incrementally
#             if hist[b] >= current_poc_vol:
#                 current_poc = b
#                 current_poc_vol = hist[b]

#             swing_poc_path.append(current_poc)

#         # Build final full profile using engine (will use same p_min)
#         profile = engine.build_profile(swing_trades)

#         profile.meta = {
#             "start_idx": start_idx,
#             "end_idx": end_idx,
#             "start_time": df.index[start_idx],
#             "end_time": df.index[end_idx],
#             "swing_low": swing_low,
#             "swing_high": swing_high,
#             "swing_poc_path": swing_poc_path,
#             "swing_poc_final": current_poc
#         }

#         swing_profiles.append(profile)

#     # ----------------------------------------------------------
#     # ADD LAST DEVELOPING SWING (same logic)
#     # ----------------------------------------------------------
#     last_idx = pivots[-1]
#     start_idx = last_idx
#     end_idx   = len(df) - 1

#     t_start = df['timestamp'].iloc[start_idx]
#     t_end   = df['timestamp'].iloc[end_idx]

#     swing_low  = df['low'].iloc[start_idx:end_idx+1].min()
#     swing_high = df['high'].iloc[start_idx:end_idx+1].max()

#     swing_trades = [
#         t for t in trades
#         if (t.raw[2] >= t_start and t.raw[2] <= t_end)
#         and (swing_low <= t.price <= swing_high)
#     ]

#     if swing_trades:

#         swing_prices = [t.price for t in swing_trades]
#         p_min = min(swing_prices)

#         hist = {}
#         swing_poc_path = []
#         current_poc = None
#         current_poc_vol = 0.0

#         for t in swing_trades:
#             b = engine._round_to_bin(t.price, p_min)

#             hist[b] = hist.get(b, 0.0) + t.size

#             if hist[b] >= current_poc_vol:
#                 current_poc = b
#                 current_poc_vol = hist[b]

#             swing_poc_path.append(current_poc)

#         profile = engine.build_profile(swing_trades)

#         profile.meta = {
#             "start_idx": start_idx,
#             "end_idx": end_idx,
#             "start_time": df.index[start_idx],
#             "end_time": df.index[end_idx],
#             "swing_low": swing_low,
#             "swing_high": swing_high,
#             "swing_poc_path": swing_poc_path,
#             "swing_poc_final": current_poc
#         }

#         swing_profiles.append(profile)
#     else:
#         swing_profiles.append(None)

#     return swing_profiles


    
def build_swing_profiles(engine, trades, df, swing_highs, swing_lows):
    pivots = sorted(swing_highs + swing_lows)
    swing_profiles = []

    bin_size = engine.bin_size

    def price_to_bin(price, p_min):
        return p_min + round((price - p_min) / bin_size) * bin_size

    # ----------------------------------------------------
    # Helper: compute swing profile
    # ----------------------------------------------------
    def compute_swing(start_idx, end_idx):
        
        t_start = df['timestamp'].iloc[start_idx]
        t_end   = df['timestamp'].iloc[end_idx]

        swing_low  = df['low'].iloc[start_idx:end_idx+1].min()
        swing_high = df['high'].iloc[start_idx:end_idx+1].max()

        # Select trades in this swing
        swing_trades = [
            t for t in trades
            if (t.raw[2] >= t_start and t.raw[2] <= t_end)
            and (swing_low <= t.price <= swing_high)
        ]
        if not swing_trades:
            return None

        swing_prices = [t.price for t in swing_trades]
        p_min = min(swing_prices)

        hist = {}
        current_poc = None
        current_poc_vol = 0

        swing_poc_path = []
        trade_pos = 0
        n_trades = len(swing_trades)

        for bar_idx in range(start_idx, end_idx + 1):

            if bar_idx < end_idx:
                boundary_time = df['timestamp'].iloc[bar_idx + 1]
            else:
                boundary_time = df['timestamp'].iloc[end_idx]

            while trade_pos < n_trades and swing_trades[trade_pos].raw[2] < boundary_time:
                t = swing_trades[trade_pos]
                trade_pos += 1

                b = price_to_bin(t.price, p_min)
                hist[b] = hist.get(b, 0) + t.size

                if hist[b] >= current_poc_vol:
                    current_poc = b
                    current_poc_vol = hist[b]

            swing_poc_path.append(current_poc)

        profile = engine.build_profile(swing_trades)
        profile.meta = {
            "start_idx": start_idx,
            "end_idx": end_idx,
            "start_time": df.index[start_idx],
            "end_time": df.index[end_idx],
            "swing_low": swing_low,
            "swing_high": swing_high,
            "swing_poc_path": swing_poc_path,
            "swing_poc_final": current_poc
        }

        return profile

    # ----------------------------------------------------
    # CASE 0: NO PIVOTS FOUND
    # ----------------------------------------------------
    if len(pivots) == 0:
        return [ compute_swing(0, len(df) - 1) ]

    # ----------------------------------------------------
    # CASE 1: ONLY ONE PIVOT
    # ----------------------------------------------------
    if len(pivots) == 1:
        start_idx = pivots[0]
        end_idx   = len(df) - 1
        return [ compute_swing(start_idx, end_idx) ]

    # ----------------------------------------------------
    # CASE 2: NORMAL MULTIPLE PIVOTS
    # ----------------------------------------------------
    for i in range(len(pivots) - 1):
        start_idx = pivots[i]
        end_idx   = pivots[i + 1]
        swing_profiles.append(compute_swing(start_idx, end_idx))

    # Final developing swing
    swing_profiles.append(compute_swing(pivots[-1], len(df) - 1))

    return swing_profiles
    
def add_swing_profiles_time_aligned(fig, swing_profiles, df):
    """
    Draw each swing profile:
      • spans FULL swing window horizontally (start_idx → end_idx)
      • uses df.index as x-axis
      • histogram scaled inside whole window
    """

    for profile in swing_profiles:
        if profile is None or profile.meta is None:
            continue

        # pull metadata
        x_left   = profile.meta["start_time"]
        x_right  = profile.meta["end_time"]
        swing_low  = profile.meta["swing_low"]
        swing_high = profile.meta["swing_high"]

        # ------------------------------------------------------------------
        # 1) Background shading
        # ------------------------------------------------------------------
        fig.add_vrect(
            x0=x_left,
            x1=x_right,
            y0=swing_low,
            y1=swing_high,
            fillcolor="rgba(150,150,150,0.06)",
            line_width=0
        )

        # ------------------------------------------------------------------
        # 2) Histogram data
        # ------------------------------------------------------------------
        prices  = np.array(list(profile.hist.keys()))
        volumes = np.array(list(profile.hist.values()))

        sort_idx = np.argsort(prices)
        prices   = prices[sort_idx]
        volumes  = volumes[sort_idx]

        max_vol = volumes.max()
        if max_vol == 0:
            continue

        # normalize 0..1
        vol_norm = volumes / max_vol

        # map volume → x coordinate inside FULL swing window
        x_positions = x_left + (x_right - x_left) * vol_norm

        # ------------------------------------------------------------------
        # 3) Build flat horizontal bars
        # ------------------------------------------------------------------
        x_plot, y_plot = [], []
        for xp, yp in zip(x_positions, prices):
            x_plot.extend([x_left, xp, None])   # horizontal bar
            y_plot.extend([yp, yp, None])

        fig.add_trace(go.Scatter(
            x=x_plot,
            y=y_plot,
            mode="lines",
            line=dict(color="yellow", width=2),
            opacity=0.2,
            showlegend=False
        ))

        # ------------------------------------------------------------------
        # 4) Plot POC inside same window
        # ------------------------------------------------------------------
        poc_vol  = profile.hist[profile.poc]
        poc_norm = poc_vol / max_vol
        poc_x    = x_left + (x_right - x_left) * poc_norm
        
        # --- Buy/Sell Delta ---
        session_buy  = sum(profile.buy_vol.values())
        session_sell = sum(profile.sell_vol.values())
        
        session_delta = session_buy - session_sell
        session_delta_pct = (session_delta / max(session_buy + session_sell, 1)) * 100
        
        # --- Build annotation text ---
        label_text = (
            f"Vol: {poc_vol:.0f}<br>"
            f"Buy%: {profile.poc_buy_pct*100:.1f}%<br>"
            #f"Buy: {session_buy:,}<br>"
            #f"Sell: {session_sell:,}<br>"
            f"Δ: {session_delta:,}<br>"
            f"Δ%: {session_delta_pct:.1f}%"
        )
        
        fig.add_trace(go.Scatter(
            x=[poc_x],
            y=[profile.poc],
            mode="markers",  # markers only, no text drawn
            marker=dict(size=10, color="orange"),
            hovertemplate=(
                "Price: %{y:,.2f}<br>"       # default y value
                + label_text +         # your custom block
                "<extra></extra>"      # hides the trace name in hover
            ),
            showlegend=False
        ))
        
        poc_path = profile.meta["swing_poc_path"]
        times = np.linspace(x_left, x_right, len(poc_path))
        
        fig.add_trace(go.Scatter(
            x=times,
            y=poc_path,
            mode="lines",
            line=dict(color="orange", width=2),
            showlegend=False
        ))
                        
    return fig
def add_swing_profiles_overlaid(fig, swing_profiles, df):
    """
    Overlay volume profiles directly on candlesticks at their price levels
    """
    # Get the full time range of the chart using the actual datetime index
    chart_start = df.index[0]
    chart_end = df.index[-1]
    
    # Calculate time range - handle both datetime and numeric indices
    if isinstance(chart_start, (int, np.integer)):
        # If index is numeric, use it directly for spacing
        total_range = chart_end - chart_start
        profile_width = total_range * 0.08  # 8% of range
    else:
        # If index is datetime
        total_range = (chart_end - chart_start).total_seconds()
        profile_width = total_range * 0.08
    
    for profile in swing_profiles:
        if profile is None:
            continue
        
        # Get swing metadata
        start_idx = profile.meta["start_idx"]
        end_idx = profile.meta["end_idx"]
        x_center = df.index[start_idx]  # Use actual index value
        
        # Histogram data
        prices = np.array(list(profile.hist.keys()))
        volumes = np.array(list(profile.hist.values()))
        sort_idx = np.argsort(prices)
        prices = prices[sort_idx]
        volumes = volumes[sort_idx]
        max_vol = volumes.max()
        
        # Normalize volumes to profile width
        vol_norm = volumes / max_vol
        
        # Create horizontal bars at each price level
        x_plot, y_plot = [], []
        for vol_n, price in zip(vol_norm, prices):
            # Bar extends from center to right based on volume
            if isinstance(x_center, (int, np.integer)):
                x_end = x_center + (profile_width * vol_n)
            else:
                bar_width = pd.Timedelta(seconds=profile_width * vol_n)
                x_end = x_center + bar_width
            
            x_plot.extend([x_center, x_end, None])
            y_plot.extend([price, price, None])
        
        # Draw profile
        fig.add_trace(go.Scatter(
            x=x_plot,
            y=y_plot,
            mode="lines",
            line=dict(color="yellow", width=2),
            opacity=0.7,
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Plot POC
        poc_vol_norm = profile.hist[profile.poc] / max_vol
        if isinstance(x_center, (int, np.integer)):
            poc_x = x_center + (profile_width * poc_vol_norm)
        else:
            poc_bar_width = pd.Timedelta(seconds=profile_width * poc_vol_norm)
            poc_x = x_center + poc_bar_width
        
        fig.add_trace(go.Scatter(
            x=[poc_x],
            y=[profile.poc],
            mode="markers",
            marker=dict(size=8, color="red", symbol="circle"),
            showlegend=False,
            hovertemplate=f"POC: {profile.poc:.2f}<extra></extra>"
        ))
    
    return fig




symbolNumList =  ['42140878', '42002475', '42005850']
symbolNameList = ['ES', 'NQ', 'YM']


intList = [str(i) for i in range(3,70)]



gclient = storage.Client(project="stockapp-401615")
bucket = gclient.get_bucket("stockapp-storage-east1")

#import duckdb
#from google.api_core.exceptions import NotFound
from dash import Dash, dcc, html, Input, Output, callback, State, callback_context
initial_inter = 500000  # Initial interval #210000#250000#80001
subsequent_inter = 180000#300000  # Subsequent interval
app = Dash()
app.title = "EnVisage"
app.layout = html.Div([
    
    dcc.Graph(id='graph', config={'modeBarButtonsToAdd': ['drawline']}),
    dcc.Interval(
        id='interval',
        interval=initial_inter,
        n_intervals=0,
      ),

    

    html.Div([
        html.Div([
            dcc.Input(id='input-on-submit', type='text', className="input-field"),
            html.Button('Submit', id='submit-val', n_clicks=0, className="submit-button"),
            html.Div(id='container-button-basic', children="Enter a symbol from ES, NQ", className="label-text"),
        ], className="sub-container"),
        dcc.Store(id='stkName-value'),

        html.Div([
            dcc.Input(id='input-on-interv', type='text', className="input-field"),
            html.Button('Submit', id='submit-interv', n_clicks=0, className="submit-button"),
            html.Div(id='interv-button-basic', children="Enter interval from 3-30, Default 10 mins", className="label-text"),
        ], className="sub-container"),
        dcc.Store(id='interv-value'),
    ], className="main-container"),

    dcc.Store(id='data-store'),
    dcc.Store(id='previous-interv'),
    dcc.Store(id='previous-stkName'),
    dcc.Store(id='interval-time', data=initial_inter),
    dcc.Store(id='graph-layout'),
])

@callback(
    Output('stkName-value', 'data'),
    Output('container-button-basic', 'children'),
    Input('submit-val', 'n_clicks'),
    State('input-on-submit', 'value'),
    prevent_initial_call=True
)

def update_output(n_clicks, value):
    value = str(value).upper().strip()
    
    if value in symbolNameList:
        print('The input symbol was "{}" '.format(value))
        return str(value).upper(), str(value).upper()
    else:
        return 'The input symbol '+str(value)+" is not accepted please try different symbol from  |'ES', 'NQ'|", 'The input symbol was '+str(value)+" is not accepted please try different symbol  |'ESH4' 'NQH4' 'CLG4' 'GCG4' 'NGG4' 'HGH4' 'YMH4' 'BTCZ3' 'RTYH4'|  "

@callback(
    Output('interv-value', 'data'),
    Output('interv-button-basic', 'children'),
    Input('submit-interv', 'n_clicks'),
    State('input-on-interv', 'value'),
    prevent_initial_call=True
)
def update_interval(n_clicks, value):
    value = str(value)
    
    if value in intList:
        print('The input interval was "{}" '.format(value))
        return str(value), str(value), 
    else:
        return 'The input interval '+str(value)+" is not accepted please try different interval from  |'1' '2' '3' '5' '10' '15'|", 'The input interval '+str(value)+" is not accepted please try different interval from  |'1' '2' '3' '5' '10' '15'|"



@callback(
    [Output('data-store', 'data'),
        Output('graph', 'figure'),
        Output('previous-stkName', 'data'),
        Output('previous-interv', 'data'),
        Output('interval', 'interval'),
        Output('graph-layout', 'data')],
    [Input('interval', 'n_intervals'),
     Input('graph', 'relayoutData')], 
    [State('stkName-value', 'data'),
        State('interv-value', 'data'),
        State('data-store', 'data'),
        State('previous-stkName', 'data'),
        State('previous-interv', 'data'),
        State('interval-time', 'data'),
        State('graph-layout', 'data')],
    prevent_initial_call=False
)

def update_graph_live(n_intervals, relayout_data, sname, interv, stored_data, previous_stkName, previous_interv, interval_time, layout_data): #interv


    symbolNumList =  ['42140878', '42002475', '42005850']
    symbolNameList = ['ES', 'NQ', 'YM']
    
    future_tick = {
        'ES':0.25, 
        'NQ':0.25, 
        'YM':1.00, 
        'BTC':5.00, 
        'CL':0.01, 
        'GC':0.10,
        'PL':0.10,
        'HG':0.0005,
        'SI':0.005} 
    
    
    if sname in symbolNameList:
        stkName = sname
        symbolNum = symbolNumList[symbolNameList.index(stkName)]   
    else:
        stkName = 'NQ' 
        sname = 'NQ'
        symbolNum = symbolNumList[symbolNameList.index(stkName)]
        
    if interv not in intList:
        interv = '5'

    #stkName = 'NQ' 
    #interv = '5'
    #symbolNum = symbolNumList[symbolNameList.index(stkName)]
    stored_data = None
    
    
        
    with ThreadPoolExecutor(max_workers=2) as executor:
        #if sname != previous_stkName:
        # Download everything when stock name changes
        futures = [
            executor.submit(download_data, bucket, 'FuturesOHLC' + str(symbolNum)),
            executor.submit(download_data, bucket, 'FuturesTrades' + str(symbolNum)),]
            #executor.submit(download_daily_data, bucket, stkName)]
        
        FuturesOHLC, FuturesTrades = [future.result() for future in futures] #, prevDf
    
    
    # Process data with pandas directly
    FuturesOHLC = pd.read_csv(io.StringIO(FuturesOHLC), header=None)
    FuturesTrades = pd.read_csv(io.StringIO(FuturesTrades), header=None)
    
    
    aggs = [ ] 
    for row in FuturesOHLC.itertuples(index=False):
        # Extract values from the row, where row[0] corresponds to the first column, row[1] to the second, etc.
        hourss = datetime.fromtimestamp(int(row[0]) // 1000000000).hour
        hourss = f"{hourss:02d}"  # Ensure it's a two-digit string
        minss = datetime.fromtimestamp(int(row[0]) // 1000000000).minute
        minss = f"{minss:02d}"  # Ensure it's a two-digit string
        
        # Construct the time string
        opttimeStamp = f"{hourss}:{minss}:00"
        
        # Append the transformed row data to the aggs list
        aggs.append([
            row[2] / 1e9,  # Convert the value at the third column (open)
            row[3] / 1e9,  # Convert the value at the fourth column (high)
            row[4] / 1e9,  # Convert the value at the fifth column (low)
            row[5] / 1e9,  # Convert the value at the sixth column (close)
            int(row[6]),   # Volume
            opttimeStamp,  # The formatted timestamp
            int(row[0]),   # Original timestamp
            int(row[1])    # Additional identifier or name
        ])
            
       
    df = pd.DataFrame(aggs, columns = ['open', 'high', 'low', 'close', 'volume', 'time', 'timestamp', 'name',])
    
    df['strTime'] = df['timestamp'].apply(lambda x: pd.Timestamp(int(x) // 10**9, unit='s', tz='EST') )
    
    df.set_index('strTime', inplace=True)
    df['volume'] = pd.to_numeric(df['volume'], downcast='integer')
    df_resampled = df.resample(interv+'min').agg({
        'timestamp': 'first',
        'name': 'last',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'time': 'first',
        'volume': 'sum'
    })
    
    df_resampled.reset_index(drop=True, inplace=True)
    
    df = df_resampled
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True) 
    
    
    blob = Blob('PrevDay', bucket) 
    PrevDay = blob.download_as_text()
        

    csv_reader  = csv.reader(io.StringIO(PrevDay))

    csv_rows = []
    for row in csv_reader:
        csv_rows.append(row)
        
    # try:
    #     df['PreviousDayLVA'] = csv_rows[[i[4] for i in csv_rows].index(symbolNum)][0]
    #     df['PreviousDayHVA'] = csv_rows[[i[4] for i in csv_rows].index(symbolNum)][1]
    #     df['PreviousDayPOC'] = csv_rows[[i[4] for i in csv_rows].index(symbolNum)][2]
    #     # previousDay = [csv_rows[[i[4] for i in csv_rows].index(symbolNum)][0], 
    #     #                 csv_rows[[i[4] for i in csv_rows].index(symbolNum)][1], 
    #     #                 csv_rows[[i[4] for i in csv_rows].index(symbolNum)][2],
    #     #                 ]
    # except(ValueError):
    #      pass
    
    
    
    AllTrades = []
    for row in FuturesTrades.itertuples(index=False):
        hourss = datetime.fromtimestamp(int(row[0]) // 1000000000).hour
        hourss = f"{hourss:02d}"  # Ensure two-digit format for hours
        minss = datetime.fromtimestamp(int(row[0]) // 1000000000).minute
        minss = f"{minss:02d}"  # Ensure two-digit format for minutes
        opttimeStamp = f"{hourss}:{minss}:00"
        AllTrades.append([int(row[1]) / 1e9, int(row[2]), int(row[0]), 0, row[3], opttimeStamp])
    
    
    dtime = df['time'].dropna().values.tolist()
    dtimeEpoch = df['timestamp'].dropna().values.tolist()
    
    
    #tempTrades = [i for i in AllTrades]
    tempTrades = sorted(AllTrades, key=lambda d: d[2], reverse=False) 
    #tradeTimes = [i[6] for i in AllTrades]
    tradeEpoch = [i[2] for i in AllTrades]
    
    
    make = []
    for ttm in range(len(dtimeEpoch)):
        
        make.append([dtimeEpoch[ttm],dtime[ttm],bisect.bisect_left(tradeEpoch, dtimeEpoch[ttm])]) #min(range(len(tradeEpoch)), key=lambda i: abs(tradeEpoch[i] - dtimeEpoch[ttm]))
    
    
    # def build_developing_profiles(engine: NodeBreachEngine, trades: List[Trade], make_list):
    #     """
    #     make_list format:  [timestamp, 'HH:MM:SS', end_index]
    #     Returns a list of NodeProfile objects, one per candle.
    #     """
    #     developing_profiles = []
    
    #     for i, entry in enumerate(make_list):
    #         end_idx = entry[2]   # how many trades belong up to this candle
            
    #         if end_idx == 0:
    #             # No trades yet — append None or empty profile
    #             developing_profiles.append(None)
    #             continue
    
    #         # Use trades from start to end_idx
    #         trades_up_to_now = trades[:end_idx]
    
    #         # Build the profile
    #         profile_i = engine.build_profile(trades_up_to_now)
    #         developing_profiles.append(profile_i)
    
    #     return developing_profiles
    
    
    
    
    
    trades = parse_raw_trades(AllTrades)
    engine = NodeBreachEngine(bin_size=future_tick[stkName], value_area_pct=0.70, compute_vwap=True)
    developing_profiles = build_developing_profiles(engine, trades, make)
    profile = engine.build_profile(trades)
    
    #plot_profile(profile, title="NQ Volume Profile")
    # print("POC:", profile.poc)
    # print("VAL/VAH:", profile.val, profile.vah)
    # print("PoV:", profile.pov)
    # print("Node strength:", profile.node_strength)
    # print("POC Buy %:", profile.poc_buy_pct)
    
    # 4. ATR (optional)
    #candles = df_to_candles(df, interv+'min')
    # atr_vals = compute_atr(candles, period=14)
    # signals = []
    
    # for i, candle in enumerate(candles):
    #     profile = developing_profiles[i]
    #     if profile is None:
    #         continue
    
    #     # ATR optional
    #     atr_val = atr_vals[i] if i < len(atr_vals) else None
    
    #     candle_signals = engine.detect_breaches(
    #         [candle],      # a single candle list
    #         profile,
    #         atr_values=[atr_val],
    #         atr_mult=0.0,
    #         wick_threshold=0.03,
    #         margin_pct=0.0005
    #     )
    
    #     signals.extend(candle_signals)
    
    
    
    
    # # # 5. Detect breaches
    # # signals = engine.detect_breaches(
    # #     candles,
    # #     profile,
    # #     atr_values=atr_vals,
    # #     atr_mult=0.0,       # set >0 to turn ATR guard on
    # #     wick_threshold=0.0, # set ~0.3 if you want wick-based confirmation
    # #     margin_pct=0.0005,  # small price margin around POC/PoV, e.g. 0.05%
    # # )
    
    # for sig in signals:
    #      print(sig.signal_type, "at", sig.level_price, "on", sig.time)
         
    
    previous_stkName = sname
    previous_interv = interv    
    
    
    poc_path = [p.poc if p else None for p in developing_profiles]
    pov_path = [p.pov if p else None for p in developing_profiles]
    vah_path = [p.vah if p else None for p in developing_profiles]
    val_path = [p.val if p else None for p in developing_profiles]
    
    
    
    swing_highs, swing_lows = detect_swings(df, order=20)
    swing_profiles = build_swing_profiles(engine, trades, df, swing_highs, swing_lows)
    fig = plot_nbe_combined(
        df=df,
        profile=profile,          # full-session or selected window profile
        poc_path=poc_path,
        pov_path=pov_path,
        vah_path=vah_path,
        val_path=val_path,
        title=stkName +' ' +str(datetime.now().time())
    )
    #fig = ensure_overlay_axes(fig, len(swing_profiles))
    fig = add_swing_profiles_time_aligned(fig, swing_profiles, df)
    fig.update_layout(
        newshape=dict(
            line_color="white",       # <— THIS makes drawline output white
            line_width=2,
            fillcolor="rgba(0,0,0,0)"  # keeps shapes hollow
        )
    )
    #fig.show(config={'modeBarButtonsToAdd': ['drawline']})
    
    
    ctx = callback_context
    if ctx.triggered and ctx.triggered[0]['prop_id'] == 'graph.relayoutData':
        # Only update the layout_data when the user interacts with the graph
        if relayout_data and ('xaxis.range[0]' in relayout_data or 'yaxis.range[0]' in relayout_data):
            layout_data = relayout_data
    
    # Apply stored layout to new figure
    if layout_data:
        # Apply x-axis range if available
        if 'xaxis.range[0]' in layout_data and 'xaxis.range[1]' in layout_data:
            fig.update_layout(xaxis_range=[layout_data['xaxis.range[0]'], layout_data['xaxis.range[1]']])
        
        # Apply y-axis range if available
        if 'yaxis.range[0]' in layout_data and 'yaxis.range[1]' in layout_data:
            fig.update_layout(yaxis_range=[layout_data['yaxis.range[0]'], layout_data['yaxis.range[1]']])
            
    #fig.show(config={'modeBarButtonsToAdd': ['drawline']})
    

           
    if interval_time == initial_inter:
        interval_time = subsequent_inter
        
    if stkName != previous_stkName  or interv != previous_interv:
        interval_time = initial_inter


    return stored_data, fig, previous_stkName, previous_interv, interval_time, relayout_data

if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=8080)
    #app.run_server(debug=False, use_reloader=False) 
