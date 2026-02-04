# engine/features.py
import numpy as np
import pandas as pd

MS_1M = 60_000

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _macd_features(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Compute MACD, signal, histogram (TradingView defaults: 12,26,9)."""
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    macd = ema_fast - ema_slow
    sig = _ema(macd, signal)
    hist = macd - sig
    return pd.DataFrame({"macd": macd, "signal": sig, "hist": hist})


def _infer_bar_ms(df: pd.DataFrame) -> float:
    if "ts" not in df.columns:
        return float("nan")

    ts = (
        pd.to_numeric(df["ts"], errors="coerce")
        .dropna()
        .astype(np.int64)
        .to_numpy()
    )
    if len(ts) < 3:
        return float("nan")

    diffs = np.diff(ts)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return float("nan")

    return float(np.median(diffs))

def _is_1m_bars(df: pd.DataFrame) -> bool:
    bar_ms = _infer_bar_ms(df)
    if not np.isfinite(bar_ms):
        return False
    return abs(bar_ms - MS_1M) <= 2_000 #tolerate small gaps/jitter

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"].shift(1)
    
    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Wilder-style ADX (trend strength).
    Returns ADX in [0, 100] (with NaNs during warmup).
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_dm_sm = plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    minus_dm_sm = minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    plus_di = 100.0 * (plus_dm_sm / atr.replace(0, np.nan))
    minus_di = 100.0 * (minus_dm_sm / atr.replace(0, np.nan))

    dx = (
        100.0
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )
    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return adx


def _bb_features(series: pd.Series, period: int = 20) -> pd.DataFrame:
    """
    Returns DataFrame with:
      - bb_width: (upper-lower)/mid
      - bb_z: (close-mid)/std
    """
    mid = series.rolling(period, min_periods=period).mean()
    std = series.rolling(period, min_periods=period).std(ddof=0)
    upper = mid + 2.0 * std
    lower = mid - 2.0 * std

    width = (upper - lower) / mid.replace(0, np.nan)
    z = (series - mid) / std.replace(0, np.nan)
    return pd.DataFrame({"bb_width": width, "bb_z": z})


def _donchian_features(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Donchian channel features:
      - donch_hi: rolling high max
      - donch_lo: rolling low min
      - donch_pos: (close - lo) / (hi - lo) in [0,1]
    """
    hi = df["high"].rolling(period, min_periods=period).max()
    lo = df["low"].rolling(period, min_periods=period).min()
    rng = (hi - lo).replace(0, np.nan)
    pos = (df["close"] - lo) / rng
    return pd.DataFrame({"donch_hi": hi, "donch_lo": lo, "donch_pos": pos})


def _htf_ohlcv(df_1m: pd.DataFrame, tf_ms: int) -> pd.DataFrame:
    """
    Build higher-timeframe OHLCV from 1m bars using integer time buckets.

    Important: 'ts' is assumed to be the 1m candle open time in ms.
    Bucket key = ts // tf_ms.
    """
    ts = pd.to_numeric(df_1m["ts"], errors="coerce").astype("Int64")
    if ts.isna().any():
        raise ValueError("ts contains NaNs; cannot build HTF candles")
    k = (ts.astype(np.int64) // int(tf_ms)).astype(np.int64)
    # Ensure the group key has a stable column name after reset_index().
    k = pd.Series(k.to_numpy(dtype=np.int64), index=df_1m.index, name="k")

    # Group by bucket; df is already sorted by ts in pipeline.
    g = df_1m.groupby(k, sort=False)

    out = g.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum") if "volume" in df_1m.columns else ("close", "size"),
    ).reset_index()

    out["k"] = pd.to_numeric(out["k"], errors="coerce").astype(np.int64)
    return out


def _align_htf_features_to_1m(
    *,
    df_1m: pd.DataFrame,
    k_1m: np.ndarray,
    df_htf: pd.DataFrame,
    k_col: str,
    feat_cols: list,
    suffix: str,
) -> pd.DataFrame:
    """
    Align HTF feature columns back to 1m rows by key lookup.
    Expects df_htf features already shifted by 1 HTF bar (no lookahead).
    """
    out = df_1m
    idx = pd.Index(df_htf[k_col].to_numpy(dtype=np.int64)).get_indexer(k_1m)

    for c in feat_cols:
        vals = df_htf[c].to_numpy(dtype=np.float64)
        arr = np.full(len(k_1m), np.nan, dtype=np.float64)
        good = idx >= 0
        arr[good] = vals[idx[good]]
        out[f"{c}{suffix}"] = arr

    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # 1. Trend (EMAs)
    feat_cols = []
    for p in [10, 20, 50, 100, 200]:
        
        col = f"ema_{p}"
        df[col] = _ema(df["close"], p)
        feat_cols.append(col)
        
    # 2. Momentum (RSI)
    df["rsi_14"] = _rsi(df["close"], 14)
    
    feat_cols.append("rsi_14")

    # 3. Volatility (ATR)
    df["atr_14"] = _atr(df, 14)
    df["atr_pct"] = (df["atr_14"] / df["close"]) * 100.0
    
    feat_cols.extend(["atr_14", "atr_pct"])

    # --- TA Pack (base timeframe) ---
    # These are "TradingView classics" that users expect on daily bars.
    try:
        macd_df = _macd_features(df["close"], fast=12, slow=26, signal=9)
        df["macd_12_26"] = macd_df["macd"]
        df["macd_signal_12_26_9"] = macd_df["signal"]
        df["macd_hist_12_26_9"] = macd_df["hist"]
        feat_cols.extend(["macd_12_26", "macd_signal_12_26_9", "macd_hist_12_26_9"])
    except Exception:
        pass

    # Only compute these on non-1m datasets (1m gets HTF versions later).
    if not _is_1m_bars(df):
        try:
            df["adx_14"] = _adx(df, 14)
            feat_cols.append("adx_14")
        except Exception:
            pass

        try:
            bb = _bb_features(df["close"], 20)
            df["bb_width_20"] = bb["bb_width"]
            df["bb_z_20"] = bb["bb_z"]
            feat_cols.extend(["bb_width_20", "bb_z_20"])
        except Exception:
            pass

        try:
            donch = _donchian_features(df, 20)
            df["donch_hi_20"] = donch["donch_hi"]
            df["donch_lo_20"] = donch["donch_lo"]
            df["donch_pos_20"] = donch["donch_pos"]
            feat_cols.extend(["donch_hi_20", "donch_lo_20", "donch_pos_20"])
        except Exception:
            pass


    # 4. Volume Flow
    if "volume" in df.columns:
        vol_ma = df["volume"].rolling(50).mean()
        df["rvol_50"] = df["volume"] / vol_ma.replace(0, np.nan)
        df["rvol_50"] = df["rvol_50"].fillna(0.0)
    else:
        df["rvol_50"] = 1.0

    feat_cols.append("rvol_50")

    # ============================================================
    # 5) Multi-timeframe (15m / 1h) context features (no lookahead)
    # ============================================================
    # We compute HTF candles from 1m, compute features on HTF candles,
    # then SHIFT by 1 HTF bar and align back to 1m by key lookup.
    #
    # This ensures the 1m bar only sees completed HTF information.
    if "ts" in df.columns and _is_1m_bars(df):
        ts = pd.to_numeric(df["ts"], errors="coerce").astype("Int64")
        if ts.isna().any():
            raise ValueError("ts has NaNs; cannot compute MTF features safely")
        ts_i64 = ts.astype(np.int64).to_numpy()

        tf_15m = 15 * 60 * 1000
        tf_1h = 60 * 60 * 1000
        k15 = (ts_i64 // tf_15m).astype(np.int64)
        k60 = (ts_i64 // tf_1h).astype(np.int64)

        # Build HTF candles
        df15 = _htf_ohlcv(df, tf_15m)
        df60 = _htf_ohlcv(df, tf_1h)

        # Compute HTF features (context only)
        def add_htf_feats(dfx: pd.DataFrame) -> pd.DataFrame:
            close = dfx["close"].astype(float)

            dfx["ema_50"] = _ema(close, 50)
            dfx["ema_200"] = _ema(close, 200)
            dfx["rsi_14"] = _rsi(close, 14)
            dfx["atr_14"] = _atr(dfx.rename(columns={"open": "open"}), 14)
            dfx["atr_pct"] = (dfx["atr_14"] / close) * 100.0

            dfx["adx_14"] = _adx(dfx, 14)

            bb = _bb_features(close, 20)
            dfx["bb_width_20"] = bb["bb_width"]
            dfx["bb_z_20"] = bb["bb_z"]

            return dfx
        df15 = add_htf_feats(df15)
        df60 = add_htf_feats(df60)

        # Donchian on 15m only (structure): compute periods as a family
        don_periods = [20, 55]
        don_cols: list = []
        for p in don_periods:
            don = _donchian_features(df15, p)
            df15[f"donch_hi_{p}"] = don["donch_hi"]
            df15[f"donch_lo_{p}"] = don["donch_lo"]
            df15[f"donch_pos_{p}"] = don["donch_pos"]
            don_cols.extend([f"donch_pos_{p}", f"donch_hi_{p}", f"donch_lo_{p}"])

        # Shift HTF features by 1 HTF candle to avoid lookahead
        htf15_cols = [
            "ema_50",
            "ema_200",
            "rsi_14",
            "atr_pct",
            "adx_14",
            "bb_width_20",
            "bb_z_20",
        ] + don_cols

        htf60_cols = [
            "ema_50",
            "ema_200",
            "rsi_14",
            "atr_pct",
            "adx_14",
            "bb_width_20",
            "bb_z_20",
        ]

        df15[htf15_cols] = df15[htf15_cols].shift(1)
        df60[htf60_cols] = df60[htf60_cols].shift(1)
        # Align back to 1m
        df = _align_htf_features_to_1m(
            df_1m=df,
            k_1m=k15,
            df_htf=df15,
            k_col="k",
            feat_cols=htf15_cols,
            suffix="_15m",
        )
        df = _align_htf_features_to_1m(
            df_1m=df,
            k_1m=k60,
            df_htf=df60,
            k_col="k",
            feat_cols=htf60_cols,
            suffix="_1h",
        )

        # Hard gate booleans (on 1m timeline, derived from aligned 1h EMAs)
        ema50_1h = pd.to_numeric(df["ema_50_1h"], errors="coerce")
        ema200_1h = pd.to_numeric(df["ema_200_1h"], errors="coerce")

        both = ema50_1h.notna() & ema200_1h.notna()
        df["trend_up_1h"] = np.where(
            both.to_numpy(),
            (ema50_1h > ema200_1h).to_numpy(dtype=bool).astype(float),
            np.nan,
        )
        df["trend_down_1h"] = np.where(
            both.to_numpy(),
            (ema50_1h < ema200_1h).to_numpy(dtype=bool).astype(float),
            np.nan,
        )

        feat_cols.extend(
            [
                "ema_50_15m",
                "ema_200_15m",
                "rsi_14_15m",
                "atr_pct_15m",
                "adx_14_15m",
                "bb_width_20_15m",
                "bb_z_20_15m",
                # donch family (15m)
                "donch_pos_20_15m",
                "donch_hi_20_15m",
                "donch_lo_20_15m",
                "donch_pos_55_15m",
                "donch_hi_55_15m",
                "donch_lo_55_15m",
                "ema_50_1h",
                "ema_200_1h",
                "rsi_14_1h",
                "atr_pct_1h",
                "adx_14_1h",
                "bb_width_20_1h",
                "bb_z_20_1h",
                "trend_up_1h",
                "trend_down_1h",
            ]
        )

    # IMPORTANT:
    # - No backfill (lookahead).
    # - Do NOT forward-fill the entire df (would smear event-like columns such as
    #   funding_rate across time). Only forward-fill the features we created.
    #
    # Warmup NaNs remain NaN until the indicator becomes defined.
    df[feat_cols] = df[feat_cols].ffill()

    return df