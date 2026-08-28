from pathlib import Path
import pandas as pd
import yfinance as yf
from datetime import timedelta
from typing import Optional, cast

class StockCache:
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = Path(cache_dir or Path.home() / ".cache" / "stock_data")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, ticker: str, start: Optional[str], end: Optional[str], interval: str = "1d") -> Path:
        # Normalize None -> "None" to make filename deterministic
        s = start if start is not None else "None"
        e = end if end is not None else "None"
        fname = f"{ticker}__s-{s}__e-{e}__int-{interval}.parquet"
        # sanitize file name for safety if needed
        return self.cache_dir / fname

    def _atomic_write_parquet(self, df: pd.DataFrame, path: Path):
        tmp = path.with_suffix(path.suffix + ".tmp")
        df.to_parquet(tmp, index=True)   # parquet is fast & compressed
        tmp.replace(path)                # atomic on most platforms

    @staticmethod
    def _ensure_frame(
        data: pd.DataFrame | pd.Series | None,
        ticker: str,
    ) -> pd.DataFrame:
        if data is None:
            raise RuntimeError(f"No data returned from yfinance for {ticker}")
        if isinstance(data, pd.Series):
            return data.to_frame()
        return data

    def get_prices(self,
                   ticker: str,
                   start: Optional[str] = None,
                   end: Optional[str] = None,
                   interval: str = "1d",
                   auto_adjust: bool = True,
                   force_refresh: bool = False,
                   incremental: bool = True) -> pd.DataFrame:
        """
        Return DataFrame of prices for ticker between start and end (strings YYYY-MM-DD or None).
        If cached and not force_refresh: load from cache.
        If incremental and cache exists: download only missing tail and append.
        """
        path = self._cache_path(ticker, start, end, interval)
        if path.exists() and (not force_refresh):
            try:
                df = self._ensure_frame(pd.read_parquet(path), ticker)
                if df.empty:
                    raise ValueError(f"Cached price data for {ticker} is empty")
                # ensure index is DatetimeIndex
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                # if user requested an explicit start/end that differ from cached file,
                # we can still slice the DataFrame to the requested window:
                if start is not None or end is not None:
                    s = pd.to_datetime(start) if start is not None else None
                    e = pd.to_datetime(end) if end is not None else None
                    return df.loc[s:e]
                return df
            except Exception:
                # if reading cache fails, fallback to re-download
                pass

        # If no cache or force_refresh, try to download (or incremental update)
        if path.exists() and incremental and (not force_refresh):
            # attempt incremental update
            try:
                df_old = self._ensure_frame(pd.read_parquet(path), ticker)
                if df_old.empty:
                    raise ValueError(f"Cached price data for {ticker} is empty")
                if not isinstance(df_old.index, pd.DatetimeIndex):
                    df_old.index = pd.to_datetime(df_old.index)
                last_date = cast(
                    pd.Timestamp,
                    pd.Timestamp(str(df_old.index.max())),
                ).normalize()
                # fetch from next day onwards
                new_start = (
                    last_date.to_pydatetime() + timedelta(days=1)
                ).strftime("%Y-%m-%d")
                # don't request end earlier than new_start
                actual_end = end
                if start is not None:
                    # if user requested a start later than cached range, full refresh
                    if pd.Timestamp(start) > last_date:
                        df_new = self._ensure_frame(
                            yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=auto_adjust),
                            ticker,
                        )
                        df_all = df_new
                    else:
                        df_new = self._ensure_frame(
                            yf.download(ticker, start=new_start, end=actual_end, interval=interval, auto_adjust=auto_adjust),
                            ticker,
                        )
                        if df_new.empty:
                            df_all = df_old
                        else:
                            # align columns and append (avoid duplicates)
                            df_new.index = pd.to_datetime(df_new.index)
                            df_combined = pd.concat([df_old, df_new])
                            df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
                            df_all = df_combined
                else:
                    # no user start specified -> append new rows
                    df_new = self._ensure_frame(
                        yf.download(ticker, start=new_start, end=actual_end, interval=interval, auto_adjust=auto_adjust),
                        ticker,
                    )
                    if df_new.empty:
                        df_all = df_old
                    else:
                        df_new.index = pd.to_datetime(df_new.index)
                        df_combined = pd.concat([df_old, df_new])
                        df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
                        df_all = df_combined

                # pandas' concat stubs also allow Series, but every branch above
                # starts from and preserves DataFrames.
                df_all = cast(pd.DataFrame, df_all)
                self._atomic_write_parquet(df_all, path)
                # return requested slice if user requested window
                if start is not None or end is not None:
                    s = pd.to_datetime(start) if start is not None else None
                    e = pd.to_datetime(end) if end is not None else None
                    return df_all.loc[s:e]
                return df_all
            except Exception:
                # fallback: full download below
                pass

        # Full download (either no cache or fallback)
        df = self._ensure_frame(
            yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=auto_adjust),
            ticker,
        )
        if df.empty:
            raise RuntimeError(
                f"No price data returned from yfinance for {ticker}; "
                "the empty response was not cached"
            )
        # ensure datetime index and timezone
        datetime_index = pd.DatetimeIndex(pd.to_datetime(df.index))
        df.index = datetime_index
        # localize tz only if naive index and you want US/Eastern (optional)
        if datetime_index.tz is None:
            # choose the timezone you prefer; often better to keep tz-naive or convert to UTC:
            # df.index = df.index.tz_localize("US/Eastern")
            pass

        # write to cache
        try:
            self._atomic_write_parquet(df, path)
        except Exception:
            # ignore caching errors but still return dataframe
            pass
        return df
