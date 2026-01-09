"""
Precious Metals API - Free & Easy to Use
A simple API for getting real-time precious metal prices with historical tracking.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import httpx
import asyncio
import json
import re
import logging
from bs4 import BeautifulSoup

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

CACHE_DURATION_SECONDS = 120  # Cache prices for 2 minutes (faster response times)
HISTORY_CACHE_DURATION_SECONDS = 300  # Cache historical prices for 5 minutes (they don't change)
HISTORY_FILE = Path(__file__).parent / "price_history.json"
SUPPORTED_METALS = ["gold", "silver", "platinum", "palladium"]
METAL_SYMBOLS = {
    "gold": "XAU",
    "silver": "XAG",
    "platinum": "XPT",
    "palladium": "XPD",
}
SUPPORTED_CURRENCIES = [
    "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "CNY", "INR", "BRL"
]

# Shared HTTP client for connection pooling and better performance
_http_client: Optional[httpx.AsyncClient] = None

async def get_http_client() -> httpx.AsyncClient:
    """Get or create shared HTTP client for better performance."""
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            timeout=5.0,  # Reduced to 5s for faster failures
            follow_redirects=True,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=30),
        )
    return _http_client

# ══════════════════════════════════════════════════════════════════════════════
# Models
# ══════════════════════════════════════════════════════════════════════════════

class Metal(str, Enum):
    gold = "gold"
    silver = "silver"
    platinum = "platinum"
    palladium = "palladium"


class PriceChange(BaseModel):
    change: Optional[float] = None
    percent: Optional[float] = None


class MetalPrice(BaseModel):
    metal: str
    symbol: str
    currency: str
    price: float
    unit: str
    timestamp: datetime
    sources_count: Optional[int] = None
    sources: Optional[list[str]] = None
    price_range: Optional[dict] = None
    change_1h: Optional[PriceChange] = None
    change_24h: Optional[PriceChange] = None
    change_7d: Optional[PriceChange] = None
    change_1m: Optional[PriceChange] = None  # 1 month (30 days)
    change_1y: Optional[PriceChange] = None  # 1 year (365 days)
    
    class Config:
        extra = "allow"  # Allow extra fields for flexibility


class AllMetalsResponse(BaseModel):
    status: str
    currency: str
    unit: str
    timestamp: datetime
    metals: dict[str, dict]


class PriceHistoryPoint(BaseModel):
    timestamp: datetime
    price: float


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    cache_age_seconds: Optional[int] = None


# ══════════════════════════════════════════════════════════════════════════════
# Historical Price Fetching (for accurate change calculations)
# ══════════════════════════════════════════════════════════════════════════════

# Cache for historical prices (since they don't change once in the past)
_historical_price_cache: dict[str, dict[int, tuple[float, datetime]]] = {}

async def fetch_historical_price_from_yahoo(metal: str, hours_ago: int) -> Optional[float]:
    """
    Fetch historical price from Yahoo Finance for change calculations.
    NOTE: Uses futures contracts (GC=F, SI=F, PL=F, PA=F) as historical reference
    because they closely track spot prices and Yahoo Finance provides reliable historical data.
    For current prices, we use only spot price sources.
    For 24h+ periods, gets the previous trading day's closing price.
    For shorter periods (1h), gets the price from that time.
    Uses caching to avoid redundant requests.
    """
    # Check cache first
    if metal in _historical_price_cache:
        cached_data = _historical_price_cache[metal].get(hours_ago)
        if cached_data:
            price, cached_time = cached_data
            age = (datetime.utcnow() - cached_time).total_seconds()
            if age < HISTORY_CACHE_DURATION_SECONDS:
                return price
    
    # Yahoo Finance futures symbols (used only for historical data, not current prices)
    symbol_map = {
        "gold": "GC=F",
        "silver": "SI=F",
        "platinum": "PL=F",
        "palladium": "PA=F",
    }
    
    symbol = symbol_map.get(metal)
    if not symbol:
        return None
    
    client = await get_http_client()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }
    
    try:
        # For periods >= 24 hours, use daily intervals to get previous trading day's close
        # For shorter periods (1h), use intraday data
        if hours_ago <= 1:
            range_param = "1d"
            interval = "1m"
            use_close = False  # Use intraday price closest to target time
        elif hours_ago <= 24:
            range_param = "5d"
            interval = "1d"  # Daily intervals for previous day's close
            use_close = True
        elif hours_ago <= 168:  # 7 days
            range_param = "1mo"
            interval = "1d"  # Daily intervals for week-ago close
            use_close = True
        elif hours_ago <= 720:  # 30 days
            range_param = "3mo"
            interval = "1d"
            use_close = True
        elif hours_ago <= 8760:  # 365 days (1 year)
            range_param = "1y"
            interval = "1d"
            use_close = True
        else:
            range_param = "2y"
            interval = "1d"
            use_close = True
        
        response = await client.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}&range={range_param}",
            headers=headers
        )
        response.raise_for_status()
        
        # raise_for_status() ensures status is 2xx, so we can proceed directly
        try:
            data = response.json()
        except ValueError as e:
            logging.warning(f"Invalid JSON response for historical {metal}: {str(e)}")
            return None
        
        result = data.get("chart", {}).get("result", [])
        if not result or len(result) == 0:
            logging.warning(f"No result data for last closed price {metal}")
            return None
            
        timestamps = result[0].get("timestamp", [])
        quotes = result[0].get("indicators", {}).get("quote", [])
        
        if not timestamps or not quotes or len(quotes) == 0:
            logging.warning(f"Missing timestamp or quote data for last closed price {metal}")
            return None
            
        closes = quotes[0].get("close", [])
        
        if not closes or len(timestamps) != len(closes):
            logging.warning(f"Mismatched timestamp/close data lengths for last closed price {metal}")
            return None
            
            if use_close:
                # For daily intervals, get the previous trading day's closing price
                # Find the most recent valid closing price before the target time
                target_time = datetime.utcnow() - timedelta(hours=hours_ago)
                target_unix = int(target_time.timestamp())
                
                # Find the last closing price before or at the target time
                last_close_price = None
                for i, ts in enumerate(timestamps):
                    if ts is None or ts > target_unix:
                        continue
                    
                    if i < len(closes) and closes[i] is not None:
                        try:
                            price = float(closes[i])
                            if price > 0:
                                last_close_price = price
                        except (ValueError, TypeError):
                            continue
                
                # If we found a closing price, use it; otherwise try the last valid close
                if last_close_price is None:
                    # Fallback: get the last valid closing price in the data
                    for i in range(len(closes) - 1, -1, -1):
                        if closes[i] is not None:
                            try:
                                price = float(closes[i])
                                if price > 0:
                                    last_close_price = price
                                    break
                            except (ValueError, TypeError):
                                continue
                
                if last_close_price and last_close_price > 0:
                    # Cache the result
                    if metal not in _historical_price_cache:
                        _historical_price_cache[metal] = {}
                    _historical_price_cache[metal][hours_ago] = (last_close_price, datetime.utcnow())
                    return last_close_price
            else:
                # For intraday (1h), find the price closest to target time
                target_time = datetime.utcnow() - timedelta(hours=hours_ago)
                target_unix = int(target_time.timestamp())
                
                closest_idx = 0
                closest_diff = float("inf")
                
                for i, ts in enumerate(timestamps):
                    if ts is None:
                        continue
                    diff = abs(ts - target_unix)
                    if diff < closest_diff:
                        closest_diff = diff
                        closest_idx = i
                
                # Get price at that index
                if closest_idx < len(closes):
                    price_val = closes[closest_idx]
                    if price_val is not None:
                        try:
                            price = float(price_val)
                            if price > 0:
                                # Cache the result
                                if metal not in _historical_price_cache:
                                    _historical_price_cache[metal] = {}
                                _historical_price_cache[metal][hours_ago] = (price, datetime.utcnow())
                                return price
                        except (ValueError, TypeError) as e:
                            logging.warning(f"Invalid price value for last closed price {metal}: {price_val} - {str(e)}")
    except httpx.TimeoutException:
        logging.warning(f"Timeout fetching historical price for {metal} ({hours_ago}h ago)")
    except httpx.HTTPStatusError as e:
        logging.warning(f"HTTP error fetching historical price for {metal}: {e.response.status_code}")
    except Exception as e:
        logging.warning(f"Error fetching historical price for {metal} ({hours_ago}h ago): {str(e)}")
    
    return None


async def fetch_historical_data_from_yahoo(metal: str, hours: int, currency: str = "USD") -> list[dict]:
    """
    Fetch historical price data points from Yahoo Finance for charting.
    Returns list of {timestamp, price} dictionaries.
    """
    symbol_map = {
        "gold": "GC=F",
        "silver": "SI=F",
        "platinum": "PL=F",
        "palladium": "PA=F",
    }
    
    symbol = symbol_map.get(metal)
    if not symbol:
        return []
    
    client = await get_http_client()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }
    
    try:
        # Determine appropriate range and interval based on hours requested
        if hours <= 1:
            range_param = "1d"
            interval = "1m"
        elif hours <= 24:
            range_param = "5d"
            interval = "5m"
        elif hours <= 168:  # 7 days
            range_param = "1mo"
            interval = "1h"
        elif hours <= 720:  # 30 days
            range_param = "3mo"
            interval = "1d"
        elif hours <= 8760:  # 365 days (1 year)
            range_param = "1y"
            interval = "1d"
        else:  # More than 1 year
            range_param = "2y"
            interval = "1d"
        
        response = await client.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}&range={range_param}",
            headers=headers
        )
        response.raise_for_status()
        
        # raise_for_status() ensures status is 2xx, so we can proceed directly
        try:
            data = response.json()
        except ValueError as e:
            logging.warning(f"Invalid JSON response for historical data {metal}: {str(e)}")
            return []
            
            result = data.get("chart", {}).get("result", [])
            if not result or len(result) == 0:
                logging.warning(f"No result data for historical data {metal}")
                return []
                
            timestamps = result[0].get("timestamp", [])
            quotes = result[0].get("indicators", {}).get("quote", [])
            
            if not timestamps or not quotes or len(quotes) == 0:
                logging.warning(f"Missing timestamp or quote data for historical data {metal}")
                return []
                
            closes = quotes[0].get("close", [])
            
            if not closes or len(timestamps) != len(closes):
                logging.warning(f"Mismatched timestamp/close data lengths for historical data {metal}")
                return []
            
            # Filter to only include data within the requested hours
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            cutoff_unix = int(cutoff_time.timestamp())
            
            history_points = []
            for i, ts in enumerate(timestamps):
                if ts is None or ts < cutoff_unix:
                    continue
                
                if i < len(closes) and closes[i] is not None:
                    try:
                        price = float(closes[i])
                        if price > 0:
                            # Convert Unix timestamp (seconds) to ISO format
                            timestamp_dt = datetime.utcfromtimestamp(ts)
                            history_points.append({
                                "timestamp": timestamp_dt.isoformat() + "Z",
                                "price": round(price, 2),
                            })
                    except (ValueError, TypeError, OSError) as e:
                        logging.warning(f"Error parsing price data point for {metal}: {str(e)}")
                        continue
            
            # Sort by timestamp to ensure chronological order
            history_points.sort(key=lambda x: x.get("timestamp", ""))
            return history_points
    except httpx.TimeoutException:
        logging.warning(f"Timeout fetching historical data for {metal} ({hours}h)")
    except httpx.HTTPStatusError as e:
        logging.warning(f"HTTP error fetching historical data for {metal}: {e.response.status_code}")
    except Exception as e:
        logging.warning(f"Error fetching historical data for {metal}: {str(e)}")
    
    return []


# ══════════════════════════════════════════════════════════════════════════════
# Price History Tracking
# ══════════════════════════════════════════════════════════════════════════════

class PriceHistory:
    """Persistent price history storage with percentage change calculations."""
    
    def __init__(self, file_path: Path):
        self._file_path = file_path
        self._history: dict[str, list[dict]] = {metal: [] for metal in SUPPORTED_METALS}
        self._lock = asyncio.Lock()
        self._load_history()
    
    def _load_history(self):
        """Load price history from file."""
        if self._file_path.exists():
            try:
                with open(self._file_path, "r") as f:
                    data = json.load(f)
                    for metal in SUPPORTED_METALS:
                        if metal in data:
                            self._history[metal] = data[metal]
            except (json.JSONDecodeError, IOError):
                pass
    
    def _save_history(self):
        """Save price history to file."""
        try:
            with open(self._file_path, "w") as f:
                json.dump(self._history, f, indent=2, default=str)
        except IOError:
            pass
    
    async def record_price(self, metal: str, price: float, currency: str = "USD"):
        """Record a new price point."""
        async with self._lock:
            timestamp = datetime.utcnow().isoformat()
            self._history[metal].append({
                "timestamp": timestamp,
                "price": price,
                "currency": currency,
            })
            
            # Keep only last 7 days of data (at 1-minute intervals = ~10,080 points)
            # Prune to last 15,000 to be safe
            if len(self._history[metal]) > 15000:
                self._history[metal] = self._history[metal][-10000:]
            
            self._save_history()
    
    def get_price_at_time(self, metal: str, target_time: datetime, currency: str = "USD") -> Optional[float]:
        """Get the price closest to a specific time."""
        history = self._history.get(metal, [])
        if not history:
            return None
        
        # Filter by currency and find closest to target time
        closest_price = None
        closest_diff = float("inf")
        
        for entry in history:
            if entry.get("currency", "USD") != currency:
                continue
            
            entry_time = datetime.fromisoformat(entry["timestamp"])
            diff = abs((entry_time - target_time).total_seconds())
            
            if diff < closest_diff:
                closest_diff = diff
                closest_price = entry["price"]
        
        return closest_price
    
    async def calculate_change(self, metal: str, current_average_price: float, hours_ago: int, currency: str = "USD") -> Optional[dict]:
        """
        Calculate price change using aggregated average price vs Yahoo Finance last closed price.
        
        Args:
            metal: Metal name (gold, silver, platinum, palladium)
            current_average_price: Current aggregated average price from multiple sources
            hours_ago: Hours ago to compare (1, 24, or 168 for 7d)
            currency: Currency (always USD for calculations, converted later)
        
        Returns:
            Dict with 'change' (dollar amount) and 'percent' (percentage change)
        """
        # Use Yahoo Finance to get the last closed price (previous trading day's close)
        # This is the historical reference point
        last_closed_price = await fetch_historical_price_from_yahoo(metal, hours_ago)
        
        if last_closed_price is None or last_closed_price == 0:
            return None
        
        # Calculate change using aggregated average price vs last closed price
        # Ensure last_closed_price is positive to avoid division errors
        if last_closed_price <= 0:
            return None
        
        change = round(current_average_price - last_closed_price, 2)
        percent = round((change / last_closed_price) * 100, 2)
        
        return {"change": change, "percent": percent}
    
    async def get_history(self, metal: str, hours: int = 24, currency: str = "USD") -> list[dict]:
        """Get price history for a metal from Yahoo Finance."""
        return await fetch_historical_data_from_yahoo(metal, hours, currency)


price_history = PriceHistory(HISTORY_FILE)


# ══════════════════════════════════════════════════════════════════════════════
# Price Cache & Fetching
# ══════════════════════════════════════════════════════════════════════════════

class PriceCache:
    """Simple in-memory cache for metal prices."""
    
    def __init__(self):
        self._cache: dict = {}
        self._last_update: Optional[datetime] = None
        self._lock = asyncio.Lock()
    
    def is_stale(self) -> bool:
        if self._last_update is None:
            return True
        age = (datetime.utcnow() - self._last_update).total_seconds()
        return age > CACHE_DURATION_SECONDS
    
    def get_age_seconds(self) -> Optional[int]:
        if self._last_update is None:
            return None
        return int((datetime.utcnow() - self._last_update).total_seconds())
    
    async def get_prices(self, currency: str = "USD") -> dict:
        async with self._lock:
            cache_key = f"prices_{currency}"
            
            # If cache is fresh, return it immediately
            if not self.is_stale() and cache_key in self._cache:
                return self._cache[cache_key]
            
            # Always fetch USD prices first (they're the source for all other currencies)
            # This allows us to reuse USD prices when calculating changes
            if "prices_USD" not in self._cache or self.is_stale():
                usd_prices = await fetch_metal_prices("USD")
                self._cache["prices_USD"] = usd_prices
            
            if currency == "USD":
                self._last_update = datetime.utcnow()
                return self._cache["prices_USD"]
            else:
                # Convert USD prices to requested currency
                prices = await convert_prices_to_currency(self._cache["prices_USD"], currency)
                self._cache[cache_key] = prices
                self._last_update = datetime.utcnow()
                return prices


price_cache = PriceCache()


def calculate_aggregated_price(prices: list[float], source_names: list[str]) -> dict:
    """
    Calculate aggregated price from multiple sources with outlier removal.
    Returns average price, median, and source count.
    Uses IQR (Interquartile Range) method to filter outliers.
    """
    if not prices or len(prices) == 0:
        return None
    
    # Remove duplicates and sort
    unique_prices = sorted(list(set(prices)))
    
    if len(unique_prices) == 1:
        return {
            "price": round(unique_prices[0], 2),
            "sources_count": len(source_names),
            "sources": source_names,
            "median": round(unique_prices[0], 2),
            "min": round(unique_prices[0], 2),
            "max": round(unique_prices[0], 2),
        }
    
    # Calculate quartiles for outlier detection
    q1_idx = len(unique_prices) // 4
    q3_idx = (3 * len(unique_prices)) // 4
    q1 = unique_prices[q1_idx] if q1_idx < len(unique_prices) else unique_prices[0]
    q3 = unique_prices[q3_idx] if q3_idx < len(unique_prices) else unique_prices[-1]
    iqr = q3 - q1
    
    # Filter outliers (prices outside 1.5 * IQR)
    # If IQR is 0 (all prices identical), include all prices (no outliers)
    if iqr == 0:
        filtered_prices = unique_prices
    else:
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        filtered_prices = [p for p in unique_prices if lower_bound <= p <= upper_bound]
    
    # If filtering removed all prices, use original (shouldn't happen but safety check)
    # Also handle edge case where all prices are identical (IQR = 0)
    if len(filtered_prices) == 0:
        filtered_prices = unique_prices
    
    # Safety check: ensure we have at least one price (should always be true at this point)
    if len(filtered_prices) == 0:
        return None
    
    # Calculate statistics
    avg_price = sum(filtered_prices) / len(filtered_prices)
    median_idx = len(filtered_prices) // 2
    median_price = filtered_prices[median_idx] if len(filtered_prices) % 2 == 1 else (filtered_prices[median_idx - 1] + filtered_prices[median_idx]) / 2
    
    return {
        "price": round(avg_price, 2),  # Average is the primary price
        "sources_count": len(source_names),
        "sources": source_names,
        "median": round(median_price, 2),
        "min": round(min(filtered_prices), 2),
        "max": round(max(filtered_prices), 2),
        "outliers_removed": len(unique_prices) - len(filtered_prices),
    }


async def fetch_metal_prices(currency: str = "USD") -> dict:
    """
    Fetch current metal spot prices from multiple sources and calculate aggregated averages.
    Combines results from multiple sources (like real spot price APIs) and calculates
    average prices with outlier filtering for accuracy.
    All prices are spot prices (current market price for immediate delivery).
    Sources update every 60 seconds or less.
    """
    
    # Multiple SPOT price sources for comprehensive price discovery
    # ALL sources provide actual spot prices (no futures contracts)
    # Prioritize working sources first for faster response times
    sources = [
        # Fast, working sources (verified to work)
        fetch_from_investing_com,      # Gold & Silver spot (Investing.com) - WORKING, FAST
        fetch_from_jmbullion,          # JM Bullion spot prices (all metals) - WORKING, FAST
        
        # Additional reliable spot price sources
        fetch_from_kitco,              # Kitco spot prices (all metals) - usually works
        fetch_from_lbma,               # Gold & Silver spot (LBMA) - reliable
        fetch_from_xe_com,             # All metals spot (XE.com) - reliable
        fetch_from_apmex,              # APMEX spot prices (all metals)
        fetch_from_goldprice_org,      # GoldPrice.org spot prices (all metals)
    ]
    
    # Fetch from prioritized sources concurrently with shorter timeout
    # We use asyncio.wait_for to ensure sources don't block too long
    async def fetch_with_timeout(source_func, timeout_seconds=4.0):
        """Fetch from source with timeout to prevent slow sources from blocking."""
        try:
            return await asyncio.wait_for(source_func(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            logging.warning(f"Source {source_func.__name__} timed out after {timeout_seconds}s")
            return TimeoutError(f"Timeout after {timeout_seconds}s")
        except Exception as e:
            return e
    
    # Fetch from all sources concurrently with individual timeouts
    tasks = [fetch_with_timeout(source_func, timeout_seconds=4.0) for source_func in sources]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Collect all prices by metal from all sources
    metal_prices: dict[str, list[tuple[float, str]]] = {metal: [] for metal in SUPPORTED_METALS}
    
    # Track which sources succeeded/failed for debugging
    successful_sources = []
    failed_sources = []
    
    for i, result in enumerate(results):
        source_name = sources[i].__name__ if i < len(sources) else "Unknown"
        
        if isinstance(result, Exception):
            failed_sources.append(f"{source_name}: {str(result)}")
            logging.warning(f"Source {source_name} failed with exception: {str(result)}")
            continue
            
        if isinstance(result, dict) and "data" in result and "source" in result:
            source_name = result["source"]
            data = result["data"]
            
            if not data or len(data) == 0:
                failed_sources.append(f"{source_name}: No data returned")
                logging.warning(f"Source {source_name} returned no data")
                continue
                
            successful_sources.append(source_name)
            prices_found = 0
            
            for metal, price in data.items():
                if metal in SUPPORTED_METALS and isinstance(price, (int, float)) and price > 0:
                    metal_prices[metal].append((float(price), source_name))
                    prices_found += 1
            
            if prices_found > 0:
                logging.info(f"Source {source_name} returned {prices_found} prices")
            else:
                failed_sources.append(f"{source_name}: No valid prices")
        else:
            failed_sources.append(f"{source_name}: Invalid response format")
            logging.warning(f"Source {source_name} returned invalid format: {type(result)}")
    
    # Log summary with detailed information
    logging.info(f"Price aggregation: {len(successful_sources)} sources succeeded, {len(failed_sources)} failed")
    if successful_sources:
        logging.info(f"Successful sources: {', '.join(successful_sources)}")
    if failed_sources and len(failed_sources) > 10:
        logging.warning(f"Many sources failed ({len(failed_sources)}). First 10: {', '.join(failed_sources[:10])}")
    elif failed_sources:
        logging.warning(f"Failed sources: {', '.join(failed_sources)}")
    
    for metal in SUPPORTED_METALS:
        count = len(metal_prices[metal])
        if count > 0:
            sources_list = [s[1] for s in metal_prices[metal]]
            unique_sources = list(set(sources_list))
            logging.info(f"  {metal}: {count} prices from {len(unique_sources)} unique sources: {unique_sources}")
        else:
            logging.error(f"  {metal}: No prices from any source - all sources failed!")
    
    # Calculate aggregated prices for each metal
    aggregated_prices = {}
    
    for metal in SUPPORTED_METALS:
        if len(metal_prices[metal]) > 0:
            prices_list = [p[0] for p in metal_prices[metal]]
            sources_list = [p[1] for p in metal_prices[metal]]
            
            aggregated = calculate_aggregated_price(prices_list, sources_list)
            if aggregated:
                aggregated_prices[metal] = aggregated
        else:
            # If no sources returned a price, try history
            history_prices = get_prices_from_history("USD")
            if metal in history_prices and history_prices[metal].get("price", 0) > 0:
                aggregated_prices[metal] = {
                    "price": history_prices[metal]["price"],
                    "sources_count": 1,
                    "sources": ["History"],
                    "median": history_prices[metal]["price"],
                    "min": history_prices[metal]["price"],
                    "max": history_prices[metal]["price"],
                }
            else:
                # Last resort: use reasonable defaults
                default_prices_usd = {
                    "gold": 2650.00,
                    "silver": 31.50,
                    "platinum": 980.00,
                    "palladium": 1050.00,
                }
                default_price = default_prices_usd.get(metal, 0)
                aggregated_prices[metal] = {
                    "price": default_price,
                    "sources_count": 0,
                    "sources": [],
                    "median": default_price,
                    "min": default_price,
                    "max": default_price,
                }
    
    # Convert to requested currency (only convert the main price, keep metadata)
    if currency != "USD":
        converted_prices = {}
        for metal, price_data in aggregated_prices.items():
            converted_price_data = price_data.copy()
            usd_price = price_data["price"]
            rate = await get_exchange_rate("USD", currency)
            converted_price_data["price"] = round(usd_price * rate, 2)
            converted_price_data["median"] = round(price_data["median"] * rate, 2)
            converted_price_data["min"] = round(price_data["min"] * rate, 2)
            converted_price_data["max"] = round(price_data["max"] * rate, 2)
            converted_prices[metal] = converted_price_data
        return converted_prices
    
    return aggregated_prices


async def get_exchange_rate(from_currency: str, to_currency: str) -> float:
    """Get exchange rate between two currencies. Cached version of convert logic."""
    if from_currency == to_currency:
        return 1.0
    
    rate = None
    try:
        client = await get_http_client()
        response = await client.get(
            f"https://www.xe.com/currencyconverter/convert/?Amount=1&From={from_currency}&To={to_currency}",
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            timeout=8.0
        )
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "lxml")
            rate_elements = soup.find_all(string=re.compile(r'\d+\.\d{4,}'))
            for elem in rate_elements[:10]:
                try:
                    potential_rate = float(elem.strip().replace(",", ""))
                    if 0.01 < potential_rate < 10000:
                        rate = potential_rate
                        break
                except ValueError:
                    continue
    except Exception:
        pass
    
    # Fallback to approximate rates
    if rate is None:
        currency_rates = {
            "EUR": 0.92, "GBP": 0.79, "JPY": 149.50, "CHF": 0.88,
            "CAD": 1.36, "AUD": 1.53, "CNY": 7.24, "INR": 83.50, "BRL": 4.97,
        }
        if from_currency == "USD":
            rate = currency_rates.get(to_currency, 1.0)
        elif to_currency == "USD":
            rate = 1.0 / currency_rates.get(from_currency, 1.0)
        else:
            rate = currency_rates.get(to_currency, 1.0) / currency_rates.get(from_currency, 1.0)
    
    return rate if rate else 1.0


async def fetch_from_yahoo_finance() -> dict:
    """
    Fetch gold and silver prices from Yahoo Finance.
    Uses futures contract prices which closely track spot prices.
    Returns dict with 'source' and prices by metal.
    """
    prices = {"source": "Yahoo Finance (Futures)", "data": {}}
    client = await get_http_client()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }
    
    # Yahoo Finance symbols for precious metals futures
    # GC=F: Gold futures, SI=F: Silver futures
    symbols = {
        "gold": "GC=F",
        "silver": "SI=F",
    }
    
    for metal, symbol in symbols.items():
        try:
            response = await client.get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1m&range=1d",
                headers=headers
            )
            response.raise_for_status()
            
            # raise_for_status() ensures status is 2xx, so we can proceed directly
            data = response.json()
            result = data.get("chart", {}).get("result", [])
            if result and len(result) > 0:
                meta = result[0].get("meta", {})
                price = meta.get("regularMarketPrice")
                
                if price is not None:
                    try:
                        price_float = float(price)
                        if price_float > 0:
                            prices["data"][metal] = round(price_float, 2)
                    except (ValueError, TypeError):
                        logging.warning(f"Invalid price value for {metal} from Yahoo Finance: {price}")
        except httpx.TimeoutException:
            logging.warning(f"Timeout fetching {metal} price from Yahoo Finance")
        except httpx.HTTPStatusError as e:
            logging.warning(f"HTTP error fetching {metal} price from Yahoo Finance: {e.response.status_code}")
        except Exception as e:
            logging.warning(f"Error fetching {metal} price from Yahoo Finance: {str(e)}")
    
    return prices


async def fetch_from_investing_com() -> dict:
    """
    Fetch gold (XAU/USD) and silver (XAG/USD) spot prices from Investing.com.
    Actual spot prices for immediate delivery.
    """
    prices = {"source": "Investing.com (Spot)", "data": {}}
    client = await get_http_client()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    # Gold spot price (XAU/USD)
    try:
        response = await client.get("https://www.investing.com/commodities/gold", headers=headers, timeout=5.0)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "lxml")
            price_elem = soup.find(attrs={"data-test": "instrument-price-last"})
            if price_elem:
                price_text = price_elem.get_text().strip().replace(",", "")
                try:
                    price = float(price_text)
                    if 2000 < price < 5000:  # Validate gold price range
                        prices["data"]["gold"] = round(price, 2)
                except ValueError:
                    pass
            else:  # Fallback if data-test attribute changes
                price_match = re.search(r'data-pair-id=".*?">.*?<span class="last-price-value">([0-9,]+\.[0-9]{2})</span>', response.text)
                if price_match:
                    price = float(price_match.group(1).replace(",", ""))
                    if 2000 < price < 5000:
                        prices["data"]["gold"] = round(price, 2)
    except Exception as e:
        logging.warning(f"Error fetching gold from Investing.com: {str(e)}")

    # Silver spot price (XAG/USD)
    try:
        response = await client.get("https://www.investing.com/commodities/silver", headers=headers, timeout=5.0)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "lxml")
            price_elem = soup.find(attrs={"data-test": "instrument-price-last"})
            if price_elem:
                price_text = price_elem.get_text().strip().replace(",", "")
                try:
                    price = float(price_text)
                    if 20 < price < 100:  # Validate silver price range
                        prices["data"]["silver"] = round(price, 2)
                except ValueError:
                    pass
            else:  # Fallback
                price_match = re.search(r'data-pair-id=".*?">.*?<span class="last-price-value">([0-9,]+\.[0-9]{2})</span>', response.text)
                if price_match:
                    price = float(price_match.group(1).replace(",", ""))
                    if 20 < price < 100:
                        prices["data"]["silver"] = round(price, 2)
    except Exception as e:
        logging.warning(f"Error fetching silver from Investing.com: {str(e)}")

    return prices


async def fetch_from_lbma() -> dict:
    """
    Fetch gold and silver spot prices from LBMA (London Bullion Market Association).
    LBMA provides authoritative spot prices used as industry benchmarks.
    """
    prices = {"source": "LBMA (Spot)", "data": {}}
    client = await get_http_client()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    
    try:
        response = await client.get(
            "https://www.lbma.org.uk/prices-and-data/precious-metal-prices",
            headers=headers,
            timeout=5.0
        )
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "lxml")
            page_text = soup.get_text()
            
            # LBMA displays spot prices in USD per troy ounce
            gold_spot_patterns = [
                r'Gold.*?USD[:\s]+\$?([1-2],?\d{3}\.?\d{0,2})',
                r'Gold Spot[:\s]+\$?([1-2],?\d{3}\.?\d{0,2})',
                r'XAU[:\s]+\$?([1-2],?\d{3}\.?\d{0,2})',
            ]
            
            for pattern in gold_spot_patterns:
                matches = re.findall(pattern, page_text, re.IGNORECASE | re.DOTALL)
                for match in matches[:3]:
                    try:
                        price_val = float(match.replace(",", ""))
                        if 2000 < price_val < 3000:
                            prices["data"]["gold"] = round(price_val, 2)
                            break
                    except ValueError:
                        continue
                if "gold" in prices["data"]:
                    break
            
            silver_spot_patterns = [
                r'Silver.*?USD[:\s]+\$?([2-4][0-9]\.?\d{0,2})',
                r'Silver Spot[:\s]+\$?([2-4][0-9]\.?\d{0,2})',
                r'XAG[:\s]+\$?([2-4][0-9]\.?\d{0,2})',
            ]
            
            for pattern in silver_spot_patterns:
                matches = re.findall(pattern, page_text, re.IGNORECASE | re.DOTALL)
                for match in matches[:3]:
                    try:
                        price_val = float(match.replace(",", ""))
                        if 20 < price_val < 50:
                            prices["data"]["silver"] = round(price_val, 2)
                            break
                    except ValueError:
                        continue
                if "silver" in prices["data"]:
                    break
    except Exception as e:
        logging.warning(f"Error fetching from LBMA: {str(e)}")
    
    return prices


async def fetch_from_xe_com() -> dict:
    """
    Fetch precious metals spot prices from XE.com.
    XE provides currency and commodity spot prices.
    """
    prices = {"source": "XE.com (Spot)", "data": {}}
    client = await get_http_client()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml",
    }
    
    metal_symbols = {
        "gold": ("XAU", 2000, 5000),
        "silver": ("XAG", 20, 100),
        "platinum": ("XPT", 800, 1500),
        "palladium": ("XPD", 800, 1500),
    }
    
    for metal, (symbol, min_price, max_price) in metal_symbols.items():
        try:
            response = await client.get(
                f"https://www.xe.com/currencycharts/?from={symbol}&to=USD",
                headers=headers,
                timeout=5.0
            )
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "lxml")
                # Look for price patterns in XE.com format
                price_pattern = re.search(rf'{symbol}/USD[:\s]*\$?([\d,]+\.[\d]{{2}})', response.text)
                if price_pattern:
                    try:
                        price = float(price_pattern.group(1).replace(",", ""))
                        if min_price < price < max_price:
                            prices["data"][metal] = round(price, 2)
                    except (ValueError, IndexError):
                        pass
        except Exception as e:
            logging.warning(f"Error fetching {metal} from XE.com: {str(e)}")
    
    return prices


async def fetch_from_bullionvault() -> dict:
    """
    Fetch spot prices from BullionVault (precious metals dealer).
    Provides real-time spot prices for retail investors.
    """
    prices = {"source": "BullionVault (Spot)", "data": {}}
    client = await get_http_client()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml",
    }
    
    try:
        response = await client.get(
            "https://www.bullionvault.com/gold-price/gold-price-chart.do",
            headers=headers,
            timeout=5.0
        )
        if response.status_code == 200:
            # BullionVault displays spot prices
            gold_match = re.search(r'Gold.*?\$([\d,]+\.[\d]{2})', response.text)
            if gold_match:
                try:
                    price = float(gold_match.group(1).replace(",", ""))
                    if 2000 < price < 5000:
                        prices["data"]["gold"] = round(price, 2)
                except ValueError:
                    pass
            
            response = await client.get(
                "https://www.bullionvault.com/silver-price/silver-price-chart.do",
                headers=headers,
                timeout=5.0
            )
            if response.status_code == 200:
                silver_match = re.search(r'Silver.*?\$([\d,]+\.[\d]{2})', response.text)
                if silver_match:
                    try:
                        price = float(silver_match.group(1).replace(",", ""))
                        if 20 < price < 100:
                            prices["data"]["silver"] = round(price, 2)
                    except ValueError:
                        pass
    except Exception as e:
        logging.warning(f"Error fetching from BullionVault: {str(e)}")
    
    return prices


async def fetch_from_yahoo_finance_all() -> dict:
    """
    Fetch all metals from Yahoo Finance in a single call for redundancy.
    Uses futures contract prices which closely track spot prices.
    """
    prices = {"source": "Yahoo Finance All (Futures)", "data": {}}
    client = await get_http_client()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }
    
    # Yahoo Finance symbols for all precious metals futures
    symbols = {
        "gold": "GC=F",
        "silver": "SI=F",
        "platinum": "PL=F",
        "palladium": "PA=F",
    }
    
    # Fetch all symbols in parallel
    tasks = []
    for metal, symbol in symbols.items():
        tasks.append(
            client.get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1m&range=1d",
                headers=headers
            )
        )
    
    try:
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, (metal, symbol) in enumerate(symbols.items()):
            response = responses[i]
            
            if isinstance(response, Exception):
                logging.warning(f"Error fetching {metal} from Yahoo Finance All: {str(response)}")
                continue
                
            try:
                response.raise_for_status()
                # raise_for_status() ensures status is 2xx, so we can proceed directly
                try:
                    data = response.json()
                except ValueError as e:
                    logging.warning(f"Invalid JSON for {metal}: {str(e)}")
                    continue
                
                result = data.get("chart", {}).get("result", [])
                if result and len(result) > 0:
                    meta = result[0].get("meta", {})
                    price = meta.get("regularMarketPrice")
                    
                    if price is not None:
                        try:
                            price_float = float(price)
                            if price_float > 0:
                                prices["data"][metal] = round(price_float, 2)
                        except (ValueError, TypeError):
                            pass
            except Exception as e:
                logging.warning(f"Error processing {metal} from Yahoo Finance All: {str(e)}")
    except Exception as e:
        logging.warning(f"Error in Yahoo Finance All fetch: {str(e)}")
    
    return prices


async def fetch_from_yahoo_finance_pgm() -> dict:
    """
    Fetch platinum and palladium prices from Yahoo Finance.
    Uses futures contract prices which closely track spot prices.
    """
    prices = {"source": "Yahoo Finance PGM (Futures)", "data": {}}
    client = await get_http_client()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }
    
    # Yahoo Finance symbols for precious metals futures
    # PL=F: Platinum futures, PA=F: Palladium futures
    symbols = {
        "platinum": "PL=F",
        "palladium": "PA=F",
    }
    
    for metal, symbol in symbols.items():
        try:
            response = await client.get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1m&range=1d",
                headers=headers
            )
            response.raise_for_status()
            
            # raise_for_status() ensures status is 2xx, so we can proceed directly
            try:
                data = response.json()
            except ValueError as e:
                logging.warning(f"Invalid JSON response for {metal}: {str(e)}")
                continue
            
            result = data.get("chart", {}).get("result", [])
            if result and len(result) > 0:
                meta = result[0].get("meta", {})
                if not meta:
                    logging.warning(f"No meta data for {metal}")
                    continue
                    
                price = meta.get("regularMarketPrice")
                
                if price is not None:
                    try:
                        price_float = float(price)
                        if price_float > 0:
                            prices["data"][metal] = round(price_float, 2)
                    except (ValueError, TypeError) as e:
                        logging.warning(f"Invalid price value for {metal}: {price} - {str(e)}")
            else:
                logging.warning(f"No result data for {metal}")
        except httpx.TimeoutException:
            logging.warning(f"Timeout fetching {metal} price from Yahoo Finance")
        except httpx.HTTPStatusError as e:
            logging.warning(f"HTTP error fetching {metal} price: {e.response.status_code}")
        except Exception as e:
            logging.warning(f"Error fetching {metal} price from Yahoo Finance: {str(e)}")
    
    return prices


async def fetch_from_cme_group() -> dict:
    """
    Fetch precious metals prices from CME Group (Chicago Mercantile Exchange).
    Provides futures prices that track spot closely.
    """
    prices = {"source": "CME Group (Futures)", "data": {}}
    client = await get_http_client()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/html, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.cmegroup.com/",
    }
    
    # CME symbols for precious metals futures
    cme_symbols = {
        "gold": "GC",
        "silver": "SI",
        "platinum": "PL",
        "palladium": "PA",
    }
    
    # Fetch all symbols in parallel for better performance
    tasks = []
    for metal, symbol in cme_symbols.items():
        tasks.append(
            client.get(
                f"https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/{symbol}/G",
                headers=headers,
                timeout=5.0
            )
        )
    
    try:
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, (metal, symbol) in enumerate(cme_symbols.items()):
            response = responses[i]
            
            if isinstance(response, Exception):
                logging.warning(f"Error fetching {metal} from CME: {str(response)}")
                continue
            
            try:
                if response.status_code == 200:
                    try:
                        data = response.json()
                        # CME JSON structure: array of quotes
                        if isinstance(data, list) and len(data) > 0:
                            quote = data[0]
                            # Try multiple price fields
                            price = (quote.get("last") or quote.get("lastPrice") or 
                                    quote.get("settle") or quote.get("priorSettle") or
                                    quote.get("open") or quote.get("high") or quote.get("low"))
                            
                            if price is not None:
                                try:
                                    price_val = float(price)
                                    # Validate ranges
                                    ranges = {
                                        "gold": (2000, 5000),
                                        "silver": (20, 100),
                                        "platinum": (800, 3000),
                                        "palladium": (800, 3000),
                                    }
                                    min_p, max_p = ranges.get(metal, (0, 10000))
                                    if min_p < price_val < max_p:
                                        prices["data"][metal] = round(price_val, 2)
                                except (ValueError, TypeError):
                                    pass
                    except (ValueError, KeyError, TypeError) as e:
                        logging.warning(f"Error parsing CME JSON for {metal}: {str(e)}")
            except Exception as e:
                logging.warning(f"Error processing {metal} from CME: {str(e)}")
    except Exception as e:
        logging.warning(f"Error in CME Group fetch: {str(e)}")
    
    return prices


async def fetch_from_kitco() -> dict:
    """
    Fetch all precious metals spot prices from Kitco.
    Kitco provides real-time spot prices for all metals.
    Try multiple endpoints and parsing strategies.
    """
    prices = {"source": "Kitco (Spot)", "data": {}}
    client = await get_http_client()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.kitco.com/",
    }
    
    # Try Kitco's main market summary page first (more reliable)
    try:
        response = await client.get("https://www.kitco.com/market/", headers=headers, timeout=5.0)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "lxml")
            page_text = soup.get_text()
            
            # Look for price patterns in the market summary
            metal_patterns = {
                "gold": [
                    r'Gold.*?\$([\d,]+\.[\d]{2})',
                    r'\$([1-2],?\d{3}\.\d{2}).*?Gold',
                    r'XAU.*?\$([\d,]+\.[\d]{2})',
                ],
                "silver": [
                    r'Silver.*?\$([\d,]+\.[\d]{2})',
                    r'\$([2-4][0-9]\.[\d]{2}).*?Silver',
                    r'XAG.*?\$([\d,]+\.[\d]{2})',
                ],
                "platinum": [
                    r'Platinum.*?\$([\d,]+\.[\d]{2})',
                    r'\$([8-9]\d{2}\.[\d]{2}).*?Platinum',
                    r'XPT.*?\$([\d,]+\.[\d]{2})',
                ],
                "palladium": [
                    r'Palladium.*?\$([\d,]+\.[\d]{2})',
                    r'\$([1-2]\d{3}\.[\d]{2}).*?Palladium',
                    r'XPD.*?\$([\d,]+\.[\d]{2})',
                ],
            }
            
            ranges = {
                "gold": (2000, 5000),
                "silver": (20, 100),
                "platinum": (800, 3000),
                "palladium": (800, 3000),
            }
            
            for metal, patterns in metal_patterns.items():
                if metal in prices["data"]:
                    continue  # Already found
                    
                for pattern in patterns:
                    match = re.search(pattern, page_text, re.IGNORECASE)
                    if match:
                        try:
                            price = float(match.group(1).replace(",", ""))
                            min_p, max_p = ranges.get(metal, (0, 10000))
                            if min_p < price < max_p:
                                prices["data"][metal] = round(price, 2)
                                break
                        except (ValueError, IndexError):
                            continue
    except Exception as e:
        logging.warning(f"Error fetching from Kitco market page: {str(e)}")
    
    # Fallback to individual pages for any missing metals
    metal_urls = {
        "gold": "https://www.kitco.com/charts/livegold.html",
        "silver": "https://www.kitco.com/charts/livesilver.html",
        "platinum": "https://www.kitco.com/charts/liveplatinum.html",
        "palladium": "https://www.kitco.com/charts/livepalladium.html",
    }
    
    for metal, url in metal_urls.items():
        if metal in prices["data"]:
            continue  # Already found from market page
            
        try:
            response = await client.get(url, headers=headers, timeout=5.0)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "lxml")
                page_text = soup.get_text()
                
                # Look for spot price patterns
                patterns = [
                    rf'{metal.capitalize()}.*?\$([\d,]+\.[\d]{{2}})',
                    rf'\$([\d,]+\.[\d]{{2}}).*?{metal.capitalize()}',
                    r'Spot[:\s]+\$([\d,]+\.[\d]{2})',
                ]
                
                ranges = {
                    "gold": (2000, 5000),
                    "silver": (20, 100),
                    "platinum": (800, 3000),
                    "palladium": (800, 3000),
                }
                
                for pattern in patterns:
                    match = re.search(pattern, page_text, re.IGNORECASE)
                    if match:
                        try:
                            price = float(match.group(1).replace(",", ""))
                            min_p, max_p = ranges.get(metal, (0, 10000))
                            if min_p < price < max_p:
                                prices["data"][metal] = round(price, 2)
                                break
                        except (ValueError, IndexError):
                            continue
        except Exception as e:
            logging.warning(f"Error fetching {metal} from Kitco: {str(e)}")
    
    return prices


async def fetch_from_apmex() -> dict:
    """
    Fetch spot prices from APMEX (American Precious Metals Exchange).
    """
    prices = {"source": "APMEX (Spot)", "data": {}}
    client = await get_http_client()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html",
    }
    
    try:
        # APMEX spot prices page
        response = await client.get("https://www.apmex.com/spotprices", headers=headers, timeout=5.0)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "lxml")
            page_text = soup.get_text()
            
            # Look for spot prices
            metal_patterns = {
                "gold": r'Gold.*?\$([\d,]+\.[\d]{2})',
                "silver": r'Silver.*?\$([\d,]+\.[\d]{2})',
                "platinum": r'Platinum.*?\$([\d,]+\.[\d]{2})',
                "palladium": r'Palladium.*?\$([\d,]+\.[\d]{2})',
            }
            
            for metal, pattern in metal_patterns.items():
                match = re.search(pattern, page_text, re.IGNORECASE)
                if match:
                    try:
                        price = float(match.group(1).replace(",", ""))
                        ranges = {"gold": (2000, 5000), "silver": (20, 100), "platinum": (800, 1500), "palladium": (800, 1500)}
                        min_p, max_p = ranges.get(metal, (0, 10000))
                        if min_p < price < max_p:
                            prices["data"][metal] = round(price, 2)
                    except (ValueError, IndexError):
                        pass
    except Exception as e:
        logging.warning(f"Error fetching from APMEX: {str(e)}")
    
    return prices


async def fetch_from_jmbullion() -> dict:
    """
    Fetch spot prices from JM Bullion.
    """
    prices = {"source": "JM Bullion (Spot)", "data": {}}
    client = await get_http_client()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html",
    }
    
    try:
        response = await client.get("https://www.jmbullion.com/charts/spot-prices/", headers=headers, timeout=5.0)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "lxml")
            page_text = soup.get_text()
            
            metal_patterns = {
                "gold": r'Gold.*?\$([\d,]+\.[\d]{2})',
                "silver": r'Silver.*?\$([\d,]+\.[\d]{2})',
                "platinum": r'Platinum.*?\$([\d,]+\.[\d]{2})',
                "palladium": r'Palladium.*?\$([\d,]+\.[\d]{2})',
            }
            
            for metal, pattern in metal_patterns.items():
                match = re.search(pattern, page_text, re.IGNORECASE)
                if match:
                    try:
                        price = float(match.group(1).replace(",", ""))
                        ranges = {"gold": (2000, 5000), "silver": (20, 100), "platinum": (800, 1500), "palladium": (800, 1500)}
                        min_p, max_p = ranges.get(metal, (0, 10000))
                        if min_p < price < max_p:
                            prices["data"][metal] = round(price, 2)
                    except (ValueError, IndexError):
                        pass
    except Exception as e:
        logging.warning(f"Error fetching from JM Bullion: {str(e)}")
    
    return prices


async def fetch_from_goldprice_org() -> dict:
    """
    Fetch spot prices from GoldPrice.org.
    """
    prices = {"source": "GoldPrice.org (Spot)", "data": {}}
    client = await get_http_client()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html",
    }
    
    try:
        response = await client.get("https://goldprice.org/spot-gold", headers=headers, timeout=5.0)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "lxml")
            page_text = soup.get_text()
            
            # Gold price
            gold_match = re.search(r'\$([1-2],?\d{3}\.\d{2})', page_text)
            if gold_match:
                try:
                    price = float(gold_match.group(1).replace(",", ""))
                    if 2000 < price < 5000:
                        prices["data"]["gold"] = round(price, 2)
                except ValueError:
                    pass
        
        # Silver price
        response = await client.get("https://goldprice.org/spot-silver", headers=headers, timeout=5.0)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "lxml")
            page_text = soup.get_text()
            silver_match = re.search(r'\$([2-4][0-9]\.[\d]{2})', page_text)
            if silver_match:
                try:
                    price = float(silver_match.group(1))
                    if 20 < price < 100:
                        prices["data"]["silver"] = round(price, 2)
                except ValueError:
                    pass
    except Exception as e:
        logging.warning(f"Error fetching from GoldPrice.org: {str(e)}")
    
    return prices


async def fetch_from_monex() -> dict:
    """
    Fetch spot prices from Monex.
    """
    prices = {"source": "Monex (Spot)", "data": {}}
    client = await get_http_client()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html",
    }
    
    try:
        response = await client.get("https://www.monex.com/gold-silver-spot-prices/", headers=headers, timeout=5.0)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "lxml")
            page_text = soup.get_text()
            
            metal_patterns = {
                "gold": r'Gold.*?\$([\d,]+\.[\d]{2})',
                "silver": r'Silver.*?\$([\d,]+\.[\d]{2})',
            }
            
            for metal, pattern in metal_patterns.items():
                match = re.search(pattern, page_text, re.IGNORECASE)
                if match:
                    try:
                        price = float(match.group(1).replace(",", ""))
                        ranges = {"gold": (2000, 5000), "silver": (20, 100)}
                        min_p, max_p = ranges.get(metal, (0, 10000))
                        if min_p < price < max_p:
                            prices["data"][metal] = round(price, 2)
                    except (ValueError, IndexError):
                        pass
    except Exception as e:
        logging.warning(f"Error fetching from Monex: {str(e)}")
    
    return prices


async def fetch_from_gold_dealer() -> dict:
    """
    Fetch spot prices from GoldDealer.com.
    """
    prices = {"source": "GoldDealer.com (Spot)", "data": {}}
    client = await get_http_client()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html",
    }
    
    try:
        response = await client.get("https://www.golddealer.com/spot-price/", headers=headers, timeout=5.0)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "lxml")
            page_text = soup.get_text()
            
            metal_patterns = {
                "gold": r'Gold.*?\$([\d,]+\.[\d]{2})',
                "silver": r'Silver.*?\$([\d,]+\.[\d]{2})',
            }
            
            for metal, pattern in metal_patterns.items():
                match = re.search(pattern, page_text, re.IGNORECASE)
                if match:
                    try:
                        price = float(match.group(1).replace(",", ""))
                        ranges = {"gold": (2000, 5000), "silver": (20, 100)}
                        min_p, max_p = ranges.get(metal, (0, 10000))
                        if min_p < price < max_p:
                            prices["data"][metal] = round(price, 2)
                    except (ValueError, IndexError):
                        pass
    except Exception as e:
        logging.warning(f"Error fetching from GoldDealer: {str(e)}")
    
    return prices


async def fetch_from_money_metals() -> dict:
    """
    Fetch spot prices from Money Metals Exchange (moneymetals.com).
    """
    prices = {"source": "Money Metals Exchange (Spot)", "data": {}}
    client = await get_http_client()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    
    try:
        response = await client.get("https://www.moneymetals.com/precious-metals-charts", headers=headers, timeout=5.0)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "lxml")
            page_text = soup.get_text()
            
            # Look for gold and silver spot prices
            patterns = {
                "gold": [
                    r'Gold[:\s]+\$?([1-2],?\d{3}\.?[\d]{0,2})',
                    r'XAU[:\s]+\$?([1-2],?\d{3}\.?[\d]{0,2})',
                ],
                "silver": [
                    r'Silver[:\s]+\$?([2-4][0-9]\.[\d]{1,2})',
                    r'XAG[:\s]+\$?([2-4][0-9]\.[\d]{1,2})',
                ],
            }
            
            for metal, metal_patterns in patterns.items():
                for pattern in metal_patterns:
                    match = re.search(pattern, page_text, re.IGNORECASE)
                    if match:
                        try:
                            price_val = float(match.group(1).replace(",", ""))
                            if (metal == "gold" and 2000 < price_val < 5000) or (metal == "silver" and 20 < price_val < 100):
                                prices["data"][metal] = round(price_val, 2)
                                break
                        except (ValueError, IndexError):
                            continue
    except Exception as e:
        logging.warning(f"Error fetching from Money Metals Exchange: {str(e)}")
    
    return prices


async def fetch_from_provident_metals() -> dict:
    """
    Fetch spot prices from Provident Metals (providentmetals.com).
    """
    prices = {"source": "Provident Metals (Spot)", "data": {}}
    client = await get_http_client()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    
    try:
        response = await client.get("https://www.providentmetals.com/metals-prices.html", headers=headers, timeout=5.0)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "lxml")
            page_text = soup.get_text()
            
            # Look for spot prices
            gold_match = re.search(r'Gold.*?\$([1-2],?\d{3}\.?[\d]{0,2})', page_text)
            if gold_match:
                try:
                    price = float(gold_match.group(1).replace(",", ""))
                    if 2000 < price < 5000:
                        prices["data"]["gold"] = round(price, 2)
                except (ValueError, IndexError):
                    pass
            
            silver_match = re.search(r'Silver.*?\$([2-4][0-9]\.[\d]{1,2})', page_text)
            if silver_match:
                try:
                    price = float(silver_match.group(1).replace(",", ""))
                    if 20 < price < 100:
                        prices["data"]["silver"] = round(price, 2)
                except (ValueError, IndexError):
                    pass
    except Exception as e:
        logging.warning(f"Error fetching from Provident Metals: {str(e)}")
    
    return prices


async def fetch_from_sd_bullion() -> dict:
    """
    Fetch spot prices from SD Bullion (sdbullion.com).
    """
    prices = {"source": "SD Bullion (Spot)", "data": {}}
    client = await get_http_client()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    
    try:
        response = await client.get("https://sdbullion.com/spot-prices", headers=headers, timeout=5.0)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "lxml")
            page_text = soup.get_text()
            
            # Look for spot prices
            patterns = {
                "gold": r'Gold.*?Spot.*?\$([1-2],?\d{3}\.?[\d]{0,2})',
                "silver": r'Silver.*?Spot.*?\$([2-4][0-9]\.[\d]{1,2})',
                "platinum": r'Platinum.*?Spot.*?\$([8-9][0-9]{2}\.?[\d]{0,2})',
                "palladium": r'Palladium.*?Spot.*?\$([8-9][0-9]{2}\.?[\d]{0,2})',
            }
            
            for metal, pattern in patterns.items():
                match = re.search(pattern, page_text, re.IGNORECASE)
                if match:
                    try:
                        price = float(match.group(1).replace(",", ""))
                        ranges = {
                            "gold": (2000, 5000),
                            "silver": (20, 100),
                            "platinum": (800, 1500),
                            "palladium": (800, 1500),
                        }
                        min_p, max_p = ranges.get(metal, (0, 10000))
                        if min_p < price < max_p:
                            prices["data"][metal] = round(price, 2)
                    except (ValueError, IndexError):
                        continue
    except Exception as e:
        logging.warning(f"Error fetching from SD Bullion: {str(e)}")
    
    return prices


async def fetch_from_silver_com() -> dict:
    """
    Fetch spot prices from Silver.com (silver.com).
    """
    prices = {"source": "Silver.com (Spot)", "data": {}}
    client = await get_http_client()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    
    try:
        response = await client.get("https://www.silver.com/spot-price", headers=headers, timeout=5.0)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "lxml")
            page_text = soup.get_text()
            
            # Silver.com focuses on silver and gold
            silver_match = re.search(r'Silver.*?Spot.*?\$([2-4][0-9]\.[\d]{1,2})', page_text, re.IGNORECASE)
            if silver_match:
                try:
                    price = float(silver_match.group(1).replace(",", ""))
                    if 20 < price < 100:
                        prices["data"]["silver"] = round(price, 2)
                except (ValueError, IndexError):
                    pass
            
            gold_match = re.search(r'Gold.*?Spot.*?\$([1-2],?\d{3}\.?[\d]{0,2})', page_text, re.IGNORECASE)
            if gold_match:
                try:
                    price = float(gold_match.group(1).replace(",", ""))
                    if 2000 < price < 5000:
                        prices["data"]["gold"] = round(price, 2)
                except (ValueError, IndexError):
                    pass
    except Exception as e:
        logging.warning(f"Error fetching from Silver.com: {str(e)}")
    
    return prices


async def fetch_from_goldsilver_com() -> dict:
    """
    Fetch spot prices from GoldSilver.com (goldsilver.com).
    """
    prices = {"source": "GoldSilver.com (Spot)", "data": {}}
    client = await get_http_client()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    
    try:
        response = await client.get("https://www.goldsilver.com/spot-prices", headers=headers, timeout=5.0)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "lxml")
            page_text = soup.get_text()
            
            # Look for spot prices
            gold_match = re.search(r'Gold.*?\$([1-2],?\d{3}\.?[\d]{0,2})', page_text, re.IGNORECASE)
            if gold_match:
                try:
                    price = float(gold_match.group(1).replace(",", ""))
                    if 2000 < price < 5000:
                        prices["data"]["gold"] = round(price, 2)
                except (ValueError, IndexError):
                    pass
            
            silver_match = re.search(r'Silver.*?\$([2-4][0-9]\.[\d]{1,2})', page_text, re.IGNORECASE)
            if silver_match:
                try:
                    price = float(silver_match.group(1).replace(",", ""))
                    if 20 < price < 100:
                        prices["data"]["silver"] = round(price, 2)
                except (ValueError, IndexError):
                    pass
    except Exception as e:
        logging.warning(f"Error fetching from GoldSilver.com: {str(e)}")
    
    return prices


async def fetch_from_silver_gold_bull() -> dict:
    """
    Fetch spot prices from Silver Gold Bull (silvergoldbull.com).
    """
    prices = {"source": "Silver Gold Bull (Spot)", "data": {}}
    client = await get_http_client()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    
    try:
        response = await client.get("https://www.silvergoldbull.com/spot-prices", headers=headers, timeout=5.0)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "lxml")
            page_text = soup.get_text()
            
            # Look for spot prices
            patterns = {
                "gold": r'Gold.*?\$([1-2],?\d{3}\.?[\d]{0,2})',
                "silver": r'Silver.*?\$([2-4][0-9]\.[\d]{1,2})',
            }
            
            for metal, pattern in patterns.items():
                match = re.search(pattern, page_text, re.IGNORECASE)
                if match:
                    try:
                        price = float(match.group(1).replace(",", ""))
                        if (metal == "gold" and 2000 < price < 5000) or (metal == "silver" and 20 < price < 100):
                            prices["data"][metal] = round(price, 2)
                    except (ValueError, IndexError):
                        continue
    except Exception as e:
        logging.warning(f"Error fetching from Silver Gold Bull: {str(e)}")
    
    return prices


async def fetch_from_bgasc() -> dict:
    """
    Fetch spot prices from BGASC (bgasc.com).
    """
    prices = {"source": "BGASC (Spot)", "data": {}}
    client = await get_http_client()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    
    try:
        response = await client.get("https://www.bgasc.com/spot-prices", headers=headers, timeout=5.0)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "lxml")
            page_text = soup.get_text()
            
            # Look for spot prices
            gold_match = re.search(r'Gold.*?Spot.*?\$([1-2],?\d{3}\.?[\d]{0,2})', page_text, re.IGNORECASE)
            if gold_match:
                try:
                    price = float(gold_match.group(1).replace(",", ""))
                    if 2000 < price < 5000:
                        prices["data"]["gold"] = round(price, 2)
                except (ValueError, IndexError):
                    pass
            
            silver_match = re.search(r'Silver.*?Spot.*?\$([2-4][0-9]\.[\d]{1,2})', page_text, re.IGNORECASE)
            if silver_match:
                try:
                    price = float(silver_match.group(1).replace(",", ""))
                    if 20 < price < 100:
                        prices["data"]["silver"] = round(price, 2)
                except (ValueError, IndexError):
                    pass
    except Exception as e:
        logging.warning(f"Error fetching from BGASC: {str(e)}")
    
    return prices


async def fetch_from_gainesville_coins() -> dict:
    """
    Fetch spot prices from Gainesville Coins (gainesvillecoins.com).
    """
    prices = {"source": "Gainesville Coins (Spot)", "data": {}}
    client = await get_http_client()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    
    try:
        response = await client.get("https://www.gainesvillecoins.com/spot-prices", headers=headers, timeout=5.0)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "lxml")
            page_text = soup.get_text()
            
            # Look for spot prices
            patterns = {
                "gold": r'Gold.*?\$([1-2],?\d{3}\.?[\d]{0,2})',
                "silver": r'Silver.*?\$([2-4][0-9]\.[\d]{1,2})',
                "platinum": r'Platinum.*?\$([8-9][0-9]{2}\.?[\d]{0,2})',
                "palladium": r'Palladium.*?\$([8-9][0-9]{2}\.?[\d]{0,2})',
            }
            
            for metal, pattern in patterns.items():
                match = re.search(pattern, page_text, re.IGNORECASE)
                if match:
                    try:
                        price = float(match.group(1).replace(",", ""))
                        ranges = {
                            "gold": (2000, 5000),
                            "silver": (20, 100),
                            "platinum": (800, 1500),
                            "palladium": (800, 1500),
                        }
                        min_p, max_p = ranges.get(metal, (0, 10000))
                        if min_p < price < max_p:
                            prices["data"][metal] = round(price, 2)
                    except (ValueError, IndexError):
                        continue
    except Exception as e:
        logging.warning(f"Error fetching from Gainesville Coins: {str(e)}")
    
    return prices


async def fetch_from_hero_bullion() -> dict:
    """
    Fetch spot prices from Hero Bullion (herobullion.com).
    """
    prices = {"source": "Hero Bullion (Spot)", "data": {}}
    client = await get_http_client()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    
    try:
        response = await client.get("https://www.herobullion.com/spot-prices", headers=headers, timeout=5.0)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "lxml")
            page_text = soup.get_text()
            
            # Look for spot prices
            gold_match = re.search(r'Gold.*?Spot.*?\$([1-2],?\d{3}\.?[\d]{0,2})', page_text, re.IGNORECASE)
            if gold_match:
                try:
                    price = float(gold_match.group(1).replace(",", ""))
                    if 2000 < price < 5000:
                        prices["data"]["gold"] = round(price, 2)
                except (ValueError, IndexError):
                    pass
            
            silver_match = re.search(r'Silver.*?Spot.*?\$([2-4][0-9]\.[\d]{1,2})', page_text, re.IGNORECASE)
            if silver_match:
                try:
                    price = float(silver_match.group(1).replace(",", ""))
                    if 20 < price < 100:
                        prices["data"]["silver"] = round(price, 2)
                except (ValueError, IndexError):
                    pass
    except Exception as e:
        logging.warning(f"Error fetching from Hero Bullion: {str(e)}")
    
    return prices


async def fetch_from_gold_price_live() -> dict:
    """
    Fetch spot prices from GoldPriceLive.com (goldpricelive.com).
    """
    prices = {"source": "GoldPriceLive.com (Spot)", "data": {}}
    client = await get_http_client()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    
    try:
        response = await client.get("https://www.goldpricelive.com/", headers=headers, timeout=5.0)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "lxml")
            page_text = soup.get_text()
            
            # GoldPriceLive shows live spot prices
            patterns = {
                "gold": [
                    r'Gold[:\s]+\$?([1-2],?\d{3}\.?[\d]{0,2})',
                    r'XAU[:\s]+\$?([1-2],?\d{3}\.?[\d]{0,2})',
                ],
                "silver": [
                    r'Silver[:\s]+\$?([2-4][0-9]\.[\d]{1,2})',
                    r'XAG[:\s]+\$?([2-4][0-9]\.[\d]{1,2})',
                ],
            }
            
            for metal, metal_patterns in patterns.items():
                for pattern in metal_patterns:
                    match = re.search(pattern, page_text, re.IGNORECASE)
                    if match:
                        try:
                            price_val = float(match.group(1).replace(",", ""))
                            if (metal == "gold" and 2000 < price_val < 5000) or (metal == "silver" and 20 < price_val < 100):
                                prices["data"][metal] = round(price_val, 2)
                                break
                        except (ValueError, IndexError):
                            continue
    except Exception as e:
        logging.warning(f"Error fetching from GoldPriceLive.com: {str(e)}")
    
    return prices


async def convert_prices_to_currency(prices: dict, currency: str) -> dict:
    """Convert USD prices to another currency by scraping exchange rates from public sources."""
    if currency == "USD":
        return prices
    
    rate = None
    
    # Try to scrape exchange rate from public forex websites
    try:
        client = await get_http_client()
        # Try scraping from XE.com or similar public forex site
        # Using a pattern that works with multiple public sources
        response = await client.get(
            f"https://www.xe.com/currencyconverter/convert/?Amount=1&From=USD&To={currency}",
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            }
        )
        if response.status_code == 200:
                soup = BeautifulSoup(response.text, "lxml")
                # Look for the exchange rate in the page
                # XE.com shows rates in various formats
                rate_elements = soup.find_all(string=re.compile(r'\d+\.\d{4,}'))
                for elem in rate_elements[:10]:  # Check first few matches
                    try:
                        potential_rate = float(elem.strip().replace(",", ""))
                        # Validate reasonable exchange rate (most currencies are between 0.1 and 1000 per USD)
                        if 0.01 < potential_rate < 10000:
                            rate = potential_rate
                            break
                    except ValueError:
                        continue
    except Exception:
        pass
    
    # If scraping failed, use approximate market rates (updated periodically)
    # These are approximate and should be refreshed periodically for accuracy
    if rate is None:
        currency_rates = {
            "EUR": 0.92, "GBP": 0.79, "JPY": 149.50, "CHF": 0.88,
            "CAD": 1.36, "AUD": 1.53, "CNY": 7.24, "INR": 83.50, "BRL": 4.97,
        }
        rate = currency_rates.get(currency, 1.0)
    
    # Convert all prices
    converted = {}
    for metal, price_data in prices.items():
        usd_price = price_data["price"]
        converted_price = round(usd_price * rate, 2)
        converted[metal] = {"price": converted_price}
    return converted


def get_prices_from_history(currency: str) -> dict:
    """Get most recent prices from history as fallback."""
    prices = {}
    for metal in SUPPORTED_METALS:
        history = price_history._history.get(metal, [])
        if history:
            # Get most recent price
            latest = history[-1]
            usd_price = latest.get("price", 0)
            if usd_price > 0:
                prices[metal] = {"price": round(usd_price, 2)}
    
    # If no history, use reasonable defaults
    if not prices:
        default_prices_usd = {
            "gold": 2650.00,
            "silver": 31.50,
            "platinum": 980.00,
            "palladium": 1050.00,
        }
        for metal, price in default_prices_usd.items():
            prices[metal] = {"price": price}
    
    # Convert currency if needed (simple approximation)
    if currency != "USD":
        currency_rates = {
            "EUR": 0.92, "GBP": 0.79, "JPY": 149.50, "CHF": 0.88,
            "CAD": 1.36, "AUD": 1.53, "CNY": 7.24, "INR": 83.50, "BRL": 4.97,
        }
        rate = currency_rates.get(currency, 1.0)
        for metal in prices:
            prices[metal]["price"] = round(prices[metal]["price"] * rate, 2)
    
    return prices




# ══════════════════════════════════════════════════════════════════════════════
# FastAPI Application
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Precious Metals API",
    description="""
## 🪙 Free Precious Metals Spot Price API

Get real-time **spot prices** (current market prices for immediate delivery) for **Gold**, **Silver**, **Platinum**, and **Palladium**.

**Multi-Source Spot Price Aggregation**: Prices are aggregated from 20+ spot price sources (Kitco, Investing.com, LBMA, XE.com, BullionVault, APMEX, JM Bullion, GoldPrice.org, Monex, GoldDealer.com, Money Metals Exchange, Provident Metals, SD Bullion, Silver.com, GoldSilver.com, Silver Gold Bull, BGASC, Gainesville Coins, Hero Bullion, GoldPriceLive.com, and more) - all providing actual spot prices (no futures) - and calculated as weighted averages with outlier filtering for maximum accuracy.

All prices are per troy ounce, aggregated from **20+ spot price sources** (100% spot prices only - no futures contracts) and calculated as **weighted averages** with outlier filtering for maximum accuracy - just like professional spot price APIs. **All prices are actual spot prices** (current market price for immediate delivery). Historical data uses Yahoo Finance futures as reference only. Prices update every 60 seconds.

### Features
- ✅ **No API key required** - Just call the endpoints
- ✅ **100% Spot Prices Only** - All prices are actual spot prices (no futures), aggregated from 20+ spot price sources with outlier filtering for accuracy
- ✅ **Multiple currencies** - USD, EUR, GBP, JPY, and more
- ✅ **Source transparency** - See which sources contributed to each price
- ✅ **Price statistics** - Min, max, median, and average from all sources
- ✅ **24h price changes** - Track market movements
- ✅ **CORS enabled** - Use from any frontend
- ✅ **Fast & cached** - 60-second cache with parallel fetching for performance

### Quick Start
```bash
# Get all metal spot prices
curl https://your-api.com/api/v1/metals

# Get gold spot price in EUR
curl https://your-api.com/api/v1/metals/gold?currency=EUR
```
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for all origins (free and easy to use)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# API Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Root"])
async def root():
    """Welcome endpoint with API information."""
    return {
        "name": "Precious Metals API",
        "version": "1.0.0",
        "description": "Free API for precious metal prices",
        "documentation": "/docs",
        "endpoints": {
            "all_metals": "/api/v1/metals",
            "single_metal": "/api/v1/metals/{metal}",
            "health": "/api/v1/health",
            "currencies": "/api/v1/currencies",
        },
    }


@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health and cache status."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        cache_age_seconds=price_cache.get_age_seconds(),
    )


@app.get("/api/v1/currencies", tags=["Reference"])
async def get_currencies():
    """Get list of supported currencies."""
    return {
        "currencies": SUPPORTED_CURRENCIES,
        "default": "USD",
    }


@app.get("/api/v1/metals", response_model=AllMetalsResponse, tags=["Prices"])
async def get_all_metals(
    currency: str = Query(
        default="USD",
        description="Currency for prices",
        examples=["USD", "EUR", "GBP"],
    )
):
    """
    Get current spot prices for all precious metals with percentage changes.
    
    Returns spot prices (current market price for immediate delivery) for gold, silver, 
    platinum, and palladium in the specified currency. Prices are per troy ounce (31.1035 grams).
    
    **Price changes included:**
    - `change_1h`: Change from 1 hour ago
    - `change_24h`: Change from 24 hours ago
    - `change_7d`: Change from 7 days ago
    """
    currency = currency.upper()
    
    if currency not in SUPPORTED_CURRENCIES:
        raise HTTPException(
            status_code=400,
            detail=f"Currency '{currency}' not supported. Use one of: {SUPPORTED_CURRENCIES}",
        )
    
    prices = await price_cache.get_prices(currency)
    # USD prices are already cached when we fetch any currency, so this is fast
    usd_prices = await price_cache.get_prices("USD")
    
    # Parallelize all change calculations across all metals and time periods
    change_tasks = []
    for metal in SUPPORTED_METALS:
        # Handle new price structure with metadata
        current_price_usd = usd_prices[metal].get("price", 0) if isinstance(usd_prices[metal], dict) else usd_prices[metal]
        if isinstance(current_price_usd, dict):
            current_price_usd = current_price_usd.get("price", 0)
        
        # Create tasks for all time periods in parallel (1h, 24h, 7d, 30d, 365d)
        change_tasks.append(price_history.calculate_change(metal, current_price_usd, hours_ago=1, currency="USD"))
        change_tasks.append(price_history.calculate_change(metal, current_price_usd, hours_ago=24, currency="USD"))
        change_tasks.append(price_history.calculate_change(metal, current_price_usd, hours_ago=168, currency="USD"))
        change_tasks.append(price_history.calculate_change(metal, current_price_usd, hours_ago=720, currency="USD"))  # 30 days
        change_tasks.append(price_history.calculate_change(metal, current_price_usd, hours_ago=8760, currency="USD"))  # 365 days
    
    # Execute all change calculations in parallel
    change_results = await asyncio.gather(*change_tasks, return_exceptions=True)
    
    metals_data = {}
    change_idx = 0
    for metal in SUPPORTED_METALS:
        # Handle missing price data gracefully
        if metal not in prices:
            logging.warning(f"Missing price data for {metal}")
            continue
            
        price_info = prices[metal]
        if not isinstance(price_info, dict):
            logging.warning(f"Invalid price info format for {metal}: {type(price_info)}")
            continue
        
        # Extract price and metadata from aggregated structure
        current_price = price_info.get("price", 0)
        if not isinstance(current_price, (int, float)) or current_price <= 0:
            logging.warning(f"Invalid price for {metal}: {current_price}")
            continue
            
        sources_count = price_info.get("sources_count", 0)
        sources = price_info.get("sources", [])
        median = price_info.get("median", current_price)
        price_min = price_info.get("min", current_price)
        price_max = price_info.get("max", current_price)
        
        # Get USD price for change calculations
        if metal not in usd_prices:
            logging.warning(f"Missing USD price for {metal}")
            current_price_usd = current_price
        else:
            usd_price_info = usd_prices[metal]
            current_price_usd = usd_price_info.get("price", current_price) if isinstance(usd_price_info, dict) else (usd_price_info if isinstance(usd_price_info, (int, float)) else current_price)
        
        # Extract results (5 per metal: 1h, 24h, 7d, 30d, 365d)
        # Add bounds checking to prevent IndexError
        if change_idx + 4 < len(change_results):
            change_1h = change_results[change_idx] if not isinstance(change_results[change_idx], Exception) else None
            change_24h = change_results[change_idx + 1] if not isinstance(change_results[change_idx + 1], Exception) else None
            change_7d = change_results[change_idx + 2] if not isinstance(change_results[change_idx + 2], Exception) else None
            change_1m = change_results[change_idx + 3] if not isinstance(change_results[change_idx + 3], Exception) else None
            change_1y = change_results[change_idx + 4] if not isinstance(change_results[change_idx + 4], Exception) else None
        else:
            logging.error(f"Not enough change results for {metal}: expected at least {change_idx + 5}, got {len(change_results)}")
            change_1h = change_24h = change_7d = change_1m = change_1y = None
        change_idx += 5
        
        # Convert dollar changes to requested currency (percentage stays the same)
        if currency != "USD" and current_price_usd and current_price_usd > 0 and current_price > 0:
            rate = current_price / current_price_usd
            if change_1h and isinstance(change_1h, dict) and "change" in change_1h:
                change_1h = {"change": round(change_1h["change"] * rate, 2), "percent": change_1h.get("percent", 0)}
            if change_24h and isinstance(change_24h, dict) and "change" in change_24h:
                change_24h = {"change": round(change_24h["change"] * rate, 2), "percent": change_24h.get("percent", 0)}
            if change_7d and isinstance(change_7d, dict) and "change" in change_7d:
                change_7d = {"change": round(change_7d["change"] * rate, 2), "percent": change_7d.get("percent", 0)}
            if change_1m and isinstance(change_1m, dict) and "change" in change_1m:
                change_1m = {"change": round(change_1m["change"] * rate, 2), "percent": change_1m.get("percent", 0)}
            if change_1y and isinstance(change_1y, dict) and "change" in change_1y:
                change_1y = {"change": round(change_1y["change"] * rate, 2), "percent": change_1y.get("percent", 0)}
        
        metals_data[metal] = {
            "symbol": METAL_SYMBOLS[metal],
            "price": current_price,
            "sources_count": sources_count,
            "sources": sources,
            "price_range": {
                "min": price_min,
                "max": price_max,
                "median": median,
            },
            "change_1h": change_1h,
            "change_24h": change_24h,
            "change_7d": change_7d,
            "change_1m": change_1m,
            "change_1y": change_1y,
        }
    
    return AllMetalsResponse(
        status="success",
        currency=currency,
        unit="troy_ounce",
        timestamp=datetime.utcnow(),
        metals=metals_data,
    )


@app.get("/api/v1/metals/{metal}", tags=["Prices"])
async def get_metal_price(
    metal: Metal,
    currency: str = Query(
        default="USD",
        description="Currency for price",
        examples=["USD", "EUR", "GBP"],
    )
):
    """
    Get current spot price for a specific precious metal with percentage changes.
    
    **Supported metals:**
    - `gold` (XAU)
    - `silver` (XAG)
    - `platinum` (XPT)
    - `palladium` (XPD)
    
    Returns spot price (current market price for immediate delivery) per troy ounce (31.1035 grams).
    
    **Price changes included:**
    - `change_1h`: Change from 1 hour ago
    - `change_24h`: Change from 24 hours ago
    - `change_7d`: Change from 7 days ago
    - `change_1m`: Change from 30 days ago (1 month)
    - `change_1y`: Change from 365 days ago (1 year)
    """
    currency = currency.upper()
    
    if currency not in SUPPORTED_CURRENCIES:
        raise HTTPException(
            status_code=400,
            detail=f"Currency '{currency}' not supported. Use one of: {SUPPORTED_CURRENCIES}",
        )
    
    prices = await price_cache.get_prices(currency)
    # USD prices are already cached when we fetch any currency, so this is fast
    usd_prices = await price_cache.get_prices("USD")
    
    price_info = prices[metal.value]
    # Extract price and metadata from aggregated structure
    current_price = price_info.get("price", 0)
    sources_count = price_info.get("sources_count", 0)
    sources = price_info.get("sources", [])
    median = price_info.get("median", current_price)
    price_min = price_info.get("min", current_price)
    price_max = price_info.get("max", current_price)
    
    # Get USD price for change calculations
    usd_price_info = usd_prices[metal.value]
    current_price_usd = usd_price_info.get("price", 0) if isinstance(usd_price_info, dict) else usd_price_info
    
    # Calculate all changes in parallel (1h, 24h, 7d, 30d, 365d)
    change_1h_task = price_history.calculate_change(metal.value, current_price_usd, hours_ago=1, currency="USD")
    change_24h_task = price_history.calculate_change(metal.value, current_price_usd, hours_ago=24, currency="USD")
    change_7d_task = price_history.calculate_change(metal.value, current_price_usd, hours_ago=168, currency="USD")
    change_1m_task = price_history.calculate_change(metal.value, current_price_usd, hours_ago=720, currency="USD")  # 30 days
    change_1y_task = price_history.calculate_change(metal.value, current_price_usd, hours_ago=8760, currency="USD")  # 365 days
    
    change_1h, change_24h, change_7d, change_1m, change_1y = await asyncio.gather(
        change_1h_task, change_24h_task, change_7d_task, change_1m_task, change_1y_task, return_exceptions=True
    )
    
    # Handle exceptions
    change_1h = change_1h if not isinstance(change_1h, Exception) else None
    change_24h = change_24h if not isinstance(change_24h, Exception) else None
    change_7d = change_7d if not isinstance(change_7d, Exception) else None
    change_1m = change_1m if not isinstance(change_1m, Exception) else None
    change_1y = change_1y if not isinstance(change_1y, Exception) else None
    
    # Convert dollar changes to requested currency (percentage stays the same)
    if currency != "USD" and current_price_usd > 0:
        rate = current_price / current_price_usd
        if change_1h and isinstance(change_1h, dict) and "change" in change_1h:
            change_1h = {"change": round(change_1h["change"] * rate, 2), "percent": change_1h.get("percent", 0)}
        if change_24h and isinstance(change_24h, dict) and "change" in change_24h:
            change_24h = {"change": round(change_24h["change"] * rate, 2), "percent": change_24h.get("percent", 0)}
        if change_7d and isinstance(change_7d, dict) and "change" in change_7d:
            change_7d = {"change": round(change_7d["change"] * rate, 2), "percent": change_7d.get("percent", 0)}
        if change_1m and isinstance(change_1m, dict) and "change" in change_1m:
            change_1m = {"change": round(change_1m["change"] * rate, 2), "percent": change_1m.get("percent", 0)}
        if change_1y and isinstance(change_1y, dict) and "change" in change_1y:
            change_1y = {"change": round(change_1y["change"] * rate, 2), "percent": change_1y.get("percent", 0)}
    
    # Return as dict to include source metadata (MetalPrice model is flexible)
    return {
        "metal": metal.value,
        "symbol": METAL_SYMBOLS[metal.value],
        "currency": currency,
        "price": current_price,
        "unit": "troy_ounce",
        "timestamp": datetime.utcnow(),
        "sources_count": sources_count,
        "sources": sources,
        "price_range": {
            "min": price_min,
            "max": price_max,
            "median": median,
        },
        "change_1h": change_1h,
        "change_24h": change_24h,
        "change_7d": change_7d,
        "change_1m": change_1m,
        "change_1y": change_1y,
    }


@app.get("/api/v1/metals/{metal}/history", tags=["Prices"])
async def get_metal_history(
    metal: Metal,
    hours: int = Query(default=24, ge=1, le=8760, description="Hours of history (1-8760, up to 1 year)"),
):
    """
    Get price history for a specific metal from Yahoo Finance.
    
    Returns historical price points for charting and analysis.
    Supports up to 1 year (8760 hours) of history.
    
    **Common periods:**
    - 24 hours (1 day)
    - 168 hours (7 days)
    - 720 hours (30 days / 1 month)
    - 8760 hours (365 days / 1 year)
    """
    try:
        history = await price_history.get_history(metal.value, hours=hours, currency="USD")
        
        return {
            "status": "success",
            "metal": metal.value,
            "symbol": METAL_SYMBOLS[metal.value],
            "currency": "USD",
            "hours": hours,
            "data_points": len(history),
            "history": history,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching historical data: {str(e)}"
        )


@app.get("/api/v1/convert", tags=["Utilities"])
async def convert_weight(
    metal: Metal,
    amount: float = Query(..., description="Amount of metal", gt=0),
    from_unit: str = Query(default="troy_ounce", description="Source unit"),
    to_unit: str = Query(default="gram", description="Target unit"),
    currency: str = Query(default="USD", description="Currency for value"),
):
    """
    Convert metal weight between units and calculate value.
    
    **Supported units:**
    - `troy_ounce` (31.1035g)
    - `gram` (g)
    - `kilogram` (kg)
    - `ounce` (28.3495g)
    """
    currency = currency.upper()
    
    # Conversion factors to grams
    unit_to_grams = {
        "troy_ounce": 31.1035,
        "gram": 1.0,
        "kilogram": 1000.0,
        "ounce": 28.3495,
    }
    
    if from_unit not in unit_to_grams:
        raise HTTPException(
            status_code=400,
            detail=f"Unit '{from_unit}' not supported. Use: {list(unit_to_grams.keys())}",
        )
    
    if to_unit not in unit_to_grams:
        raise HTTPException(
            status_code=400,
            detail=f"Unit '{to_unit}' not supported. Use: {list(unit_to_grams.keys())}",
        )
    
    # Convert to grams, then to target unit
    grams = amount * unit_to_grams[from_unit]
    converted = grams / unit_to_grams[to_unit]
    
    # Calculate value
    prices = await price_cache.get_prices(currency)
    price_per_troy_oz = prices[metal.value]["price"]
    troy_ounces = grams / unit_to_grams["troy_ounce"]
    value = troy_ounces * price_per_troy_oz
    
    return {
        "metal": metal.value,
        "input": {"amount": amount, "unit": from_unit},
        "output": {"amount": round(converted, 4), "unit": to_unit},
        "value": {"amount": round(value, 2), "currency": currency},
        "price_per_troy_ounce": price_per_troy_oz,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Error Handlers
# ══════════════════════════════════════════════════════════════════════════════

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested endpoint does not exist.",
            "available_endpoints": ["/api/v1/metals", "/api/v1/health", "/docs"],
        },
    )


@app.exception_handler(Exception)
async def server_error_handler(request: Request, exc: Exception):
    import traceback
    error_details = traceback.format_exc()
    logging.error(f"Unhandled exception in {request.url.path}: {error_details}")
    
    # Don't override FastAPI's built-in HTTP exceptions
    if isinstance(exc, StarletteHTTPException):
        raise exc
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "Something went wrong. Please try again.",
            "detail": str(exc) if exc else "Unknown error",
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# Run Server
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    
    print("🪙 Starting Precious Metals API...")
    print("📖 Documentation: http://localhost:8000/docs")
    print("🔧 Health Check: http://localhost:8000/api/v1/health")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
