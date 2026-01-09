# 🪙 Precious Metals API

A **free and easy-to-use** REST API for real-time precious metal prices. No API key required!

## Data Sources

Prices are aggregated from **20+ spot price sources** for maximum accuracy:

- **Kitco** - Real-time spot prices (all metals)
- **Investing.com** - Gold & Silver spot prices (XAU/USD, XAG/USD)
- **LBMA** - London Bullion Market Association spot prices
- **XE.com** - Currency and commodity spot prices (all metals)
- **BullionVault** - Precious metals dealer spot prices
- **APMEX** - American Precious Metals Exchange spot prices
- **JM Bullion** - Precious metals dealer spot prices
- **GoldPrice.org** - Precious metals spot prices
- **Monex** - Precious metals dealer spot prices
- **GoldDealer.com** - Precious metals dealer spot prices
- **Money Metals Exchange** - Precious metals dealer spot prices
- **Provident Metals** - Precious metals dealer spot prices
- **SD Bullion** - Precious metals dealer spot prices (all metals)
- **Silver.com** - Gold & Silver spot prices
- **GoldSilver.com** - Gold & Silver spot prices
- **Silver Gold Bull** - Gold & Silver spot prices
- **BGASC** - Precious metals dealer spot prices
- **Gainesville Coins** - Precious metals dealer spot prices (all metals)
- **Hero Bullion** - Precious metals dealer spot prices
- **GoldPriceLive.com** - Real-time precious metals spot prices

**All prices are actual spot prices** (current market price for immediate delivery), **not futures contracts**.

The API:
- Fetches prices from all sources in parallel
- Calculates weighted averages with **outlier filtering** (IQR method)
- Returns aggregated prices with source transparency (see which sources contributed)
- Provides price statistics (min, max, median, average)

Prices update every 60 seconds and are cached for performance.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Server

```bash
python main.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Get All Metal Prices

```bash
GET /api/v1/metals
GET /api/v1/metals?currency=EUR
```

**Response:**
```json
{
  "status": "success",
  "currency": "USD",
  "unit": "troy_ounce",
  "timestamp": "2026-01-09T12:00:00Z",
  "metals": {
    "gold": {
      "symbol": "XAU",
      "price": 2680.50,
      "sources_count": 8,
      "sources": [
        "Kitco (Spot)",
        "Investing.com (Spot)",
        "LBMA (Spot)",
        "XE.com (Spot)",
        "BullionVault (Spot)",
        "APMEX (Spot)",
        "JM Bullion (Spot)",
        "GoldPrice.org (Spot)"
      ],
      "price_range": {
        "min": 2678.20,
        "max": 2682.75,
        "median": 2680.45
      },
      "change_1h": { "change": 2.50, "percent": 0.09 },
      "change_24h": { "change": 12.50, "percent": 0.47 },
      "change_7d": { "change": -35.00, "percent": -1.30 },
      "change_1m": { "change": 85.50, "percent": 3.29 },
      "change_1y": { "change": 320.75, "percent": 13.59 }
    },
    "silver": {
      "symbol": "XAG",
      "price": 31.25,
      "sources_count": 7,
      "sources": [
        "Kitco (Spot)",
        "Investing.com (Spot)",
        "LBMA (Spot)",
        "XE.com (Spot)",
        "BullionVault (Spot)",
        "APMEX (Spot)",
        "JM Bullion (Spot)"
      ],
      "price_range": {
        "min": 31.18,
        "max": 31.32,
        "median": 31.24
      },
      "change_1h": { "change": 0.08, "percent": 0.25 },
      "change_24h": { "change": 0.35, "percent": 1.12 },
      "change_7d": { "change": 0.75, "percent": 2.44 },
      "change_1m": { "change": 1.25, "percent": 4.16 },
      "change_1y": { "change": 5.50, "percent": 21.35 }
    },
    "platinum": {
      "symbol": "XPT",
      "price": 985.00,
      "sources_count": 5,
      "sources": [
        "Kitco (Spot)",
        "XE.com (Spot)",
        "APMEX (Spot)",
        "JM Bullion (Spot)",
        "Monex (Spot)"
      ],
      "price_range": {
        "min": 983.50,
        "max": 986.75,
        "median": 984.90
      },
      "change_1h": { "change": -2.10, "percent": -0.21 },
      "change_24h": { "change": -8.20, "percent": -0.83 },
      "change_7d": { "change": 15.00, "percent": 1.56 },
      "change_1m": { "change": -25.50, "percent": -2.52 },
      "change_1y": { "change": 120.00, "percent": 13.87 }
    },
    "palladium": {
      "symbol": "XPD",
      "price": 1025.00,
      "sources_count": 4,
      "sources": [
        "Kitco (Spot)",
        "XE.com (Spot)",
        "APMEX (Spot)",
        "JM Bullion (Spot)"
      ],
      "price_range": {
        "min": 1023.25,
        "max": 1026.50,
        "median": 1025.10
      },
      "change_1h": { "change": 3.50, "percent": 0.33 },
      "change_24h": { "change": 15.00, "percent": 1.45 },
      "change_7d": { "change": -22.00, "percent": -2.05 },
      "change_1m": { "change": 45.00, "percent": 4.59 },
      "change_1y": { "change": -150.00, "percent": -12.77 }
    }
  }
}
```

### Get Single Metal Price

```bash
GET /api/v1/metals/gold
GET /api/v1/metals/silver?currency=GBP
```

**Supported metals:** `gold`, `silver`, `platinum`, `palladium`

**Response:**
```json
{
  "metal": "gold",
  "symbol": "XAU",
  "currency": "USD",
  "price": 2680.50,
  "unit": "troy_ounce",
  "timestamp": "2026-01-09T12:00:00Z",
  "sources_count": 8,
  "sources": [
    "Kitco (Spot)",
    "Investing.com (Spot)",
    "LBMA (Spot)",
    "XE.com (Spot)",
    "BullionVault (Spot)",
    "APMEX (Spot)",
    "JM Bullion (Spot)",
    "GoldPrice.org (Spot)"
  ],
  "price_range": {
    "min": 2678.20,
    "max": 2682.75,
    "median": 2680.45
  },
  "change_1h": { "change": 2.50, "percent": 0.09 },
  "change_24h": { "change": 12.50, "percent": 0.47 },
  "change_7d": { "change": -35.00, "percent": -1.30 },
  "change_1m": { "change": 85.50, "percent": 3.29 },
  "change_1y": { "change": 320.75, "percent": 13.59 }
}
```

### Get Price History

```bash
GET /api/v1/metals/gold/history
GET /api/v1/metals/gold/history?hours=48
```

**Response:**
```json
{
  "metal": "gold",
  "symbol": "XAU",
  "currency": "USD",
  "hours": 24,
  "data_points": 1440,
  "history": [
    { "timestamp": "2026-01-08T12:00:00Z", "price": 2675.50 },
    { "timestamp": "2026-01-08T12:01:00Z", "price": 2676.20 }
  ]
}
```

### Convert Weight & Calculate Value

```bash
GET /api/v1/convert?metal=gold&amount=100&from_unit=gram&to_unit=troy_ounce
```

**Supported units:** `troy_ounce`, `gram`, `kilogram`, `ounce`

### Health Check

```bash
GET /api/v1/health
```

### List Supported Currencies

```bash
GET /api/v1/currencies
```

## Supported Currencies

| Code | Currency |
|------|----------|
| USD | US Dollar |
| EUR | Euro |
| GBP | British Pound |
| JPY | Japanese Yen |
| CHF | Swiss Franc |
| CAD | Canadian Dollar |
| AUD | Australian Dollar |
| CNY | Chinese Yuan |
| INR | Indian Rupee |
| BRL | Brazilian Real |

## Integration Examples

### JavaScript / Fetch

```javascript
// Get all metal prices with percentage changes
const response = await fetch('https://your-api.com/api/v1/metals');
const data = await response.json();

const gold = data.metals.gold;
console.log(`Gold: $${gold.price}`);
console.log(`  Sources: ${gold.sources_count} (${gold.sources.join(', ')})`);
console.log(`  Price range: $${gold.price_range.min} - $${gold.price_range.max} (median: $${gold.price_range.median})`);
console.log(`  1h change: ${gold.change_1h?.percent ?? 0}%`);
console.log(`  24h change: ${gold.change_24h?.percent ?? 0}%`);
console.log(`  7d change: ${gold.change_7d?.percent ?? 0}%`);
console.log(`  1m change: ${gold.change_1m?.percent ?? 0}%`);
console.log(`  1y change: ${gold.change_1y?.percent ?? 0}%`);

// Get gold price history for charting
const historyRes = await fetch('https://your-api.com/api/v1/metals/gold/history?hours=24');
const historyData = await historyRes.json();
console.log(`Data points: ${historyData.data_points}`);
```

### Python

```python
import requests

# Get all metal prices with percentage changes
response = requests.get('https://your-api.com/api/v1/metals')
data = response.json()

for metal, info in data['metals'].items():
    change_24h = info['change_24h']['percent'] if info['change_24h'] else 0
    change_1y = info['change_1y']['percent'] if info['change_1y'] else 0
    print(f"{metal.upper()}: ${info['price']} ({change_24h:+.2f}% 24h, {change_1y:+.2f}% 1y)")
    print(f"  Sources: {info['sources_count']} ({', '.join(info['sources'][:3])}...)")
    print(f"  Range: ${info['price_range']['min']} - ${info['price_range']['max']}")

# Get price history for a specific metal
history = requests.get('https://your-api.com/api/v1/metals/gold/history?hours=48').json()
print(f"Gold history: {history['data_points']} data points")
```

### cURL

```bash
# All metals with percentage changes
curl https://your-api.com/api/v1/metals

# Gold in EUR
curl "https://your-api.com/api/v1/metals/gold?currency=EUR"

# Get 48 hours of gold price history
curl "https://your-api.com/api/v1/metals/gold/history?hours=48"

# Convert 100 grams of silver
curl "https://your-api.com/api/v1/convert?metal=silver&amount=100&from_unit=gram&to_unit=troy_ounce"
```

### React Hook Example

```jsx
import { useState, useEffect } from 'react';

function useMetalPrices(currency = 'USD') {
  const [prices, setPrices] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchPrices = async () => {
      try {
        const res = await fetch(
          `https://your-api.com/api/v1/metals?currency=${currency}`
        );
        const data = await res.json();
        setPrices(data.metals);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchPrices();
    const interval = setInterval(fetchPrices, 60000); // Refresh every minute
    return () => clearInterval(interval);
  }, [currency]);

  return { prices, loading, error };
}

// Usage in component
function PriceCard({ metal, data }) {
  const change24h = data.change_24h?.percent ?? 0;
  const change1y = data.change_1y?.percent ?? 0;
  const isUp = change24h >= 0;
  
  return (
    <div className="price-card">
      <h3>{metal.toUpperCase()}</h3>
      <p className="price">${data.price.toLocaleString()}</p>
      <p className="sources">{data.sources_count} sources: {data.sources.slice(0, 2).join(', ')}...</p>
      <p className={isUp ? 'up' : 'down'}>
        {isUp ? '▲' : '▼'} {Math.abs(change24h)}% (24h)
      </p>
      <p className="range">
        Range: ${data.price_range.min} - ${data.price_range.max}
      </p>
      <p className="yearly">
        1 Year: {change1y >= 0 ? '+' : ''}{change1y.toFixed(2)}%
      </p>
    </div>
  );
}
```

## Features

- ✅ **No API Key** - Completely free, no registration required
- ✅ **100% Spot Prices Only** - All prices are actual spot prices (no futures contracts), aggregated from 20+ spot price sources with outlier filtering for accuracy
- ✅ **Multi-Source Aggregation** - Combines 20+ spot price sources for maximum accuracy
- ✅ **Outlier Filtering** - Uses IQR (Interquartile Range) method to filter outliers
- ✅ **Source Transparency** - See which sources contributed to each price
- ✅ **Price Statistics** - Min, max, median, and average prices from all sources
- ✅ **Real-Time Prices** - Live spot prices updated every 60 seconds
- ✅ **CORS Enabled** - Works from any frontend
- ✅ **Multiple Currencies** - 10 major currencies supported
- ✅ **Unit Conversion** - Convert between weight units
- ✅ **Price Change Tracking** - 1h, 24h, 7d, 1 month, and 1 year percentage changes
- ✅ **Price History** - Up to 1 year of historical data from Yahoo Finance
- ✅ **Fast Caching** - 60-second cache with parallel fetching for performance
- ✅ **Connection Pooling** - Shared HTTP client for optimal performance
- ✅ **OpenAPI Docs** - Interactive docs at `/docs`

## API Documentation

Once running, visit:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## Deployment

### Heroku / DigitalOcean

The repo includes:
- `runtime.txt` - Specifies Python 3.11
- `Procfile` - Defines the start command
- `requirements.txt` - Lists dependencies

Just push to your platform of choice.

### Using Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .
COPY price_history.json .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t metals-api .
docker run -p 8000:8000 metals-api
```

## How It Works

The API uses a **professional multi-source aggregation** approach:

1. **Parallel Fetching**: Queries all 20+ spot price sources simultaneously
2. **Data Collection**: Gathers prices from each successful source
3. **Outlier Filtering**: Removes outliers using IQR (Interquartile Range) method
4. **Price Calculation**: Calculates weighted average from filtered prices
5. **Statistics**: Provides min, max, median, and average from all sources
6. **Transparency**: Shows which sources contributed to the final price

This ensures maximum accuracy and reliability, just like professional spot price APIs.

## Price Calculation Details

- **Primary Price**: Weighted average of all successful sources (with outliers removed)
- **Change Calculations**: Uses aggregated average price vs Yahoo Finance last closed price
- **Historical Reference**: Yahoo Finance futures (for historical data only, not current prices)
- **Outlier Detection**: IQR method - removes prices outside 1.5 × IQR range
- **Source Count**: Number of sources that successfully returned prices
- **Price Range**: Min, max, and median from all contributing sources

## License

MIT - Use freely in your projects!
