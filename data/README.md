# Local market data

Market CSV files are not committed to Git. Each input file must contain:

| Column | Meaning | Example |
|---|---|---|
| `date` | Trading date in `YYYYMMDD` format | `20260204` |
| `minute_end` | Minute ending time in `HHMMSS` format | `091600` |
| `symbol` | NIFTY future or option symbol | `NIFTY2621025750CE` |
| `last_trade_price` | Raw price in paise | `15300` |

The loader divides `last_trade_price` by 100. Option symbols are expected to
end in a five-digit strike followed by `CE` or `PE`. The current symbol family
encodes expiry as two-digit year, one-digit month, and two-digit day.
