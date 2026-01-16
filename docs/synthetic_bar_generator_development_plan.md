# Synthetic Bar Generator Development Plan
## 24-Hour Rolling Forecast System

---

# Document Overview

This development plan provides step-by-step implementation guidance for building a production-grade synthetic bar generator that forecasts 24 hours ahead continuously. The system progresses through statistical model phases (GARCH → A-GARCH → Heston → Jump-Diffusion → SVCJ) before implementing agent-based path generation.

**Reference Document:** `monte_carlo_architecture.md` contains theoretical foundations, parameter specifications, and validation frameworks referenced throughout this plan.

**System Output:** 10,000 simulated price paths updated every minute, producing probability distributions for support/resistance zones, confidence metrics, and position management signals.

**Timeline Relationship:** This development plan aligns with the 30-week roadmap outlined in `monte_carlo_architecture.md` Part VI (Development Roadmap):
- Phase 1 (GARCH MVP): Weeks 1-4
- Phase 2 (Statistical Enhancements): Weeks 5-10 (A-GARCH: 5-6, Heston: 7-8, Jump-Diffusion: 9-10)
- Phase 3 (Trading Signal Layer): Weeks 11-14
- Phase 4 (Agent-Based System): Weeks 15-24 (deferred to Phase 8 in this plan)
- Phase 5 (Production & Comparison): Weeks 25-30

**Note:** Agent-based implementation (Phase 8) aligns with architecture doc Phase 4 but is deferred in this plan until statistical models are complete and deployed, as per architecture doc recommendation.

---

# Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Data Infrastructure](#2-data-infrastructure)
3. [Phase 1: GARCH MVP](#3-phase-1-garch-mvp)
4. [Phase 2: A-GARCH Enhancement](#4-phase-2-a-garch-enhancement)
5. [Phase 3: Heston Stochastic Volatility](#5-phase-3-heston-stochastic-volatility)
6. [Phase 4: Jump-Diffusion](#6-phase-4-jump-diffusion)
7. [Phase 5: SVCJ Combined Model](#7-phase-5-svcj-combined-model)
8. [Phase 6: Position Management System](#8-phase-6-position-management-system)
9. [Phase 7: Production Deployment](#9-phase-7-production-deployment)
10. [Phase 8: Agent-Based Implementation](#10-phase-8-agent-based-implementation)
11. [Testing Protocols](#11-testing-protocols)
12. [Monitoring and Maintenance](#12-monitoring-and-maintenance)

---

# 1. System Requirements

## 1.1 Hardware Specifications

### Development Environment
```
CPU: 8+ cores (parallel simulation benefits)
RAM: 32GB minimum (10,000 particles × 4 instruments × state data)
Storage: 500GB SSD (historical data, logs, checkpoints)
GPU: Optional but recommended (CUDA-capable for future optimization)
```

### Production Environment
```
CPU: 16+ cores or cloud equivalent
RAM: 64GB (headroom for spikes, multiple model versions)
Storage: 1TB SSD (extended history, model artifacts, logs)
Network: Low-latency connection to data feed
Redundancy: Hot standby instance recommended
```

## 1.2 Software Stack

### Core Languages
```
Python 3.11+    Primary implementation language
Cython          Performance-critical inner loops (optional Phase 7+)
```

### Required Packages
```
# Data handling
numpy>=1.24.0
pandas>=2.0.0
polars>=0.19.0          # Fast DataFrame operations for production

# Statistical modeling
scipy>=1.11.0
statsmodels>=0.14.0     # GARCH implementation baseline
arch>=6.0.0             # Production GARCH/A-GARCH

# Numerical methods
numba>=0.58.0           # JIT compilation for simulation loops

# Data feeds
websocket-client>=1.6.0
asyncio                 # Built-in

# Storage
sqlite3                 # Built-in, development
redis>=5.0.0            # Production state management
timescaledb             # Production time-series storage (via psycopg2)

# Monitoring
prometheus-client>=0.17.0
logging                 # Built-in

# Testing
pytest>=7.4.0
hypothesis>=6.82.0      # Property-based testing
```

## 1.3 Data Feed Requirements

### Real-Time Data
```
Source: CME Group direct feed, or broker API (Interactive Brokers, etc.)
Instruments: ES, NQ, GC, SI (front-month contracts)
Fields required:
  - Timestamp (millisecond precision minimum)
  - Bid price
  - Ask price
  - Last trade price
  - Last trade size
  - Volume (cumulative and per-bar)

Update frequency: Tick-by-tick or 1-second bars minimum
Latency requirement: <500ms from exchange to system
```

### Historical Data
```
Granularity: 1-minute OHLCV bars
History depth: 2+ years per instrument (for calibration stability)
Fields required:
  - Open, High, Low, Close
  - Volume
  - Number of trades (if available)
  
Additional (for validation):
  - Tick data for select periods (volatility signature validation)
```

## 1.4 Directory Structure

```
synthetic_bar_generator/
├── config/
│   ├── instruments.yaml          # Per-instrument parameters
│   ├── model_params.yaml         # Model hyperparameters
│   ├── thresholds.yaml           # Trading signal thresholds
│   └── logging.yaml              # Logging configuration
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── feed_handler.py       # Real-time data ingestion
│   │   ├── historical_loader.py  # Historical data loading
│   │   ├── bar_aggregator.py     # Tick-to-bar conversion
│   │   └── storage.py            # Database interactions
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py         # Abstract base class
│   │   ├── garch.py              # GARCH implementation
│   │   ├── agarch.py             # Asymmetric GARCH
│   │   ├── heston.py             # Heston stochastic volatility
│   │   ├── jump_diffusion.py     # Merton jump-diffusion
│   │   ├── svcj.py               # Combined SVCJ model
│   │   └── calibration.py        # Parameter estimation
│   │
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── path_generator.py     # Monte Carlo path generation
│   │   ├── particle_filter.py    # Bayesian updating engine
│   │   └── resampler.py          # Systematic resampling
│   │
│   ├── signals/
│   │   ├── __init__.py
│   │   ├── distribution_analyzer.py  # Extract S/R zones
│   │   ├── confidence_calculator.py  # Std dev metrics
│   │   └── position_manager.py       # Entry/exit logic
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── time_utils.py         # Session handling, timezone, session effects
                                        # Reference: monte_carlo_architecture.md Sections 10.3-13.3
│   │   ├── math_utils.py         # Statistical functions
│   │   └── validation.py         # Input validation
│   │
│   └── main.py                   # Application entry point
│
├── tests/
│   ├── unit/
│   │   ├── test_garch.py
│   │   ├── test_particle_filter.py
│   │   └── ...
│   ├── integration/
│   │   ├── test_full_pipeline.py
│   │   └── ...
│   └── validation/
│       ├── test_stylized_facts.py
│       └── test_forecast_calibration.py
│
├── scripts/
│   ├── calibrate_models.py       # Offline calibration
│   ├── backtest.py               # Historical validation
│   └── generate_reports.py       # Performance reports
│
├── data/
│   ├── historical/               # Historical bar data
│   ├── calibration/              # Saved model parameters
│   └── logs/                     # Application logs
│
├── notebooks/
│   ├── exploration/              # Research notebooks
│   └── validation/               # Validation analysis
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yaml
│
├── requirements.txt
├── setup.py
└── README.md
```

---

# 2. Data Infrastructure

## 2.1 Historical Data Pipeline

### 2.1.1 Data Acquisition

**Task:** Obtain 2+ years of 1-minute OHLCV data for ES, NQ, GC, SI.

**Sources (in order of preference):**
1. CME DataMine (official, highest quality)
2. Broker historical data API (Interactive Brokers, etc.)
3. Third-party vendors (Polygon.io, Databento)

**Implementation Steps:**

```python
# historical_loader.py

class HistoricalDataLoader:
    """
    Loads and validates historical bar data for model calibration.
    """
    
    REQUIRED_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
    def load_instrument(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load historical data for a single instrument.
        
        Args:
            symbol: Instrument symbol (ES, NQ, GC, SI)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with OHLCV data indexed by timestamp
        """
        # Implementation details:
        # 1. Load from parquet/CSV files
        # 2. Validate column presence
        # 3. Handle missing bars (market closures, holidays)
        # 4. Convert timestamps to UTC
        # 5. Validate OHLC relationships (high >= open, close, low)
        pass
    
    def validate_data_quality(self, df: pd.DataFrame, symbol: str) -> dict:
        """
        Run data quality checks and return report.
        
        Checks:
        - Missing bar percentage (should be <1% during RTH)
        - OHLC validity (high >= all, low <= all)
        - Volume anomalies (zero volume bars)
        - Price continuity (gaps > 5% flagged)
        - Timestamp continuity
        """
        pass
```

### 2.1.2 Data Storage Schema

**Development (SQLite):**

```sql
-- Historical bars table
CREATE TABLE bars (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timestamp INTEGER NOT NULL,  -- Unix timestamp (seconds)
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume INTEGER NOT NULL,
    trade_count INTEGER,
    UNIQUE(symbol, timestamp)
);

CREATE INDEX idx_bars_symbol_timestamp ON bars(symbol, timestamp);

-- Calibration parameters table
CREATE TABLE calibration_params (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    model_type TEXT NOT NULL,  -- 'garch', 'agarch', 'heston', etc.
    param_name TEXT NOT NULL,
    param_value REAL NOT NULL,
    calibration_date INTEGER NOT NULL,
    window_start INTEGER NOT NULL,
    window_end INTEGER NOT NULL,
    UNIQUE(symbol, model_type, param_name, calibration_date)
);
```

**Production (TimescaleDB):**

```sql
-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Bars hypertable
CREATE TABLE bars (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume BIGINT NOT NULL,
    trade_count INTEGER
);

SELECT create_hypertable('bars', 'timestamp');
CREATE INDEX idx_bars_symbol ON bars(symbol, timestamp DESC);

-- Particle state snapshots (for recovery)
CREATE TABLE particle_snapshots (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    snapshot_data BYTEA NOT NULL,  -- Compressed particle array
    model_version TEXT NOT NULL
);

SELECT create_hypertable('particle_snapshots', 'timestamp');
```

## 2.2 Real-Time Data Pipeline

### 2.2.1 Feed Handler Architecture

```python
# feed_handler.py

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Optional
from datetime import datetime
import logging

@dataclass
class Tick:
    """Single market data tick."""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float
    last_size: int
    volume: int

@dataclass  
class Bar:
    """Aggregated OHLCV bar."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    trade_count: int
    vwap: float

class FeedHandler(ABC):
    """Abstract base class for market data feeds."""
    
    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to data source."""
        pass
    
    @abstractmethod
    async def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to market data for symbols."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Clean disconnection."""
        pass
    
    @abstractmethod
    def register_callback(self, callback: Callable[[Tick], None]) -> None:
        """Register callback for tick updates."""
        pass


class BarAggregator:
    """
    Aggregates ticks into 1-minute bars with precise boundary handling.
    """
    
    def __init__(self, bar_callback: Callable[[Bar], None]):
        self.bar_callback = bar_callback
        self.current_bars: Dict[str, dict] = {}
        self.last_bar_minute: Dict[str, int] = {}
        
    def process_tick(self, tick: Tick) -> Optional[Bar]:
        """
        Process incoming tick, emit bar if minute boundary crossed.
        
        Implementation details:
        1. Determine current minute boundary
        2. If new minute:
           a. Finalize previous bar
           b. Emit via callback
           c. Initialize new bar with this tick
        3. If same minute:
           a. Update high/low
           b. Update close
           c. Accumulate volume
        """
        current_minute = tick.timestamp.replace(second=0, microsecond=0)
        symbol = tick.symbol
        
        # Check if this is a new bar
        if symbol not in self.current_bars:
            # First tick for this symbol
            self._init_bar(symbol, tick, current_minute)
            return None
            
        if current_minute > self.last_bar_minute[symbol]:
            # New minute - finalize and emit previous bar
            completed_bar = self._finalize_bar(symbol)
            self._init_bar(symbol, tick, current_minute)
            self.bar_callback(completed_bar)
            return completed_bar
        else:
            # Same minute - update current bar
            self._update_bar(symbol, tick)
            return None
    
    def _init_bar(self, symbol: str, tick: Tick, minute: datetime) -> None:
        """Initialize new bar from tick."""
        mid_price = (tick.bid + tick.ask) / 2
        self.current_bars[symbol] = {
            'open': mid_price,
            'high': mid_price,
            'low': mid_price,
            'close': mid_price,
            'volume': tick.last_size,
            'trade_count': 1,
            'vwap_numerator': mid_price * tick.last_size,
            'timestamp': minute
        }
        self.last_bar_minute[symbol] = minute
        
    def _update_bar(self, symbol: str, tick: Tick) -> None:
        """Update current bar with new tick."""
        bar = self.current_bars[symbol]
        mid_price = (tick.bid + tick.ask) / 2
        
        bar['high'] = max(bar['high'], mid_price)
        bar['low'] = min(bar['low'], mid_price)
        bar['close'] = mid_price
        bar['volume'] += tick.last_size
        bar['trade_count'] += 1
        bar['vwap_numerator'] += mid_price * tick.last_size
        
    def _finalize_bar(self, symbol: str) -> Bar:
        """Create Bar object from accumulated data."""
        bar = self.current_bars[symbol]
        vwap = bar['vwap_numerator'] / bar['volume'] if bar['volume'] > 0 else bar['close']
        
        return Bar(
            symbol=symbol,
            timestamp=bar['timestamp'],
            open=bar['open'],
            high=bar['high'],
            low=bar['low'],
            close=bar['close'],
            volume=bar['volume'],
            trade_count=bar['trade_count'],
            vwap=vwap
        )
```

### 2.2.2 Feed Handler Implementations

**Interactive Brokers Implementation:**

```python
# feed_handler_ib.py

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import threading

class IBFeedHandler(FeedHandler, EWrapper, EClient):
    """
    Interactive Brokers TWS/Gateway feed handler.
    """
    
    SYMBOL_TO_CONTRACT = {
        'ES': {'symbol': 'ES', 'exchange': 'CME', 'secType': 'FUT'},
        'NQ': {'symbol': 'NQ', 'exchange': 'CME', 'secType': 'FUT'},
        'GC': {'symbol': 'GC', 'exchange': 'COMEX', 'secType': 'FUT'},
        'SI': {'symbol': 'SI', 'exchange': 'COMEX', 'secType': 'FUT'},
    }
    
    def __init__(self, host: str = '127.0.0.1', port: int = 7497, client_id: int = 1):
        EWrapper.__init__(self)
        EClient.__init__(self, wrapper=self)
        
        self.host = host
        self.port = port
        self.client_id = client_id
        self.tick_callbacks: list[Callable] = []
        self.request_id_to_symbol: Dict[int, str] = {}
        self.next_request_id = 1
        
    async def connect(self) -> None:
        """Connect to TWS/Gateway."""
        self.connect(self.host, self.port, self.client_id)
        
        # Start message processing thread
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
        
        # Wait for connection confirmation
        await asyncio.sleep(2)
        
    async def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to market data for symbols."""
        for symbol in symbols:
            contract = self._create_contract(symbol)
            req_id = self.next_request_id
            self.next_request_id += 1
            self.request_id_to_symbol[req_id] = symbol
            
            # Request market data
            self.reqMktData(req_id, contract, '', False, False, [])
            
    def _create_contract(self, symbol: str) -> Contract:
        """Create IB contract for symbol."""
        spec = self.SYMBOL_TO_CONTRACT[symbol]
        contract = Contract()
        contract.symbol = spec['symbol']
        contract.secType = spec['secType']
        contract.exchange = spec['exchange']
        contract.currency = 'USD'
        # Note: Need to set lastTradeDateOrContractMonth for front month
        return contract
    
    # EWrapper callbacks
    def tickPrice(self, reqId, tickType, price, attrib):
        """Handle price tick from IB."""
        # Convert to Tick object and dispatch
        pass
    
    def tickSize(self, reqId, tickType, size):
        """Handle size tick from IB."""
        pass
```

### 2.2.3 Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA FLOW                                      │
└─────────────────────────────────────────────────────────────────────────┘

Exchange (CME)
      │
      ▼
┌──────────────┐
│ Feed Handler │ ◄── Tick-by-tick data
└──────────────┘
      │
      ▼
┌────────────────┐
│ Bar Aggregator │ ◄── Aggregates to 1-minute bars
└────────────────┘
      │
      ├────────────────────────────────────────┐
      ▼                                        ▼
┌──────────────┐                    ┌─────────────────────┐
│   Storage    │                    │   Simulation Core   │
│  (Database)  │                    │  (Particle Filter)  │
└──────────────┘                    └─────────────────────┘
                                              │
                                              ▼
                                    ┌─────────────────────┐
                                    │   Signal Generator  │
                                    └─────────────────────┘
                                              │
                                              ▼
                                    ┌─────────────────────┐
                                    │  Position Manager   │
                                    └─────────────────────┘
```

---

# 3. Phase 1: GARCH MVP

**Reference:** monte_carlo_architecture.md Part IV (Development Roadmap) - Phase 1: GARCH MVP (Weeks 1-4)

## 3.1 Objectives

1. Implement basic GARCH(1,1) path generation (architecture doc Section 4.2)
2. Build particle filter infrastructure (architecture doc Part III, Section 6)
3. Establish signal extraction pipeline (architecture doc Part IV, Section 7)
4. Validate against stylized facts (architecture doc Part VII, Section 15)
5. Create baseline performance metrics

## 3.2 GARCH Model Implementation

### 3.2.1 Core GARCH Class

```python
# models/garch.py

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Optional, Tuple
from abc import ABC, abstractmethod

@dataclass
class GARCHParams:
    """
    GARCH(1,1) parameters.
    
    Reference: monte_carlo_architecture.md Section 4.2 for parameter specifications.
    """
    omega: float      # Baseline variance constant
    alpha: float      # Reaction to shocks (ARCH term)
    beta: float       # Persistence (GARCH term)
    mu: float = 0.0   # Drift term
    nu: float = 2.0   # GED shape parameter (nu > 1, lower = fatter tails)
    skew: float = 0.0 # GED skewness parameter (-1 to 1, 0 = symmetric)
    
    def validate(self) -> bool:
        """Check parameter constraints."""
        # omega > 0
        if self.omega <= 0:
            return False
        # alpha >= 0, beta >= 0
        if self.alpha < 0 or self.beta < 0:
            return False
        # alpha + beta < 1 (stationarity)
        if self.alpha + self.beta >= 1:
            return False
        return True
    
    @property
    def unconditional_variance(self) -> float:
        """Long-run variance: omega / (1 - alpha - beta)."""
        return self.omega / (1 - self.alpha - self.beta)
    
    @property
    def persistence(self) -> float:
        """Volatility persistence: alpha + beta."""
        return self.alpha + self.beta


class GARCHModel:
    """
    GARCH(1,1) model for variance forecasting and path simulation.
    
    Variance equation:
        σ²_t = ω + α * ε²_{t-1} + β * σ²_{t-1}
    
    Return equation:
        r_t = μ + σ_t * z_t,  where z_t ~ Skewed GED(ν, skew)
    
    Innovation Distribution: Uses Skewed Generalized Error Distribution (GED)
    instead of Gaussian for better tail behavior, as specified in
    monte_carlo_architecture.md Section 4.2.
    
    Reference: monte_carlo_architecture.md Section 4.2 for theoretical foundation.
    """
    
    def __init__(self, params: Optional[GARCHParams] = None):
        self.params = params
        self._fitted = params is not None
        
    def fit(self, returns: np.ndarray, method: str = 'mle') -> GARCHParams:
        """
        Estimate GARCH parameters from historical returns.
        
        Args:
            returns: Array of log returns
            method: 'mle' for maximum likelihood, 'mom' for method of moments
            
        Returns:
            Fitted GARCHParams
        """
        if method == 'mle':
            params = self._fit_mle(returns)
        elif method == 'mom':
            params = self._fit_mom(returns)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        self.params = params
        self._fitted = True
        return params
    
    def _fit_mle(self, returns: np.ndarray, estimate_ged: bool = True) -> GARCHParams:
        """
        Maximum likelihood estimation of GARCH parameters.
        
        Likelihood function:
            For GED innovations: L = Σ [log(f(ε_t/σ_t)) - log(σ_t)]
            where f is the Skewed GED density function.
        
        Reference: monte_carlo_architecture.md Section 4.2.
        
        Args:
            returns: Array of log returns
            estimate_ged: If True, estimate GED parameters (nu, skew). 
                         If False, use defaults (nu=2.0, skew=0.0).
        """
        n = len(returns)
        
        # Initial guess based on sample statistics
        sample_var = np.var(returns)
        
        if estimate_ged:
            # Estimate nu and skew from residual kurtosis and skewness
            sample_skew = self._calculate_skewness(returns)
            sample_kurt = self._calculate_kurtosis(returns)
            # Map kurtosis to nu (higher kurtosis = lower nu)
            nu_init = max(1.2, min(3.0, 6.0 / (sample_kurt + 1)))
            skew_init = np.clip(sample_skew * 0.5, -0.5, 0.5)
            
            x0 = [
                sample_var * 0.05,  # omega
                0.10,               # alpha
                0.85,               # beta
                np.mean(returns),   # mu
                nu_init,            # nu (GED shape)
                skew_init           # skew (GED skewness)
            ]
            
            def neg_log_likelihood(params):
                omega, alpha, beta, mu, nu, skew = params
        else:
            x0 = [
                sample_var * 0.05,  # omega
                0.10,               # alpha
                0.85,               # beta
                np.mean(returns)    # mu
            ]
            
            def neg_log_likelihood(params):
                omega, alpha, beta, mu = params
                nu, skew = 2.0, 0.0  # Default: normal distribution
            
            # Parameter constraints
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
                return 1e10
            
            # Initialize variance
            sigma2 = np.zeros(n)
            sigma2[0] = sample_var  # Start with sample variance
            
            # Compute variance path
            eps = returns - mu
            for t in range(1, n):
                sigma2[t] = omega + alpha * eps[t-1]**2 + beta * sigma2[t-1]
            
            # Prevent numerical issues
            sigma2 = np.maximum(sigma2, 1e-10)
            
            # Negative log-likelihood
            ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + eps**2 / sigma2)
            return -ll
        
        # Optimize
        if estimate_ged:
            result = minimize(
                neg_log_likelihood,
                x0,
                method='L-BFGS-B',
                bounds=[
                    (1e-10, None),  # omega > 0
                    (0, 0.5),       # alpha in [0, 0.5]
                    (0, 0.999),     # beta in [0, 0.999]
                    (None, None),   # mu unconstrained
                    (1.1, 5.0),     # nu in [1.1, 5.0]
                    (-1.0, 1.0)     # skew in [-1, 1]
                ]
            )
            omega, alpha, beta, mu, nu, skew = result.x
        else:
            result = minimize(
                neg_log_likelihood,
                x0,
                method='L-BFGS-B',
                bounds=[
                    (1e-10, None),  # omega > 0
                    (0, 0.5),       # alpha in [0, 0.5]
                    (0, 0.999),     # beta in [0, 0.999]
                    (None, None)    # mu unconstrained
                ]
            )
            omega, alpha, beta, mu = result.x
            nu, skew = 2.0, 0.0  # Default: normal distribution
        
        # Ensure stationarity
        if alpha + beta >= 0.999:
            total = alpha + beta
            alpha = alpha / total * 0.99
            beta = beta / total * 0.99
        
        # Ensure nu is in valid range
        nu = np.clip(nu, 1.1, 5.0)
        skew = np.clip(skew, -1.0, 1.0)
        
        return GARCHParams(omega=omega, alpha=alpha, beta=beta, mu=mu, nu=nu, skew=skew)
    
    @staticmethod
    def _calculate_skewness(data: np.ndarray) -> float:
        """Calculate sample skewness."""
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if std == 0:
            return 0.0
        n = len(data)
        return (n / ((n-1) * (n-2))) * np.sum(((data - mean) / std) ** 3)
    
    @staticmethod
    def _calculate_kurtosis(data: np.ndarray) -> float:
        """Calculate excess kurtosis."""
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if std == 0:
            return 0.0
        n = len(data)
        return (n * (n+1) / ((n-1) * (n-2) * (n-3))) * np.sum(((data - mean) / std) ** 4) - 3 * (n-1)**2 / ((n-2) * (n-3))
    
    def forecast_variance(
        self, 
        current_variance: float, 
        last_shock: float, 
        horizon: int
    ) -> np.ndarray:
        """
        Forecast variance path for given horizon.
        
        Args:
            current_variance: Current conditional variance σ²_t
            last_shock: Last innovation ε_{t-1}
            horizon: Number of steps to forecast
            
        Returns:
            Array of variance forecasts [σ²_{t+1}, σ²_{t+2}, ..., σ²_{t+horizon}]
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
            
        omega = self.params.omega
        alpha = self.params.alpha
        beta = self.params.beta
        
        variances = np.zeros(horizon)
        
        # First step uses actual last shock
        variances[0] = omega + alpha * last_shock**2 + beta * current_variance
        
        # Subsequent steps use expected squared shock (= variance)
        for h in range(1, horizon):
            variances[h] = omega + (alpha + beta) * variances[h-1]
        
        return variances
    
    def simulate_paths(
        self,
        n_paths: int,
        horizon: int,
        initial_price: float,
        initial_variance: float,
        last_return: float,
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Monte Carlo price paths.
        
        Args:
            n_paths: Number of simulation paths
            horizon: Number of timesteps (bars) to simulate
            initial_price: Starting price S_0
            initial_variance: Current conditional variance σ²_0
            last_return: Previous return r_{-1} for first variance calculation
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (prices, variances) arrays with shape (n_paths, horizon+1)
            First column is initial values.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        if random_state is not None:
            np.random.seed(random_state)
            
        omega = self.params.omega
        alpha = self.params.alpha
        beta = self.params.beta
        mu = self.params.mu
        
        # Initialize arrays
        prices = np.zeros((n_paths, horizon + 1))
        variances = np.zeros((n_paths, horizon + 1))
        
        prices[:, 0] = initial_price
        variances[:, 0] = initial_variance
        
        # Pre-generate all random innovations using Skewed GED
        # Reference: monte_carlo_architecture.md Section 4.2 - "Use Skewed
        # Generalized Error Distribution rather than Gaussian for better tail behavior"
        if random_state is not None:
            np.random.seed(random_state)
        z = self._generate_skewed_ged(n_paths, horizon, self.params.nu, self.params.skew)
        
        # Previous shock for first step
        last_shock = last_return - mu
        
        # Simulate paths
        for t in range(horizon):
            if t == 0:
                # First step uses actual last shock
                variances[:, t+1] = omega + alpha * last_shock**2 + beta * variances[:, t]
            else:
                # Subsequent steps use simulated shocks
                eps_prev = returns[:, t-1] - mu  # Previous simulated shock
                variances[:, t+1] = omega + alpha * eps_prev**2 + beta * variances[:, t]
            
            # Ensure variance stays positive
            variances[:, t+1] = np.maximum(variances[:, t+1], 1e-10)
            
            # Generate returns
            sigma = np.sqrt(variances[:, t+1])
            returns_t = mu + sigma * z[:, t]
            
            # Update prices
            prices[:, t+1] = prices[:, t] * np.exp(returns_t)
            
            # Store returns for next variance calculation
            if t == 0:
                returns = np.zeros((n_paths, horizon))
            returns[:, t] = returns_t
        
        return prices, variances
    
    @staticmethod
    def _generate_skewed_ged(n: int, size: int, nu: float, skew: float) -> np.ndarray:
        """
        Generate random samples from Skewed Generalized Error Distribution.
        
        GED is a generalization of normal (nu=2) with fatter tails (nu<2)
        or thinner tails (nu>2). Skewed GED adds asymmetry.
        
        Args:
            n: Number of samples (rows)
            size: Sample size per row (columns)
            nu: Shape parameter (nu > 1, typical range 1.2-2.5)
            skew: Skewness parameter (-1 to 1, 0 = symmetric)
            
        Returns:
            Array of shape (n, size) with Skewed GED samples
        """
        import scipy.special
        import scipy.stats as stats
        
        # Generate standard GED (symmetric)
        # GED pdf: f(x) = (nu / (2*lambda*gamma(1/nu))) * exp(-0.5*|x/lambda|^nu)
        # where lambda = sqrt(2^(-2/nu) * gamma(1/nu) / gamma(3/nu))
        # For implementation, we approximate using t-distribution for nu < 2
        
        # Approximate GED using t-distribution with degrees of freedom ≈ nu
        # For exact implementation, use acceptance-rejection or transformation
        # This is a practical approximation for MVP
        if nu < 2:
            # Fatter tails: use t-distribution
            df = max(1.1, nu)  # Degrees of freedom (lower = fatter tails)
            z_symmetric = stats.t.rvs(df, size=(n, size), random_state=None) * np.sqrt((df-2)/df) if df > 2 else stats.t.rvs(df, size=(n, size), random_state=None)
        elif nu > 2:
            # Thinner tails: use normal with scaling
            z_symmetric = np.random.standard_normal((n, size)) * (2.0 / nu)**0.5
        else:
            # nu = 2: standard normal
            z_symmetric = np.random.standard_normal((n, size))
        
        # Apply skewness transformation
        # Simple skewness via inverse hyperbolic sine transform
        if abs(skew) > 1e-6:
            # Transform: z_skewed = sinh(skew * asinh(z_symmetric) + asinh(skew))
            z_skewed = np.sinh(skew * np.arcsinh(z_symmetric) + np.arcsinh(skew))
            # Renormalize to maintain variance ≈ 1
            z_skewed = z_skewed / np.std(z_skewed)
            return z_skewed
        
        return z_symmetric
    
    def get_state(self) -> dict:
        """Return current model state for serialization."""
        return {
            'params': {
                'omega': self.params.omega,
                'alpha': self.params.alpha,
                'beta': self.params.beta,
                'mu': self.params.mu,
                'nu': self.params.nu,
                'skew': self.params.skew
            } if self.params else None,
            'fitted': self._fitted
        }
    
    @classmethod
    def from_state(cls, state: dict) -> 'GARCHModel':
        """Reconstruct model from serialized state."""
        if state['params']:
            params = GARCHParams(**state['params'])
            return cls(params)
        return cls()
```

### 3.2.2 Calibration Pipeline

```python
# models/calibration.py

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

from .garch import GARCHModel, GARCHParams

logger = logging.getLogger(__name__)

class CalibrationPipeline:
    """
    Handles model calibration with rolling windows and validation.
    """
    
    DEFAULT_WINDOW_DAYS = 252  # One trading year
    
    def __init__(
        self,
        window_days: int = DEFAULT_WINDOW_DAYS,
        recalibration_frequency: str = 'daily',
        min_observations: int = 100
    ):
        """
        Args:
            window_days: Lookback window for calibration
            recalibration_frequency: 'daily', 'weekly', 'on_demand'
            min_observations: Minimum data points required
        """
        self.window_days = window_days
        self.recalibration_frequency = recalibration_frequency
        self.min_observations = min_observations
        self.calibration_history: Dict[str, List[dict]] = {}
        
    def calibrate_garch(
        self,
        symbol: str,
        bars: pd.DataFrame,
        as_of_date: Optional[datetime] = None
    ) -> GARCHParams:
        """
        Calibrate GARCH model for a single instrument.
        
        Args:
            symbol: Instrument symbol
            bars: DataFrame with OHLCV data (must have 'close' column)
            as_of_date: Calibration date (uses end of data if None)
            
        Returns:
            Calibrated GARCHParams
        """
        # Filter to calibration window
        if as_of_date:
            bars = bars[bars.index <= as_of_date]
        
        bars = bars.tail(self.window_days * 24 * 60)  # Approximate minutes
        
        if len(bars) < self.min_observations:
            raise ValueError(
                f"Insufficient data for calibration: {len(bars)} < {self.min_observations}"
            )
        
        # Calculate log returns
        returns = np.log(bars['close'] / bars['close'].shift(1)).dropna().values
        
        # Fit model with GED parameter estimation
        # Reference: monte_carlo_architecture.md Section 4.2 - Skewed GED innovations
        model = GARCHModel()
        params = model.fit(returns, method='mle')
        
        # Validate
        if not params.validate():
            logger.warning(f"Invalid parameters for {symbol}, using defaults")
            params = self._get_default_params(symbol)
        
        # Store calibration history
        self._record_calibration(symbol, params, as_of_date or bars.index[-1])
        
        logger.info(
            f"Calibrated {symbol}: ω={params.omega:.6f}, "
            f"α={params.alpha:.4f}, β={params.beta:.4f}, "
            f"ν={params.nu:.3f}, skew={params.skew:.3f}"
        )
        
        return params
    
    def calibrate_all_instruments(
        self,
        data: Dict[str, pd.DataFrame],
        as_of_date: Optional[datetime] = None
    ) -> Dict[str, GARCHParams]:
        """
        Calibrate models for all instruments.
        
        Args:
            data: Dict mapping symbol to DataFrame
            as_of_date: Calibration date
            
        Returns:
            Dict mapping symbol to GARCHParams
        """
        params = {}
        for symbol, bars in data.items():
            try:
                params[symbol] = self.calibrate_garch(symbol, bars, as_of_date)
            except Exception as e:
                logger.error(f"Calibration failed for {symbol}: {e}")
                params[symbol] = self._get_default_params(symbol)
        return params
    
    def _get_default_params(self, symbol: str) -> GARCHParams:
        """
        Return default parameters from monte_carlo_architecture.md Appendix B.
        
        Reference: monte_carlo_architecture.md Section 4.2 and Appendix B.
        GED parameters (nu, skew) default to 2.0 and 0.0 (normal distribution).
        """
        defaults = {
            'ES': GARCHParams(omega=0.00002, alpha=0.10, beta=0.88, nu=2.0, skew=0.0),
            'NQ': GARCHParams(omega=0.00002, alpha=0.12, beta=0.85, nu=2.0, skew=0.0),
            'GC': GARCHParams(omega=0.00002, alpha=0.08, beta=0.90, nu=2.0, skew=0.0),
            'SI': GARCHParams(omega=0.00003, alpha=0.10, beta=0.87, nu=2.0, skew=0.0),
        }
        return defaults.get(symbol, GARCHParams(omega=0.00002, alpha=0.10, beta=0.85, nu=2.0, skew=0.0))
    
    def _record_calibration(
        self,
        symbol: str,
        params: GARCHParams,
        calibration_date: datetime
    ) -> None:
        """Store calibration results for tracking."""
        if symbol not in self.calibration_history:
            self.calibration_history[symbol] = []
            
        self.calibration_history[symbol].append({
            'date': calibration_date,
            'omega': params.omega,
            'alpha': params.alpha,
            'beta': params.beta,
            'mu': params.mu,
            'nu': params.nu,
            'skew': params.skew,
            'persistence': params.persistence,
            'unconditional_vol': np.sqrt(params.unconditional_variance)
        })
    
    def needs_recalibration(
        self,
        symbol: str,
        current_date: datetime
    ) -> bool:
        """Check if recalibration is needed based on frequency."""
        if symbol not in self.calibration_history:
            return True
            
        last_calibration = self.calibration_history[symbol][-1]['date']
        
        if self.recalibration_frequency == 'daily':
            return current_date.date() > last_calibration.date()
        elif self.recalibration_frequency == 'weekly':
            return (current_date - last_calibration).days >= 7
        else:
            return False
```

## 3.3 Particle Filter Implementation

### 3.3.1 Core Particle Filter

**Note on Particle Injection:** For agent-based models (Phase 8), particle injection
protocol for news events will be implemented. See Phase 8 and
monte_carlo_architecture.md Section 5.3 for details. The current MVP uses
statistical models (GARCH) which do not require particle injection on news detection.

```python
# simulation/particle_filter.py

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
import logging

logger = logging.getLogger(__name__)

@dataclass
class Particle:
    """
    Single particle representing one simulation path.
    
    For GARCH MVP, state consists of:
    - Current simulated price
    - Current conditional variance
    - Weight (probability)
    """
    price: float
    variance: float
    weight: float = 1.0
    
    # Path history (last K values for diagnostics)
    price_history: List[float] = field(default_factory=list)
    
    def update_history(self, max_history: int = 100) -> None:
        """Append current price to history, maintaining max length."""
        self.price_history.append(self.price)
        if len(self.price_history) > max_history:
            self.price_history.pop(0)


@dataclass
class ParticleFilterState:
    """Complete state of particle filter for serialization/recovery."""
    particles: List[Particle]
    effective_sample_size: float
    resampling_count: int
    last_observation_time: float
    model_params: dict


class ParticleFilter:
    """
    Sequential Monte Carlo particle filter for real-time Bayesian updating.
    
    Algorithm (Reference: monte_carlo_architecture.md Section 6.2):
    1. PREDICT: Run each particle forward using GARCH dynamics
    2. OBSERVE: Receive real market price and volatility signature
    3. WEIGHT: Score particles by likelihood of observation
    4. RESAMPLE: Duplicate high-weight, discard low-weight particles (when ESS < N/2)
    
    Likelihood function matches monte_carlo_architecture.md Section 6.3:
    - Price matching component: L_price = exp(-0.5 × ((S_observed - S_particle) / σ_obs)²)
    - Path characteristic matching: L_path = exp(-0.5 × ((vol_observed - vol_particle) / σ_vol)²)
    - Combined: Likelihood = L_price × L_path
    """
    
    def __init__(
        self,
        n_particles: int = 10000,
        resample_threshold: float = 0.5,  # ESS threshold as fraction of N (N/2)
        tick_size: Optional[float] = None,
        spread: Optional[float] = None
    ):
        """
        Args:
            n_particles: Number of particles to maintain (N = 10,000 standard)
            resample_threshold: Resample when ESS < threshold * n_particles (default 0.5 = N/2)
            tick_size: Instrument tick size for σ_obs calculation (architecture doc Section 6.3)
            spread: Current bid-ask spread for σ_obs calculation
        """
        self.n_particles = n_particles
        self.resample_threshold = resample_threshold  # Default: N/2 per architecture doc
        self.tick_size = tick_size
        self.spread = spread
        
        self.particles: List[Particle] = []
        self.resampling_count = 0
        self.ess_history: List[float] = []
        
    def initialize(
        self,
        initial_price: float,
        initial_variance: float,
        variance_spread: float = 0.2
    ) -> None:
        """
        Initialize particle population.
        
        Args:
            initial_price: Starting price (all particles start here)
            initial_variance: Base conditional variance
            variance_spread: Fractional spread around initial variance
        """
        self.particles = []
        
        for _ in range(self.n_particles):
            # All particles start at same price
            price = initial_price
            
            # Variance has some spread to capture uncertainty
            var_noise = np.random.uniform(1 - variance_spread, 1 + variance_spread)
            variance = initial_variance * var_noise
            
            self.particles.append(Particle(
                price=price,
                variance=variance,
                weight=1.0 / self.n_particles
            ))
        
        self.resampling_count = 0
        self.ess_history = []
        
        logger.info(f"Initialized {self.n_particles} particles at price {initial_price}")
    
    def predict(
        self,
        garch_model: 'GARCHModel',
        timesteps: int = 1
    ) -> None:
        """
        Propagate all particles forward using GARCH dynamics.
        
        Args:
            garch_model: Fitted GARCH model
            timesteps: Number of timesteps to advance
        """
        omega = garch_model.params.omega
        alpha = garch_model.params.alpha
        beta = garch_model.params.beta
        mu = garch_model.params.mu
        
        for particle in self.particles:
            for _ in range(timesteps):
                # Update variance
                # Note: Using expected squared shock for prediction
                last_shock_sq = particle.variance  # E[ε²] = σ²
                new_variance = omega + alpha * last_shock_sq + beta * particle.variance
                new_variance = max(new_variance, 1e-10)
                
                # Generate return
                z = np.random.standard_normal()
                sigma = np.sqrt(new_variance)
                ret = mu + sigma * z
                
                # Update particle state
                particle.variance = new_variance
                particle.price = particle.price * np.exp(ret)
                particle.update_history()
    
    def update(
        self, 
        observed_price: float,
        observed_vol: Optional[float] = None,
        realized_vol: Optional[float] = None,
        dt: float = 1.0
    ) -> float:
        """
        Update particle weights based on observation.
        
        Likelihood function matches monte_carlo_architecture.md Section 6.3:
        - σ_obs = max(tick_size, spread/2, realized_vol × √Δt)
        - L_price = exp(-0.5 × ((S_observed - S_particle) / σ_obs)²)
        - L_path = exp(-0.5 × ((vol_observed - vol_particle) / σ_vol)²)
        - Combined likelihood = L_price × L_path
        
        Args:
            observed_price: Actual market price
            observed_vol: Realized volatility from recent returns (for path matching)
            realized_vol: Realized volatility for σ_obs calculation
            dt: Time step (in minutes, default 1.0)
            
        Returns:
            Effective Sample Size after update
        """
        # Calculate σ_obs per architecture doc Section 6.3:
        # σ_obs = max(tick_size, spread/2, realized_vol × √Δt)
        sigma_obs_candidates = []
        if self.tick_size is not None:
            sigma_obs_candidates.append(self.tick_size)
        if self.spread is not None:
            sigma_obs_candidates.append(self.spread / 2.0)
        if realized_vol is not None:
            sigma_obs_candidates.append(realized_vol * np.sqrt(dt))
        
        # If no candidates provided, use 0.1% of price as fallback
        if not sigma_obs_candidates:
            sigma_obs = observed_price * 0.001
        else:
            sigma_obs = max(sigma_obs_candidates)
        
        # Calculate σ_vol for path matching (if volatility observed)
        sigma_vol = None
        if observed_vol is not None:
            # Use 10% of observed volatility as uncertainty
            sigma_vol = observed_vol * 0.1
        
        # Update weights using combined likelihood
        for particle in self.particles:
            # Price matching component (architecture doc Section 6.3)
            price_diff = observed_price - particle.price
            l_price = np.exp(-0.5 * (price_diff / sigma_obs) ** 2)
            
            # Path characteristic matching (architecture doc Section 6.3)
            l_path = 1.0
            if observed_vol is not None and sigma_vol is not None:
                particle_vol = np.sqrt(particle.variance)
                vol_diff = observed_vol - particle_vol
                l_path = np.exp(-0.5 * (vol_diff / sigma_vol) ** 2)
            
            # Combined likelihood
            likelihood = l_price * l_path
            particle.weight *= likelihood
        
        # Normalize weights
        total_weight = sum(p.weight for p in self.particles)
        if total_weight > 0:
            for particle in self.particles:
                particle.weight /= total_weight
        else:
            # All weights collapsed - reset to uniform
            logger.warning("All particle weights collapsed, resetting to uniform")
            for particle in self.particles:
                particle.weight = 1.0 / self.n_particles
        
        # Calculate ESS
        ess = self._calculate_ess()
        self.ess_history.append(ess)
        
        # Resample if needed (ESS < N/2 per architecture doc Section 6.2)
        if ess < self.resample_threshold * self.n_particles:
            self.resample()
        
        return ess
    
    def resample(self) -> None:
        """
        Systematic resampling to duplicate high-weight particles.
        
        Algorithm matches monte_carlo_architecture.md Section 6.4:
        1. Compute cumulative weight distribution
        2. Generate single uniform random u ~ U(0, 1/N)
        3. Select particles at positions u, u + 1/N, u + 2/N, ...
        4. This naturally duplicates high-weight particles proportionally
        
        Systematic resampling is O(N) and has lower variance than
        multinomial resampling.
        """
        n = self.n_particles
        
        # Build cumulative weight distribution
        weights = np.array([p.weight for p in self.particles])
        cumsum = np.cumsum(weights)
        
        # Systematic resampling
        u0 = np.random.uniform(0, 1/n)
        u = u0 + np.arange(n) / n
        
        # Select particles
        indices = np.searchsorted(cumsum, u)
        indices = np.clip(indices, 0, n-1)
        
        # Create new particle list (deep copy states)
        new_particles = []
        for idx in indices:
            old = self.particles[idx]
            new_particles.append(Particle(
                price=old.price,
                variance=old.variance,
                weight=1.0 / n,
                price_history=old.price_history.copy()
            ))
        
        self.particles = new_particles
        self.resampling_count += 1
        
        logger.debug(f"Resampled particles (count: {self.resampling_count})")
    
    def _calculate_ess(self) -> float:
        """
        Calculate Effective Sample Size.
        
        ESS = 1 / Σ(w_i²)
        
        When all weights equal: ESS = N
        When one weight = 1: ESS = 1
        """
        weights = np.array([p.weight for p in self.particles])
        return 1.0 / np.sum(weights ** 2)
    
    def get_price_distribution(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get weighted price distribution.
        
        Returns:
            Tuple of (prices, weights) arrays
        """
        prices = np.array([p.price for p in self.particles])
        weights = np.array([p.weight for p in self.particles])
        return prices, weights
    
    def get_statistics(self) -> dict:
        """
        Calculate distribution statistics.
        
        Returns:
            Dict with mean, std, percentiles, etc.
        """
        prices, weights = self.get_price_distribution()
        
        # Weighted statistics
        mean = np.average(prices, weights=weights)
        variance = np.average((prices - mean) ** 2, weights=weights)
        std = np.sqrt(variance)
        
        # Percentiles (using sorted prices and cumulative weights)
        sorted_indices = np.argsort(prices)
        sorted_prices = prices[sorted_indices]
        sorted_weights = weights[sorted_indices]
        cumsum = np.cumsum(sorted_weights)
        
        percentiles = {}
        for p in [5, 10, 25, 50, 75, 90, 95]:
            idx = np.searchsorted(cumsum, p / 100)
            idx = min(idx, len(sorted_prices) - 1)
            percentiles[f'p{p}'] = sorted_prices[idx]
        
        return {
            'mean': mean,
            'std': std,
            'variance': variance,
            'ess': self._calculate_ess(),
            **percentiles
        }
    
    def get_state(self) -> ParticleFilterState:
        """Serialize current state for checkpointing."""
        return ParticleFilterState(
            particles=self.particles.copy(),
            effective_sample_size=self._calculate_ess(),
            resampling_count=self.resampling_count,
            last_observation_time=0,  # Set by caller
            model_params={}  # Set by caller
        )
    
    @classmethod
    def from_state(cls, state: ParticleFilterState) -> 'ParticleFilter':
        """Restore from serialized state."""
        pf = cls(n_particles=len(state.particles))
        pf.particles = state.particles
        pf.resampling_count = state.resampling_count
        return pf
```

### 3.3.2 Path Generator with Particle Filter Integration

```python
# simulation/path_generator.py

import numpy as np
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta
import logging

from ..models.garch import GARCHModel
from .particle_filter import ParticleFilter

logger = logging.getLogger(__name__)

class PathGenerator:
    """
    Generates and maintains Monte Carlo price paths with real-time updating.
    
    Responsibilities:
    1. Initialize simulations at start of each session
    2. Propagate particles forward as time passes
    3. Coordinate particle filter updates when observations arrive
    4. Provide access to current distribution statistics
    """
    
    MINUTES_PER_DAY = 24 * 60  # 1440 minutes in 24 hours
    
    def __init__(
        self,
        symbol: str,
        n_particles: int = 10000,
        horizon_minutes: int = 1440,  # 24 hours
        update_frequency_minutes: int = 1
    ):
        """
        Args:
            symbol: Instrument symbol
            n_particles: Number of simulation particles
            horizon_minutes: Forecast horizon in minutes
            update_frequency_minutes: How often to update (1 = every minute)
        """
        self.symbol = symbol
        self.n_particles = n_particles
        self.horizon_minutes = horizon_minutes
        self.update_frequency = update_frequency_minutes
        
        self.garch_model: Optional[GARCHModel] = None
        self.particle_filter: Optional[ParticleFilter] = None
        
        self.current_price: float = 0.0
        self.current_variance: float = 0.0
        self.last_update_time: Optional[datetime] = None
        self.simulation_start_time: Optional[datetime] = None
        
        # Full 24-hour path storage
        # Shape: (n_particles, horizon_minutes + 1)
        self.simulated_paths: Optional[np.ndarray] = None
        self.simulated_variances: Optional[np.ndarray] = None
        
    def initialize(
        self,
        garch_model: GARCHModel,
        initial_price: float,
        initial_variance: float,
        start_time: datetime
    ) -> None:
        """
        Initialize simulation system with model and starting conditions.
        
        Args:
            garch_model: Calibrated GARCH model
            initial_price: Current market price
            initial_variance: Current estimated variance
            start_time: Simulation start time
        """
        self.garch_model = garch_model
        self.current_price = initial_price
        self.current_variance = initial_variance
        self.simulation_start_time = start_time
        self.last_update_time = start_time
        
        # Initialize particle filter
        self.particle_filter = ParticleFilter(
            n_particles=self.n_particles,
            resample_threshold=0.5,
            observation_noise=self._get_observation_noise()
        )
        self.particle_filter.initialize(initial_price, initial_variance)
        
        # Generate full 24-hour paths
        self._generate_full_paths()
        
        logger.info(
            f"PathGenerator initialized for {self.symbol}: "
            f"price={initial_price}, variance={initial_variance:.6f}"
        )
    
    def _get_observation_noise(self) -> float:
        """
        Get observation noise parameter based on instrument.
        
        Returns fraction of price to use as σ_obs.
        Based on typical spreads:
        - ES: ~0.25 point on ~5000 = 0.00005
        - NQ: ~0.25 point on ~18000 = 0.000014
        - GC: ~0.10 on ~2000 = 0.00005
        - SI: ~0.005 on ~25 = 0.0002
        """
        noise_map = {
            'ES': 0.0001,
            'NQ': 0.0001,
            'GC': 0.0001,
            'SI': 0.0003,  # Higher due to wider spreads
        }
        return noise_map.get(self.symbol, 0.0001)
    
    def _generate_full_paths(self) -> None:
        """
        Generate full 24-hour paths for all particles.
        
        This creates the initial path ensemble. Paths will be
        trimmed/updated as real data arrives.
        """
        prices, variances = self.garch_model.simulate_paths(
            n_paths=self.n_particles,
            horizon=self.horizon_minutes,
            initial_price=self.current_price,
            initial_variance=self.current_variance,
            last_return=0.0  # Assume zero for initial generation
        )
        
        self.simulated_paths = prices
        self.simulated_variances = variances
        
        logger.debug(f"Generated {self.n_particles} paths over {self.horizon_minutes} minutes")
    
    def process_bar(self, bar: 'Bar') -> dict:
        """
        Process new bar: update particle filter and regenerate forward paths.
        
        Args:
            bar: New OHLCV bar
            
        Returns:
            Dict with updated statistics
        """
        # Time elapsed since last update
        if self.last_update_time:
            elapsed_minutes = int((bar.timestamp - self.last_update_time).total_seconds() / 60)
        else:
            elapsed_minutes = 1
        
        # 1. Propagate particles forward to current time
        if elapsed_minutes > 0:
            self.particle_filter.predict(self.garch_model, timesteps=elapsed_minutes)
        
        # 2. Update particle weights based on observation
        observed_price = bar.close
        ess = self.particle_filter.update(observed_price)
        
        # 3. Update current state
        self.current_price = observed_price
        self.current_variance = self._estimate_current_variance(bar)
        self.last_update_time = bar.timestamp
        
        # 4. Regenerate forward paths from current particle states
        self._regenerate_forward_paths()
        
        # 5. Calculate and return statistics
        stats = self.particle_filter.get_statistics()
        stats['timestamp'] = bar.timestamp
        stats['observed_price'] = observed_price
        stats['ess'] = ess
        stats['elapsed_minutes'] = self._minutes_since_start()
        stats['remaining_minutes'] = self.horizon_minutes - stats['elapsed_minutes']
        
        return stats
    
    def _estimate_current_variance(self, bar: 'Bar') -> float:
        """
        Estimate current variance from bar data.
        
        Uses Parkinson volatility estimator for single bar:
        σ² ≈ (ln(H/L))² / (4 * ln(2))
        """
        if bar.high > bar.low:
            log_range = np.log(bar.high / bar.low)
            parkinson_var = log_range ** 2 / (4 * np.log(2))
            return parkinson_var
        else:
            # Flat bar - use unconditional variance
            return self.garch_model.params.unconditional_variance
    
    def _regenerate_forward_paths(self) -> None:
        """
        Regenerate forward paths from current particle states.
        
        After particle filter update, each particle has:
        - Current price (adjusted toward observation)
        - Current variance
        
        We regenerate paths from these updated states.
        """
        remaining = self.horizon_minutes - self._minutes_since_start()
        if remaining <= 0:
            # Rolling window: reset to full horizon
            self._generate_full_paths()
            return
        
        # Extract current states from particles
        current_prices = np.array([p.price for p in self.particle_filter.particles])
        current_variances = np.array([p.variance for p in self.particle_filter.particles])
        
        # Generate forward paths for each particle
        omega = self.garch_model.params.omega
        alpha = self.garch_model.params.alpha
        beta = self.garch_model.params.beta
        mu = self.garch_model.params.mu
        
        paths = np.zeros((self.n_particles, remaining + 1))
        variances = np.zeros((self.n_particles, remaining + 1))
        
        paths[:, 0] = current_prices
        variances[:, 0] = current_variances
        
        # Vectorized simulation
        z = np.random.standard_normal((self.n_particles, remaining))
        
        for t in range(remaining):
            # Variance update (using expected squared shock)
            variances[:, t+1] = omega + alpha * variances[:, t] + beta * variances[:, t]
            variances[:, t+1] = np.maximum(variances[:, t+1], 1e-10)
            
            # Price update
            sigma = np.sqrt(variances[:, t+1])
            returns = mu + sigma * z[:, t]
            paths[:, t+1] = paths[:, t] * np.exp(returns)
        
        self.simulated_paths = paths
        self.simulated_variances = variances
    
    def _minutes_since_start(self) -> int:
        """Calculate minutes elapsed since simulation start."""
        if self.simulation_start_time and self.last_update_time:
            delta = self.last_update_time - self.simulation_start_time
            return int(delta.total_seconds() / 60)
        return 0
    
    def get_forecast_at_horizon(self, minutes_ahead: int) -> dict:
        """
        Get price distribution at specific future time.
        
        Args:
            minutes_ahead: Minutes from current time
            
        Returns:
            Dict with distribution statistics at that horizon
        """
        if self.simulated_paths is None:
            raise RuntimeError("Paths not generated. Call initialize() first.")
        
        # Clamp to available horizon
        minutes_ahead = min(minutes_ahead, self.simulated_paths.shape[1] - 1)
        
        prices = self.simulated_paths[:, minutes_ahead]
        weights = np.array([p.weight for p in self.particle_filter.particles])
        
        # Calculate statistics
        mean = np.average(prices, weights=weights)
        std = np.sqrt(np.average((prices - mean) ** 2, weights=weights))
        
        # Percentiles
        sorted_idx = np.argsort(prices)
        sorted_prices = prices[sorted_idx]
        sorted_weights = weights[sorted_idx]
        cumsum = np.cumsum(sorted_weights)
        
        percentiles = {}
        for p in [5, 10, 25, 50, 75, 90, 95]:
            idx = np.searchsorted(cumsum, p / 100)
            idx = min(idx, len(sorted_prices) - 1)
            percentiles[f'p{p}'] = sorted_prices[idx]
        
        return {
            'minutes_ahead': minutes_ahead,
            'mean': mean,
            'std': std,
            'current_price': self.current_price,
            **percentiles
        }
    
    def get_full_distribution_timeline(
        self,
        intervals: list = [15, 30, 60, 120, 240, 480, 720, 1440]
    ) -> list:
        """
        Get distribution statistics at multiple horizons.
        
        Args:
            intervals: List of minutes-ahead values
            
        Returns:
            List of dicts with statistics at each horizon
        """
        return [self.get_forecast_at_horizon(m) for m in intervals]
```

## 3.4 Signal Extraction

### 3.4.1 Distribution Analyzer

```python
# signals/distribution_analyzer.py

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from scipy import stats

@dataclass
class PriceZone:
    """Identified support/resistance zone."""
    price_low: float
    price_high: float
    density: float          # Fraction of particles in zone
    zone_type: str          # 'support', 'resistance', 'neutral'
    strength: str           # 'strong', 'moderate', 'weak'
    
@dataclass
class SignalOutput:
    """Complete signal output for trading decisions."""
    timestamp: float
    current_price: float
    
    # Direction
    directional_bias: str   # 'bullish', 'bearish', 'neutral'
    bias_confidence: float  # 0-1 scale
    
    # Key levels
    zones: List[PriceZone]
    
    # Distribution metrics
    mean_forecast: float
    std_forecast: float
    skewness: float
    
    # Trading recommendations
    primary_target: float
    secondary_target: float
    stop_zone: Tuple[float, float]


class DistributionAnalyzer:
    """
    Extracts trading signals from particle distribution.
    
    Responsibilities:
    1. Identify price zones with high particle density
    2. Classify zones as support/resistance
    3. Calculate directional bias
    4. Determine confidence levels
    """
    
    def __init__(
        self,
        n_bins: int = 50,
        density_threshold_strong: float = 0.10,   # 10% of particles
        density_threshold_moderate: float = 0.05  # 5% of particles
    ):
        """
        Args:
            n_bins: Number of histogram bins for density calculation
            density_threshold_strong: Density for 'strong' zone
            density_threshold_moderate: Density for 'moderate' zone
        """
        self.n_bins = n_bins
        self.density_threshold_strong = density_threshold_strong
        self.density_threshold_moderate = density_threshold_moderate
    
    def analyze(
        self,
        prices: np.ndarray,
        weights: np.ndarray,
        current_price: float,
        timestamp: float
    ) -> SignalOutput:
        """
        Full analysis of particle distribution.
        
        Args:
            prices: Array of particle prices at target horizon
            weights: Particle weights
            current_price: Current market price
            timestamp: Current timestamp
            
        Returns:
            SignalOutput with all trading signals
        """
        # Basic statistics
        mean = np.average(prices, weights=weights)
        variance = np.average((prices - mean) ** 2, weights=weights)
        std = np.sqrt(variance)
        
        # Skewness (third standardized moment)
        skewness = np.average(((prices - mean) / std) ** 3, weights=weights)
        
        # Identify zones
        zones = self._identify_zones(prices, weights, current_price)
        
        # Directional bias
        bias, confidence = self._calculate_directional_bias(
            mean, current_price, std, skewness
        )
        
        # Trading levels
        primary_target, secondary_target = self._identify_targets(
            zones, current_price, bias
        )
        stop_zone = self._identify_stop_zone(zones, current_price, bias)
        
        return SignalOutput(
            timestamp=timestamp,
            current_price=current_price,
            directional_bias=bias,
            bias_confidence=confidence,
            zones=zones,
            mean_forecast=mean,
            std_forecast=std,
            skewness=skewness,
            primary_target=primary_target,
            secondary_target=secondary_target,
            stop_zone=stop_zone
        )
    
    def _identify_zones(
        self,
        prices: np.ndarray,
        weights: np.ndarray,
        current_price: float
    ) -> List[PriceZone]:
        """
        Identify price zones with high particle density.
        
        Uses weighted histogram to find clusters.
        """
        # Create weighted histogram
        price_min = np.min(prices)
        price_max = np.max(prices)
        bin_edges = np.linspace(price_min, price_max, self.n_bins + 1)
        
        # Weighted histogram
        hist, _ = np.histogram(prices, bins=bin_edges, weights=weights)
        
        # Find significant bins (density above threshold)
        zones = []
        i = 0
        while i < len(hist):
            if hist[i] >= self.density_threshold_moderate:
                # Start of a zone - find contiguous high-density bins
                zone_start = i
                zone_density = 0
                while i < len(hist) and hist[i] >= self.density_threshold_moderate * 0.5:
                    zone_density += hist[i]
                    i += 1
                zone_end = i - 1
                
                # Create zone
                price_low = bin_edges[zone_start]
                price_high = bin_edges[zone_end + 1]
                
                # Classify zone
                if price_high < current_price:
                    zone_type = 'support'
                elif price_low > current_price:
                    zone_type = 'resistance'
                else:
                    zone_type = 'neutral'
                
                # Strength classification
                if zone_density >= self.density_threshold_strong:
                    strength = 'strong'
                elif zone_density >= self.density_threshold_moderate:
                    strength = 'moderate'
                else:
                    strength = 'weak'
                
                zones.append(PriceZone(
                    price_low=price_low,
                    price_high=price_high,
                    density=zone_density,
                    zone_type=zone_type,
                    strength=strength
                ))
            else:
                i += 1
        
        # Sort by density (strongest first)
        zones.sort(key=lambda z: z.density, reverse=True)
        
        return zones
    
    def _calculate_directional_bias(
        self,
        mean: float,
        current_price: float,
        std: float,
        skewness: float
    ) -> Tuple[str, float]:
        """
        Calculate directional bias and confidence.
        
        Bias based on:
        1. Mean vs current price
        2. Distribution width (std)
        3. Distribution asymmetry (skewness)
        """
        # Price difference in std units
        diff_std = (mean - current_price) / std if std > 0 else 0
        
        # Base confidence from distance
        # More than 1 std = high confidence
        # Less than 0.5 std = low confidence
        base_confidence = min(abs(diff_std), 2) / 2  # Cap at 1.0
        
        # Adjust for skewness (confirms or contradicts direction)
        if diff_std > 0 and skewness > 0:
            # Bullish mean, positive skew = confirms
            confidence = base_confidence * (1 + min(skewness, 1) * 0.2)
        elif diff_std < 0 and skewness < 0:
            # Bearish mean, negative skew = confirms
            confidence = base_confidence * (1 + min(abs(skewness), 1) * 0.2)
        else:
            # Skew contradicts direction
            confidence = base_confidence * (1 - min(abs(skewness), 1) * 0.1)
        
        confidence = np.clip(confidence, 0, 1)
        
        # Determine bias
        if diff_std > 0.3:
            bias = 'bullish'
        elif diff_std < -0.3:
            bias = 'bearish'
        else:
            bias = 'neutral'
        
        return bias, confidence
    
    def _identify_targets(
        self,
        zones: List[PriceZone],
        current_price: float,
        bias: str
    ) -> Tuple[float, float]:
        """
        Identify primary and secondary profit targets.
        
        Target selection based on:
        1. Direction of bias
        2. Zone strength
        3. Distance from current price
        """
        if bias == 'bullish':
            # Look for resistance zones above current price
            targets = [z for z in zones if z.zone_type == 'resistance']
        elif bias == 'bearish':
            # Look for support zones below current price
            targets = [z for z in zones if z.zone_type == 'support']
        else:
            # Neutral - return current price as both targets
            return current_price, current_price
        
        if not targets:
            return current_price, current_price
        
        # Sort by distance from current price (closest first)
        if bias == 'bullish':
            targets.sort(key=lambda z: z.price_low)
            primary = targets[0].price_low  # Near edge of closest zone
            secondary = targets[0].price_high if len(targets) == 1 else targets[1].price_low
        else:
            targets.sort(key=lambda z: -z.price_high)
            primary = targets[0].price_high  # Near edge of closest zone
            secondary = targets[0].price_low if len(targets) == 1 else targets[1].price_high
        
        return primary, secondary
    
    def _identify_stop_zone(
        self,
        zones: List[PriceZone],
        current_price: float,
        bias: str
    ) -> Tuple[float, float]:
        """
        Identify stop loss zone.
        
        Stop placed beyond zones in the opposite direction of bias.
        """
        if bias == 'bullish':
            # Stop below support
            support_zones = [z for z in zones if z.zone_type == 'support']
            if support_zones:
                # Below the nearest support
                support_zones.sort(key=lambda z: -z.price_high)
                nearest = support_zones[0]
                return (nearest.price_low * 0.998, nearest.price_low)
            else:
                # Default: 1% below current
                return (current_price * 0.99, current_price * 0.995)
        
        elif bias == 'bearish':
            # Stop above resistance
            resistance_zones = [z for z in zones if z.zone_type == 'resistance']
            if resistance_zones:
                resistance_zones.sort(key=lambda z: z.price_low)
                nearest = resistance_zones[0]
                return (nearest.price_high, nearest.price_high * 1.002)
            else:
                return (current_price * 1.005, current_price * 1.01)
        
        else:
            # Neutral - wide stop
            return (current_price * 0.99, current_price * 1.01)
```

### 3.4.2 Confidence Calculator

```python
# signals/confidence_calculator.py

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class ConfidenceLevel(Enum):
    """Discrete confidence levels for decision making."""
    VERY_HIGH = 'very_high'
    HIGH = 'high'
    MODERATE = 'moderate'
    LOW = 'low'
    VERY_LOW = 'very_low'

@dataclass
class ConfidenceMetrics:
    """Complete confidence assessment."""
    raw_std: float              # Standard deviation of particle prices
    normalized_std: float       # Std as percentage of current price
    confidence_level: ConfidenceLevel
    can_enter: bool             # Confidence sufficient for new positions
    can_adjust: bool            # Confidence sufficient to adjust TP/SL
    should_exit: bool           # Confidence collapsed, exit recommended


class ConfidenceCalculator:
    """
    Calculates confidence metrics from particle distribution.
    
    Confidence based on standard deviation of particle distribution:
    - Low std = particles agree = high confidence
    - High std = particles disagree = low confidence
    
    Thresholds calibrated per instrument based on typical volatility.
    """
    
    # Default thresholds as percentage of price
    # Will be calibrated from backtesting
    DEFAULT_THRESHOLDS = {
        'ES': {'very_high': 0.003, 'high': 0.005, 'moderate': 0.008, 'low': 0.012},
        'NQ': {'very_high': 0.004, 'high': 0.006, 'moderate': 0.010, 'low': 0.015},
        'GC': {'very_high': 0.003, 'high': 0.005, 'moderate': 0.008, 'low': 0.012},
        'SI': {'very_high': 0.006, 'high': 0.010, 'moderate': 0.015, 'low': 0.020},
    }
    
    def __init__(
        self,
        symbol: str,
        custom_thresholds: Optional[Dict] = None
    ):
        """
        Args:
            symbol: Instrument symbol
            custom_thresholds: Override default thresholds
        """
        self.symbol = symbol
        self.thresholds = custom_thresholds or self.DEFAULT_THRESHOLDS.get(
            symbol, self.DEFAULT_THRESHOLDS['ES']
        )
    
    def calculate(
        self,
        prices: np.ndarray,
        weights: np.ndarray,
        current_price: float
    ) -> ConfidenceMetrics:
        """
        Calculate confidence metrics from particle distribution.
        
        Args:
            prices: Particle price predictions
            weights: Particle weights
            current_price: Current market price
            
        Returns:
            ConfidenceMetrics object
        """
        # Weighted statistics
        mean = np.average(prices, weights=weights)
        variance = np.average((prices - mean) ** 2, weights=weights)
        std = np.sqrt(variance)
        
        # Normalize by price
        normalized_std = std / current_price
        
        # Classify confidence level
        if normalized_std <= self.thresholds['very_high']:
            level = ConfidenceLevel.VERY_HIGH
        elif normalized_std <= self.thresholds['high']:
            level = ConfidenceLevel.HIGH
        elif normalized_std <= self.thresholds['moderate']:
            level = ConfidenceLevel.MODERATE
        elif normalized_std <= self.thresholds['low']:
            level = ConfidenceLevel.LOW
        else:
            level = ConfidenceLevel.VERY_LOW
        
        # Decision flags
        can_enter = level in [ConfidenceLevel.VERY_HIGH, ConfidenceLevel.HIGH]
        can_adjust = level in [ConfidenceLevel.VERY_HIGH, ConfidenceLevel.HIGH, ConfidenceLevel.MODERATE]
        should_exit = level == ConfidenceLevel.VERY_LOW
        
        return ConfidenceMetrics(
            raw_std=std,
            normalized_std=normalized_std,
            confidence_level=level,
            can_enter=can_enter,
            can_adjust=can_adjust,
            should_exit=should_exit
        )
    
    def calibrate_thresholds(
        self,
        historical_stds: np.ndarray,
        historical_outcomes: np.ndarray
    ) -> Dict:
        """
        Calibrate thresholds from historical data.
        
        Args:
            historical_stds: Normalized std values from backtests
            historical_outcomes: Binary outcomes (1 = prediction correct)
            
        Returns:
            Calibrated threshold dict
        """
        # Sort by std
        sorted_idx = np.argsort(historical_stds)
        sorted_stds = historical_stds[sorted_idx]
        sorted_outcomes = historical_outcomes[sorted_idx]
        
        # Find std values at different accuracy cutoffs
        # very_high: 80%+ accuracy
        # high: 70%+ accuracy
        # moderate: 60%+ accuracy
        # low: 50%+ accuracy
        
        n = len(sorted_stds)
        cumsum = np.cumsum(sorted_outcomes)
        cumcount = np.arange(1, n + 1)
        accuracy = cumsum / cumcount
        
        thresholds = {}
        for level, target_acc in [('very_high', 0.80), ('high', 0.70), 
                                  ('moderate', 0.60), ('low', 0.50)]:
            # Find last index where accuracy >= target
            valid_idx = np.where(accuracy >= target_acc)[0]
            if len(valid_idx) > 0:
                thresholds[level] = sorted_stds[valid_idx[-1]]
            else:
                thresholds[level] = sorted_stds[0]
        
        self.thresholds = thresholds
        return thresholds
```

## 3.5 Main Application Loop

### 3.5.1 Core Application

```python
# main.py

import asyncio
import logging
from datetime import datetime
from typing import Dict
import signal
import sys

from data.feed_handler import FeedHandler, BarAggregator, Bar
from data.historical_loader import HistoricalDataLoader
from models.garch import GARCHModel
from models.calibration import CalibrationPipeline
from simulation.path_generator import PathGenerator
from signals.distribution_analyzer import DistributionAnalyzer
from signals.confidence_calculator import ConfidenceCalculator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SyntheticBarGenerator:
    """
    Main application coordinating all components.
    
    Lifecycle:
    1. Load configuration
    2. Initialize data feeds
    3. Calibrate models
    4. Start simulation loop
    5. Process bars and generate signals
    6. Handle graceful shutdown
    """
    
    INSTRUMENTS = ['ES', 'NQ', 'GC', 'SI']
    
    def __init__(self, config_path: str = 'config/'):
        self.config_path = config_path
        
        # Component containers
        self.feed_handler: FeedHandler = None
        self.bar_aggregator: BarAggregator = None
        self.calibration_pipeline: CalibrationPipeline = None
        
        # Per-instrument components
        self.garch_models: Dict[str, GARCHModel] = {}
        self.path_generators: Dict[str, PathGenerator] = {}
        self.distribution_analyzers: Dict[str, DistributionAnalyzer] = {}
        self.confidence_calculators: Dict[str, ConfidenceCalculator] = {}
        
        # State
        self.running = False
        self.last_signals: Dict[str, dict] = {}
        
    async def initialize(self) -> None:
        """Initialize all components."""
        logger.info("Initializing Synthetic Bar Generator...")
        
        # Load historical data and calibrate
        await self._calibrate_models()
        
        # Initialize per-instrument components
        for symbol in self.INSTRUMENTS:
            self.path_generators[symbol] = PathGenerator(
                symbol=symbol,
                n_particles=10000,
                horizon_minutes=1440
            )
            self.distribution_analyzers[symbol] = DistributionAnalyzer()
            self.confidence_calculators[symbol] = ConfidenceCalculator(symbol)
        
        # Initialize data feed
        await self._initialize_feed()
        
        logger.info("Initialization complete")
    
    async def _calibrate_models(self) -> None:
        """Load data and calibrate GARCH models."""
        logger.info("Calibrating models...")
        
        loader = HistoricalDataLoader('data/historical/')
        self.calibration_pipeline = CalibrationPipeline(
            window_days=252,
            recalibration_frequency='daily'
        )
        
        for symbol in self.INSTRUMENTS:
            try:
                # Load 2 years of data
                bars = loader.load_instrument(
                    symbol,
                    start_date='2023-01-01',
                    end_date='2025-01-01'
                )
                
                # Calibrate
                params = self.calibration_pipeline.calibrate_garch(symbol, bars)
                self.garch_models[symbol] = GARCHModel(params)
                
                logger.info(f"Calibrated {symbol}: α={params.alpha:.4f}, β={params.beta:.4f}")
                
            except Exception as e:
                logger.error(f"Calibration failed for {symbol}: {e}")
                # Use default parameters
                self.garch_models[symbol] = GARCHModel(
                    self.calibration_pipeline._get_default_params(symbol)
                )
    
    async def _initialize_feed(self) -> None:
        """Initialize real-time data feed."""
        # Implementation depends on chosen feed provider
        # Placeholder for IB or other feed handler
        pass
    
    def _on_bar(self, bar: Bar) -> None:
        """
        Callback when new bar is completed.
        
        This is the main processing loop for each bar.
        """
        symbol = bar.symbol
        
        try:
            # Process bar through path generator
            generator = self.path_generators[symbol]
            
            if generator.particle_filter is None:
                # First bar - initialize
                initial_var = self._estimate_initial_variance(bar)
                generator.initialize(
                    garch_model=self.garch_models[symbol],
                    initial_price=bar.close,
                    initial_variance=initial_var,
                    start_time=bar.timestamp
                )
            else:
                # Update with new bar
                stats = generator.process_bar(bar)
                
                # Get full distribution at target horizons
                distribution = generator.get_full_distribution_timeline()
                
                # Analyze distribution for signals
                prices, weights = generator.particle_filter.get_price_distribution()
                signal = self.distribution_analyzers[symbol].analyze(
                    prices=prices,
                    weights=weights,
                    current_price=bar.close,
                    timestamp=bar.timestamp.timestamp()
                )
                
                # Calculate confidence
                confidence = self.confidence_calculators[symbol].calculate(
                    prices=prices,
                    weights=weights,
                    current_price=bar.close
                )
                
                # Store and emit signal
                self.last_signals[symbol] = {
                    'signal': signal,
                    'confidence': confidence,
                    'distribution': distribution,
                    'stats': stats
                }
                
                self._emit_signal(symbol, self.last_signals[symbol])
                
        except Exception as e:
            logger.error(f"Error processing bar for {symbol}: {e}", exc_info=True)
    
    def _estimate_initial_variance(self, bar: Bar) -> float:
        """Estimate initial variance from first bar."""
        if bar.high > bar.low:
            log_range = np.log(bar.high / bar.low)
            return log_range ** 2 / (4 * np.log(2))
        return 0.0001  # Default
    
    def _emit_signal(self, symbol: str, signal_data: dict) -> None:
        """
        Emit signal for consumption.
        
        Override this method to integrate with trading system.
        """
        signal = signal_data['signal']
        conf = signal_data['confidence']
        
        logger.info(
            f"{symbol}: bias={signal.directional_bias}, "
            f"confidence={conf.confidence_level.value}, "
            f"target={signal.primary_target:.2f}, "
            f"std={conf.normalized_std:.4f}"
        )
    
    async def run(self) -> None:
        """Main run loop."""
        self.running = True
        
        # Set up signal handlers
        for sig in (signal.SIGTERM, signal.SIGINT):
            asyncio.get_event_loop().add_signal_handler(
                sig, lambda: asyncio.create_task(self.shutdown())
            )
        
        logger.info("Starting main loop...")
        
        # Start data feed
        # In production, this would be the feed handler connection
        # For now, we'll use a placeholder loop
        
        while self.running:
            await asyncio.sleep(1)
    
    async def shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down...")
        self.running = False
        
        # Disconnect feed
        if self.feed_handler:
            await self.feed_handler.disconnect()
        
        # Save state for recovery
        self._save_state()
        
        logger.info("Shutdown complete")
    
    def _save_state(self) -> None:
        """Save current state for recovery."""
        # Implementation: serialize particle filter states, model params, etc.
        pass


async def main():
    """Entry point."""
    generator = SyntheticBarGenerator()
    await generator.initialize()
    await generator.run()


if __name__ == '__main__':
    asyncio.run(main())
```

## 3.6 MVP Testing Checklist

### 3.6.1 Unit Tests

```
□ GARCH Model
  □ Parameter fitting produces valid parameters
  □ Fitted parameters satisfy stationarity constraint
  □ Variance forecasts are positive
  □ Path simulation produces correct shape
  □ Serialization/deserialization preserves state

□ Particle Filter
  □ Initialization creates correct number of particles
  □ Weights sum to 1 after update
  □ ESS calculation is correct
  □ Resampling maintains particle count
  □ Resampling preserves weight sum

□ Path Generator
  □ Forward paths have correct horizon
  □ Distribution statistics are valid
  □ Update correctly incorporates observation

□ Distribution Analyzer
  □ Zone identification finds clusters
  □ Bias calculation is directionally correct
  □ Target identification respects bias

□ Confidence Calculator
  □ Classification matches thresholds
  □ Decision flags are consistent with level
```

### 3.6.2 Integration Tests

```
□ Full Pipeline
  □ Bar ingestion triggers update
  □ Signal emission contains all fields
  □ Graceful handling of missing data
  □ Recovery from checkpoint

□ Calibration
  □ Rolling recalibration triggers correctly
  □ Parameter changes are logged
  □ Fallback to defaults on failure
```

### 3.6.3 Validation Tests

```
□ Stylized Facts
  □ Simulated returns have fat tails (kurtosis > 3)
  □ Volatility clustering present (significant AC in squared returns)
  □ Return autocorrelation near zero

□ Forecast Calibration
  □ PIT histogram approximately uniform
  □ 95% prediction intervals contain ~95% of outcomes
  □ CRPS improves over naive baseline
```

---

# 4. Phase 2: A-GARCH Enhancement

## 4.1 Objectives

1. Add asymmetric volatility response (leverage effect)
2. Calibrate per-instrument asymmetry parameters
3. Validate improvement over symmetric GARCH

## 4.2 A-GARCH Implementation

### 4.2.1 Extended Model Class

```python
# models/agarch.py

import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Optional

from .garch import GARCHModel, GARCHParams

@dataclass
class AGARCHParams(GARCHParams):
    """
    Asymmetric GARCH (GJR-GARCH) parameters.
    
    Extends GARCH with leverage effect term.
    """
    gamma: float = 0.0  # Asymmetry parameter
    
    def validate(self) -> bool:
        """Check parameter constraints."""
        if not super().validate():
            return False
        # gamma >= 0 (leverage effect is non-negative for equities)
        if self.gamma < 0:
            return False
        # Modified stationarity: alpha + beta + 0.5*gamma < 1
        if self.alpha + self.beta + 0.5 * self.gamma >= 1:
            return False
        return True
    
    @property
    def unconditional_variance(self) -> float:
        """Long-run variance with asymmetry correction."""
        return self.omega / (1 - self.alpha - self.beta - 0.5 * self.gamma)


class AGARCHModel(GARCHModel):
    """
    GJR-GARCH model with asymmetric volatility response.
    
    Variance equation:
        σ²_t = ω + α * ε²_{t-1} + γ * ε²_{t-1} * I_{ε<0} + β * σ²_{t-1}
    
    The γ term captures leverage effect: negative returns increase
    volatility more than positive returns of same magnitude.
    """
    
    def __init__(self, params: Optional[AGARCHParams] = None):
        self.params = params
        self._fitted = params is not None
    
    def fit(self, returns: np.ndarray, method: str = 'mle') -> AGARCHParams:
        """
        Estimate A-GARCH parameters from historical returns.
        """
        n = len(returns)
        sample_var = np.var(returns)
        
        # Initial guess
        x0 = [
            sample_var * 0.05,  # omega
            0.05,               # alpha
            0.10,               # gamma (asymmetry)
            0.85,               # beta
            np.mean(returns)    # mu
        ]
        
        def neg_log_likelihood(params):
            omega, alpha, gamma, beta, mu = params
            
            # Constraints
            if omega <= 0 or alpha < 0 or gamma < 0 or beta < 0:
                return 1e10
            if alpha + beta + 0.5 * gamma >= 1:
                return 1e10
            
            # Initialize
            sigma2 = np.zeros(n)
            sigma2[0] = sample_var
            
            eps = returns - mu
            
            for t in range(1, n):
                # Indicator for negative shock
                indicator = 1.0 if eps[t-1] < 0 else 0.0
                sigma2[t] = (omega + 
                           alpha * eps[t-1]**2 + 
                           gamma * eps[t-1]**2 * indicator + 
                           beta * sigma2[t-1])
            
            sigma2 = np.maximum(sigma2, 1e-10)
            
            ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + eps**2 / sigma2)
            return -ll
        
        result = minimize(
            neg_log_likelihood,
            x0,
            method='L-BFGS-B',
            bounds=[
                (1e-10, None),
                (0, 0.3),
                (0, 0.3),
                (0, 0.999),
                (None, None)
            ]
        )
        
        omega, alpha, gamma, beta, mu = result.x
        
        params = AGARCHParams(
            omega=omega,
            alpha=alpha,
            gamma=gamma,
            beta=beta,
            mu=mu
        )
        
        self.params = params
        self._fitted = True
        return params
    
    def simulate_paths(
        self,
        n_paths: int,
        horizon: int,
        initial_price: float,
        initial_variance: float,
        last_return: float,
        random_state: Optional[int] = None
    ) -> tuple:
        """
        Generate paths with asymmetric variance dynamics.
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        omega = self.params.omega
        alpha = self.params.alpha
        gamma = self.params.gamma
        beta = self.params.beta
        mu = self.params.mu
        
        prices = np.zeros((n_paths, horizon + 1))
        variances = np.zeros((n_paths, horizon + 1))
        
        prices[:, 0] = initial_price
        variances[:, 0] = initial_variance
        
        z = np.random.standard_normal((n_paths, horizon))
        
        # Track previous return for asymmetry
        prev_return = np.full(n_paths, last_return)
        
        for t in range(horizon):
            # Asymmetric variance update
            indicator = (prev_return < 0).astype(float)
            prev_shock_sq = (prev_return - mu) ** 2
            
            variances[:, t+1] = (omega + 
                                alpha * prev_shock_sq + 
                                gamma * prev_shock_sq * indicator + 
                                beta * variances[:, t])
            variances[:, t+1] = np.maximum(variances[:, t+1], 1e-10)
            
            # Generate returns
            sigma = np.sqrt(variances[:, t+1])
            returns = mu + sigma * z[:, t]
            
            # Update prices
            prices[:, t+1] = prices[:, t] * np.exp(returns)
            
            prev_return = returns
        
        return prices, variances
```

### 4.2.2 Calibration Updates

Update `CalibrationPipeline` to support A-GARCH:

```python
# Add to models/calibration.py

def calibrate_agarch(
    self,
    symbol: str,
    bars: pd.DataFrame,
    as_of_date: Optional[datetime] = None
) -> AGARCHParams:
    """
    Calibrate A-GARCH model for a single instrument.
    """
    # ... similar to calibrate_garch but using AGARCHModel
    
    model = AGARCHModel()
    params = model.fit(returns, method='mle')
    
    # Log asymmetry parameter
    logger.info(
        f"A-GARCH {symbol}: α={params.alpha:.4f}, "
        f"γ={params.gamma:.4f}, β={params.beta:.4f}"
    )
    
    return params
```

## 4.3 A/B Testing Framework

```python
# scripts/ab_test_models.py

"""
Compare GARCH vs A-GARCH forecast performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class ModelComparison:
    """Results of model comparison."""
    symbol: str
    garch_crps: float
    agarch_crps: float
    improvement: float  # Percentage improvement
    pit_uniformity_garch: float
    pit_uniformity_agarch: float
    
def compare_models(
    symbol: str,
    test_data: pd.DataFrame,
    garch_model: GARCHModel,
    agarch_model: AGARCHModel
) -> ModelComparison:
    """
    Run walk-forward comparison of GARCH vs A-GARCH.
    """
    # Implementation: generate forecasts with both models
    # Calculate CRPS, PIT uniformity for each
    # Return comparison metrics
    pass
```

---

# 5. Phase 3: Heston Stochastic Volatility

## 5.1 Objectives

1. Implement Heston model with QE discretization
2. Handle Feller condition violations
3. Add correlated volatility dynamics
4. Extend particle filter for stochastic volatility state

## 5.2 Heston Implementation

### 5.2.1 Model Class

```python
# models/heston.py

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)

@dataclass
class HestonParams:
    """
    Heston stochastic volatility parameters.
    
    Price SDE: dS = μS dt + √v S dW₁
    Variance SDE: dv = κ(θ - v) dt + ξ√v dW₂
    Correlation: Corr(dW₁, dW₂) = ρ
    """
    kappa: float    # Mean reversion speed
    theta: float    # Long-run variance
    xi: float       # Volatility of volatility
    rho: float      # Price-variance correlation
    mu: float = 0.0 # Drift
    
    def validate(self) -> bool:
        """Check parameter constraints."""
        # kappa, theta, xi > 0
        if self.kappa <= 0 or self.theta <= 0 or self.xi <= 0:
            return False
        # -1 < rho < 1
        if self.rho <= -1 or self.rho >= 1:
            return False
        return True
    
    @property
    def feller_satisfied(self) -> bool:
        """Check Feller condition: 2κθ ≥ ξ²."""
        return 2 * self.kappa * self.theta >= self.xi ** 2


class HestonModel:
    """
    Heston stochastic volatility model.
    
    Uses Quadratic-Exponential (QE) discretization scheme from
    Andersen (2007) for efficient and accurate simulation.
    """
    
    # QE scheme threshold
    PSI_CRIT = 1.5
    
    def __init__(self, params: Optional[HestonParams] = None):
        self.params = params
        self._fitted = params is not None
    
    def simulate_paths(
        self,
        n_paths: int,
        horizon: int,
        dt: float,  # Time step (in years, e.g., 1/525600 for 1 minute)
        initial_price: float,
        initial_variance: float,
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate price and variance paths using QE scheme.
        
        Args:
            n_paths: Number of simulation paths
            horizon: Number of time steps
            dt: Time step size in years
            initial_price: Starting price
            initial_variance: Starting variance
            random_state: Random seed
            
        Returns:
            Tuple of (prices, variances) with shape (n_paths, horizon+1)
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        kappa = self.params.kappa
        theta = self.params.theta
        xi = self.params.xi
        rho = self.params.rho
        mu = self.params.mu
        
        # Pre-compute constants
        exp_kappa_dt = np.exp(-kappa * dt)
        
        # Initialize arrays
        prices = np.zeros((n_paths, horizon + 1))
        variances = np.zeros((n_paths, horizon + 1))
        
        prices[:, 0] = initial_price
        variances[:, 0] = initial_variance
        
        # Pre-generate uniform randoms for QE scheme
        u_v = np.random.uniform(size=(n_paths, horizon))
        z_s = np.random.standard_normal((n_paths, horizon))
        
        for t in range(horizon):
            v_t = variances[:, t]
            
            # QE scheme for variance
            m = theta + (v_t - theta) * exp_kappa_dt
            s2 = (v_t * xi**2 * exp_kappa_dt / kappa * (1 - exp_kappa_dt) +
                  theta * xi**2 / (2 * kappa) * (1 - exp_kappa_dt)**2)
            psi = s2 / (m**2 + 1e-10)
            
            # Allocate arrays for this step
            v_next = np.zeros(n_paths)
            
            # Low psi region: quadratic scheme
            low_psi = psi <= self.PSI_CRIT
            if np.any(low_psi):
                inv_psi = 1 / (psi[low_psi] + 1e-10)
                b2 = 2 * inv_psi - 1 + np.sqrt(2 * inv_psi * (2 * inv_psi - 1))
                a = m[low_psi] / (1 + b2)
                b = np.sqrt(b2)
                z_v = np.random.standard_normal(np.sum(low_psi))
                v_next[low_psi] = a * (b + z_v)**2
            
            # High psi region: exponential scheme
            high_psi = ~low_psi
            if np.any(high_psi):
                p = (psi[high_psi] - 1) / (psi[high_psi] + 1)
                beta = (1 - p) / (m[high_psi] + 1e-10)
                
                # Inverse CDF sampling
                u = u_v[high_psi, t]
                v_next[high_psi] = np.where(
                    u <= p,
                    0,
                    np.log((1 - p) / (1 - u + 1e-10)) / beta
                )
            
            # Ensure variance stays positive
            variances[:, t+1] = np.maximum(v_next, 1e-10)
            
            # Price update with correlation
            # Log price increment: d(log S) = (μ - v/2)dt + √v dW₁
            # dW₁ = ρ dW_v + √(1-ρ²) dZ
            
            # Integrated variance for this step (trapezoidal)
            v_avg = 0.5 * (v_t + variances[:, t+1])
            
            # Correlated Brownian increments
            # We use z_s as the independent component
            # dW_v is implicit in the variance simulation
            
            # Simplified: use Euler for log-price with averaged variance
            dW = z_s[:, t] * np.sqrt(dt)
            log_return = (mu - 0.5 * v_avg) * dt + np.sqrt(v_avg) * dW
            
            prices[:, t+1] = prices[:, t] * np.exp(log_return)
        
        return prices, variances
    
    def fit(
        self,
        returns: np.ndarray,
        initial_guess: Optional[HestonParams] = None
    ) -> HestonParams:
        """
        Estimate Heston parameters from historical returns.
        
        Uses simulated method of moments targeting:
        - Mean return
        - Variance of returns
        - Autocorrelation of squared returns
        - Skewness
        - Kurtosis
        """
        # This is complex - typically done with particle MCMC or
        # options-implied calibration. Simplified version here.
        
        sample_var = np.var(returns)
        sample_kurt = self._kurtosis(returns)
        ac1_sq = self._autocorr(returns**2, 1)
        
        def objective(params):
            kappa, theta, xi, rho = params
            
            # Constraints
            if kappa <= 0 or theta <= 0 or xi <= 0:
                return 1e10
            if rho <= -1 or rho >= 1:
                return 1e10
            
            # Simulate and compute moments
            try:
                test_params = HestonParams(kappa=kappa, theta=theta, xi=xi, rho=rho)
                model = HestonModel(test_params)
                sim_prices, _ = model.simulate_paths(
                    n_paths=1000,
                    horizon=len(returns),
                    dt=1/252/390,  # Minute data
                    initial_price=100,
                    initial_variance=theta
                )
                sim_returns = np.diff(np.log(sim_prices), axis=1)
                
                # Moment matching
                sim_var = np.var(sim_returns)
                sim_kurt = self._kurtosis(sim_returns.flatten())
                sim_ac1 = self._autocorr(sim_returns.flatten()**2, 1)
                
                error = ((sim_var - sample_var) / sample_var)**2
                error += ((sim_kurt - sample_kurt) / (sample_kurt + 1))**2
                error += (sim_ac1 - ac1_sq)**2
                
                return error
            except:
                return 1e10
        
        # Initial guess based on sample statistics
        x0 = initial_guess or [2.0, sample_var, 0.5, -0.7]
        
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 1000}
        )
        
        kappa, theta, xi, rho = result.x
        
        params = HestonParams(kappa=kappa, theta=theta, xi=xi, rho=rho)
        self.params = params
        self._fitted = True
        
        if not params.feller_satisfied:
            logger.warning(
                f"Feller condition violated: 2κθ={2*kappa*theta:.4f} < ξ²={xi**2:.4f}"
            )
        
        return params
    
    @staticmethod
    def _kurtosis(x: np.ndarray) -> float:
        """Excess kurtosis."""
        n = len(x)
        m = np.mean(x)
        s = np.std(x)
        return np.mean(((x - m) / s)**4) - 3
    
    @staticmethod
    def _autocorr(x: np.ndarray, lag: int) -> float:
        """Autocorrelation at given lag."""
        n = len(x)
        m = np.mean(x)
        c0 = np.sum((x - m)**2)
        cl = np.sum((x[lag:] - m) * (x[:-lag] - m))
        return cl / c0 if c0 > 0 else 0
```

### 5.2.2 Extended Particle Filter for Stochastic Volatility

```python
# simulation/particle_filter_sv.py

"""
Extended particle filter for stochastic volatility models.

Particle state now includes both price and variance.
Variance is a latent (unobserved) state that must be inferred.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class SVParticle:
    """
    Particle for stochastic volatility model.
    
    State includes:
    - price: Observable (used for likelihood)
    - variance: Latent (inferred from price dynamics)
    """
    price: float
    variance: float
    weight: float = 1.0
    
    # Path history
    price_history: List[float] = field(default_factory=list)
    variance_history: List[float] = field(default_factory=list)


class SVParticleFilter:
    """
    Particle filter adapted for stochastic volatility.
    
    Key difference from basic filter:
    - Variance is latent state, not directly observed
    - Likelihood based on price innovation relative to particle's variance
    - Variance path must be propagated alongside price
    """
    
    def __init__(
        self,
        n_particles: int = 10000,
        resample_threshold: float = 0.5
    ):
        self.n_particles = n_particles
        self.resample_threshold = resample_threshold
        self.particles: List[SVParticle] = []
        
    def update(
        self,
        observed_price: float,
        observed_return: float,
        dt: float
    ) -> float:
        """
        Update particle weights based on observed price/return.
        
        Likelihood: p(r_t | v_t) ∝ exp(-r_t² / (2 v_t dt)) / √(v_t)
        
        Args:
            observed_price: Actual market price
            observed_return: Log return since last observation
            dt: Time step
            
        Returns:
            Effective Sample Size
        """
        for particle in self.particles:
            # Likelihood of observed return given particle's variance
            v = particle.variance
            expected_std = np.sqrt(v * dt)
            
            # Gaussian likelihood
            likelihood = np.exp(-0.5 * (observed_return / expected_std)**2) / expected_std
            particle.weight *= likelihood
            
            # Also adjust price toward observation (auxiliary)
            # This helps with particle diversity
            price_diff = observed_price - particle.price
            particle.price += 0.1 * price_diff  # Partial adjustment
        
        # Normalize
        total = sum(p.weight for p in self.particles)
        if total > 0:
            for p in self.particles:
                p.weight /= total
        
        ess = self._calculate_ess()
        
        if ess < self.resample_threshold * self.n_particles:
            self.resample()
        
        return ess
    
    # ... (rest similar to basic particle filter)
```

---

# 6. Phase 4: Jump-Diffusion

## 6.1 Objectives

1. Implement Merton jump-diffusion model
2. Calibrate jump parameters per instrument
3. Handle jump timing in simulation

## 6.2 Jump-Diffusion Implementation

```python
# models/jump_diffusion.py

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from scipy.stats import poisson, norm

@dataclass
class JumpParams:
    """
    Merton jump-diffusion parameters.
    
    Price process: dS/S = μ dt + σ dW + (J-1) dN
    
    Where:
    - dW is Brownian motion
    - N is Poisson process with intensity λ
    - J is jump multiplier: ln(J) ~ N(μ_J, σ_J²)
    """
    sigma: float        # Diffusion volatility
    lambda_: float      # Jump intensity (jumps per year)
    mu_jump: float      # Mean log jump size
    sigma_jump: float   # Std of log jump size
    mu: float = 0.0     # Drift
    
    def validate(self) -> bool:
        if self.sigma <= 0 or self.lambda_ < 0 or self.sigma_jump <= 0:
            return False
        return True
    
    @property
    def expected_jump_size(self) -> float:
        """E[J] = exp(μ_J + σ_J²/2)."""
        return np.exp(self.mu_jump + 0.5 * self.sigma_jump**2)
    
    @property
    def total_variance(self) -> float:
        """Total variance including jump contribution."""
        jump_var = self.lambda_ * (self.sigma_jump**2 + self.mu_jump**2)
        return self.sigma**2 + jump_var


class JumpDiffusionModel:
    """
    Merton jump-diffusion model.
    
    Handles compound Poisson process for jumps overlaid
    on standard geometric Brownian motion.
    """
    
    def __init__(self, params: Optional[JumpParams] = None):
        self.params = params
        self._fitted = params is not None
    
    def simulate_paths(
        self,
        n_paths: int,
        horizon: int,
        dt: float,
        initial_price: float,
        random_state: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate price paths with jumps.
        
        For each time step:
        1. Generate diffusion component: σ√dt Z
        2. Generate number of jumps: N ~ Poisson(λ dt)
        3. Generate jump sizes: Σ(ln J_i) where ln J ~ N(μ_J, σ_J²)
        4. Combine: S_{t+1} = S_t exp((μ - σ²/2)dt + diffusion + jumps)
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        sigma = self.params.sigma
        lambda_ = self.params.lambda_
        mu_j = self.params.mu_jump
        sigma_j = self.params.sigma_jump
        mu = self.params.mu
        
        prices = np.zeros((n_paths, horizon + 1))
        prices[:, 0] = initial_price
        
        for t in range(horizon):
            # Diffusion component
            z = np.random.standard_normal(n_paths)
            diffusion = sigma * np.sqrt(dt) * z
            
            # Jump component
            # Number of jumps in this period
            n_jumps = np.random.poisson(lambda_ * dt, n_paths)
            
            # Jump sizes (sum of log jumps)
            jump_component = np.zeros(n_paths)
            for i in range(n_paths):
                if n_jumps[i] > 0:
                    log_jumps = np.random.normal(mu_j, sigma_j, n_jumps[i])
                    jump_component[i] = np.sum(log_jumps)
            
            # Compensated drift (to make price martingale under risk-neutral)
            # Compensator: λ * (E[J] - 1) = λ * (exp(μ_J + σ_J²/2) - 1)
            compensator = lambda_ * (np.exp(mu_j + 0.5 * sigma_j**2) - 1)
            drift = (mu - 0.5 * sigma**2 - compensator) * dt
            
            # Update prices
            log_return = drift + diffusion + jump_component
            prices[:, t+1] = prices[:, t] * np.exp(log_return)
        
        return prices
    
    def fit(
        self,
        returns: np.ndarray,
        threshold_percentile: float = 95
    ) -> JumpParams:
        """
        Estimate jump parameters from returns.
        
        Simple approach:
        1. Separate returns into "normal" and "jumps" based on threshold
        2. Estimate diffusion σ from normal returns
        3. Estimate jump parameters from jump returns
        """
        # Threshold for identifying jumps
        abs_returns = np.abs(returns)
        threshold = np.percentile(abs_returns, threshold_percentile)
        
        # Separate
        normal_mask = abs_returns <= threshold
        jump_mask = ~normal_mask
        
        normal_returns = returns[normal_mask]
        jump_returns = returns[jump_mask]
        
        # Diffusion parameters from normal returns
        sigma = np.std(normal_returns) * np.sqrt(252 * 390)  # Annualize from minute
        mu = np.mean(normal_returns) * 252 * 390
        
        # Jump parameters
        n_jumps = np.sum(jump_mask)
        n_total = len(returns)
        dt_year = 1 / (252 * 390)  # Minutes in trading year
        
        lambda_ = n_jumps / (n_total * dt_year)
        
        if n_jumps > 0:
            mu_jump = np.mean(jump_returns)
            sigma_jump = np.std(jump_returns) if n_jumps > 1 else 0.05
        else:
            mu_jump = -0.02
            sigma_jump = 0.05
        
        params = JumpParams(
            sigma=sigma,
            lambda_=lambda_,
            mu_jump=mu_jump,
            sigma_jump=sigma_jump,
            mu=mu
        )
        
        self.params = params
        self._fitted = True
        return params
```

---

# 7. Phase 5: SVCJ Combined Model

## 7.1 Objectives

1. Combine Heston + Jump-Diffusion into SVCJ
2. Implement correlated jumps in price and variance
3. Full model calibration
4. Production optimization

## 7.2 SVCJ Implementation

```python
# models/svcj.py

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class SVCJParams:
    """
    Stochastic Volatility with Correlated Jumps parameters.
    
    Price: dS/S = μ dt + √v dW₁ + J_S dN
    Variance: dv = κ(θ - v) dt + ξ√v dW₂ + J_v dN
    
    Correlation: Corr(dW₁, dW₂) = ρ
    Correlated jumps: J_v > 0 when J_S occurs (variance jumps up on price jumps)
    """
    # Heston parameters
    kappa: float
    theta: float
    xi: float
    rho: float
    
    # Jump parameters
    lambda_: float      # Jump intensity
    mu_jump: float      # Mean price jump
    sigma_jump: float   # Std of price jump
    mu_v_jump: float    # Mean variance jump (usually positive)
    
    # Drift
    mu: float = 0.0


class SVCJModel:
    """
    Combined Stochastic Volatility + Correlated Jumps model.
    
    This is the most complete statistical model, capturing:
    - Stochastic volatility (Heston)
    - Leverage effect (ρ < 0)
    - Volatility clustering (mean-reverting v)
    - Discontinuous price moves (jumps)
    - Volatility spikes on jumps (correlated J_v)
    """
    
    def __init__(self, params: Optional[SVCJParams] = None):
        self.params = params
        self._fitted = params is not None
    
    def simulate_paths(
        self,
        n_paths: int,
        horizon: int,
        dt: float,
        initial_price: float,
        initial_variance: float,
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate price and variance paths with full SVCJ dynamics.
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        p = self.params
        
        prices = np.zeros((n_paths, horizon + 1))
        variances = np.zeros((n_paths, horizon + 1))
        
        prices[:, 0] = initial_price
        variances[:, 0] = initial_variance
        
        for t in range(horizon):
            v_t = variances[:, t]
            
            # 1. Generate correlated Brownian increments
            z1 = np.random.standard_normal(n_paths)
            z2 = np.random.standard_normal(n_paths)
            
            # Correlated increments
            dW1 = z1 * np.sqrt(dt)
            dW2 = (p.rho * z1 + np.sqrt(1 - p.rho**2) * z2) * np.sqrt(dt)
            
            # 2. Variance update (Heston)
            dv = p.kappa * (p.theta - v_t) * dt + p.xi * np.sqrt(np.maximum(v_t, 0)) * dW2
            
            # 3. Jump component
            # Number of jumps
            n_jumps = np.random.poisson(p.lambda_ * dt, n_paths)
            
            # Price jumps
            jump_price = np.zeros(n_paths)
            jump_var = np.zeros(n_paths)
            
            for i in range(n_paths):
                if n_jumps[i] > 0:
                    # Price jump
                    log_jump = np.random.normal(p.mu_jump, p.sigma_jump, n_jumps[i])
                    jump_price[i] = np.sum(log_jump)
                    
                    # Correlated variance jump
                    jump_var[i] = n_jumps[i] * p.mu_v_jump
            
            # 4. Update variance with jumps
            variances[:, t+1] = np.maximum(v_t + dv + jump_var, 1e-10)
            
            # 5. Update price
            # Use average variance for price diffusion
            v_avg = 0.5 * (v_t + variances[:, t+1])
            
            log_return = ((p.mu - 0.5 * v_avg) * dt + 
                         np.sqrt(v_avg) * dW1 + 
                         jump_price)
            
            prices[:, t+1] = prices[:, t] * np.exp(log_return)
        
        return prices, variances
```

---

# 8. Phase 6: Position Management System

## 8.1 Objectives

**Reference:** monte_carlo_architecture.md Part IV (Development Roadmap) - Phase 3: Trading Signal Layer (Weeks 11-14)

1. Extract support/resistance zones from particle distribution (architecture doc Section 7.1)
2. Calculate confidence metrics (std dev thresholds) (architecture doc Section 8)
3. Implement entry criteria (architecture doc Section 9.1)
4. Dynamic TP/SL adjustment with confidence gating (architecture doc Section 9.3)
3. Position sizing based on Kelly criterion
4. Real-time position tracking

## 8.2 Implementation

```python
# signals/position_manager.py

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class PositionSide(Enum):
    LONG = 'long'
    SHORT = 'short'
    FLAT = 'flat'

@dataclass
class Position:
    """Active position tracking."""
    symbol: str
    side: PositionSide
    entry_price: float
    entry_time: datetime
    size: float
    
    initial_tp: float
    initial_sl: float
    current_tp: float
    current_sl: float
    
    entry_confidence: float
    entry_std: float
    
    pnl: float = 0.0
    max_favorable: float = 0.0
    max_adverse: float = 0.0

@dataclass
class TradeSignal:
    """Signal for position entry/exit/adjustment."""
    action: str  # 'enter', 'exit', 'adjust_tp', 'adjust_sl', 'hold'
    side: Optional[PositionSide] = None
    price: Optional[float] = None
    size: Optional[float] = None
    reason: str = ''


class PositionManager:
    """
    Manages position entry, exit, and adjustment based on signals.
    
    Responsibilities:
    1. Evaluate entry criteria
    2. Calculate position size
    3. Set initial TP/SL
    4. Monitor and adjust TP/SL with confidence gating
    5. Trigger exits on stop conditions
    """
    
    def __init__(
        self,
        symbol: str,
        max_position_size: float,
        confidence_calculator: 'ConfidenceCalculator',
        risk_per_trade: float = 0.02  # 2% risk per trade
    ):
        self.symbol = symbol
        self.max_position_size = max_position_size
        self.confidence_calculator = confidence_calculator
        self.risk_per_trade = risk_per_trade
        
        self.current_position: Optional[Position] = None
        self.trade_history: List[Dict] = []
        
    def evaluate(
        self,
        signal: 'SignalOutput',
        confidence: 'ConfidenceMetrics',
        current_price: float,
        timestamp: datetime
    ) -> TradeSignal:
        """
        Evaluate current state and generate trade signal.
        
        Args:
            signal: Distribution analysis output
            confidence: Confidence metrics
            current_price: Current market price
            timestamp: Current time
            
        Returns:
            TradeSignal with recommended action
        """
        if self.current_position is None:
            # No position - evaluate entry
            return self._evaluate_entry(signal, confidence, current_price, timestamp)
        else:
            # Has position - evaluate management
            return self._evaluate_management(signal, confidence, current_price, timestamp)
    
    def _evaluate_entry(
        self,
        signal: 'SignalOutput',
        confidence: 'ConfidenceMetrics',
        current_price: float,
        timestamp: datetime
    ) -> TradeSignal:
        """Evaluate potential entry."""
        
        # Entry criteria
        # 1. Directional bias is not neutral
        if signal.directional_bias == 'neutral':
            return TradeSignal(action='hold', reason='No directional bias')
        
        # 2. Confidence sufficient for entry
        if not confidence.can_enter:
            return TradeSignal(
                action='hold',
                reason=f'Confidence too low: {confidence.confidence_level.value}'
            )
        
        # 3. Bias confidence above threshold
        if signal.bias_confidence < 0.6:
            return TradeSignal(
                action='hold',
                reason=f'Bias confidence too low: {signal.bias_confidence:.2f}'
            )
        
        # Entry approved - calculate size and levels
        side = PositionSide.LONG if signal.directional_bias == 'bullish' else PositionSide.SHORT
        
        # Position size from Kelly/risk management
        size = self._calculate_position_size(
            current_price=current_price,
            stop_price=signal.stop_zone[0] if side == PositionSide.LONG else signal.stop_zone[1],
            win_probability=signal.bias_confidence
        )
        
        return TradeSignal(
            action='enter',
            side=side,
            price=current_price,
            size=size,
            reason=f'{signal.directional_bias} bias with {confidence.confidence_level.value} confidence'
        )
    
    def _evaluate_management(
        self,
        signal: 'SignalOutput',
        confidence: 'ConfidenceMetrics',
        current_price: float,
        timestamp: datetime,
        volume: Optional[float] = None,
        price_change: Optional[float] = None,
        recent_returns: Optional[List[float]] = None
    ) -> TradeSignal:
        """
        Evaluate position management.
        
        Includes path-dependent stop logic per monte_carlo_architecture.md Section 9.4:
        - Check not just price level but path characteristics
        - Distinguish "grind against position" vs "spike and reject" scenarios
        - For agent-based particles (Phase 8): check which particle TYPE matched the move
        
        Reference: monte_carlo_architecture.md Section 9.4 for theoretical foundation.
        """
        
        pos = self.current_position
        
        # Update PnL tracking
        if pos.side == PositionSide.LONG:
            pos.pnl = (current_price - pos.entry_price) * pos.size
            pos.max_favorable = max(pos.max_favorable, current_price - pos.entry_price)
            pos.max_adverse = min(pos.max_adverse, current_price - pos.entry_price)
        else:
            pos.pnl = (pos.entry_price - current_price) * pos.size
            pos.max_favorable = max(pos.max_favorable, pos.entry_price - current_price)
            pos.max_adverse = min(pos.max_adverse, pos.entry_price - current_price)
        
        # 1. Check for confidence collapse (hard exit)
        if confidence.should_exit:
            return TradeSignal(
                action='exit',
                price=current_price,
                reason='Confidence collapsed - distribution too wide'
            )
        
        # 2. Path-dependent stop loss evaluation (architecture doc Section 9.4)
        # Check if price has reached stop loss level, but also evaluate path characteristics
        stop_hit = False
        if pos.side == PositionSide.LONG and current_price <= pos.current_sl:
            stop_hit = True
            adverse_move = True
        elif pos.side == PositionSide.SHORT and current_price >= pos.current_sl:
            stop_hit = True
            adverse_move = True
        else:
            adverse_move = False
        
        if stop_hit:
            # Path-dependent stop logic (architecture doc Section 9.4)
            # Path A: Grind against position (smooth move, low volume, sustained momentum)
            #   -> Chartist-dominated in agent-based models -> True trend -> Exit
            # Path B: Spike and reject (sharp move, high volume, immediate reversal attempt)
            #   -> Contested in agent-based models -> Potential exhaustion -> Hold if possible
            
            # For GARCH MVP: Use volume and volatility heuristics
            # Full implementation requires agent-based particle types (Phase 8)
            is_spike_and_reject = self._check_spike_and_reject(
                price_change=price_change,
                volume=volume,
                recent_returns=recent_returns,
                adverse_move=adverse_move
            )
            
            if is_spike_and_reject:
                # Path B: Spike and reject - potential exhaustion
                # In agent-based: contested particles matched -> reversal possible
                # For MVP: Log but still exit if stop hit (conservative)
                # Future Phase 8: Could hold position if agent mix suggests reversal
                return TradeSignal(
                    action='exit',
                    price=current_price,
                    reason=f'Stop loss hit at {pos.current_sl} (spike pattern detected - consider reversal)'
                )
            else:
                # Path A: Grind against position - true trend
                # In agent-based: chartist-dominated particles matched -> exit
                return TradeSignal(
                    action='exit',
                    price=current_price,
                    reason=f'Stop loss hit at {pos.current_sl} (sustained adverse move)'
                )
        
        # 3. Check for take profit hit
        if pos.side == PositionSide.LONG and current_price >= pos.current_tp:
            return TradeSignal(
                action='exit',
                price=current_price,
                reason=f'Take profit hit at {pos.current_tp}'
            )
        elif pos.side == PositionSide.SHORT and current_price <= pos.current_tp:
            return TradeSignal(
                action='exit',
                price=current_price,
                reason=f'Take profit hit at {pos.current_tp}'
            )
        
        # 4. Evaluate TP/SL adjustment (confidence-gated)
        if confidence.can_adjust:
            new_tp = signal.primary_target
            
            # Only adjust TP in favorable direction
            if pos.side == PositionSide.LONG:
                # Can move TP up but not down below entry
                if new_tp > pos.current_tp and new_tp > pos.entry_price:
                    return TradeSignal(
                        action='adjust_tp',
                        price=new_tp,
                        reason=f'Adjusting TP up to {new_tp:.2f}'
                    )
            else:
                # Short: can move TP down but not up above entry
                if new_tp < pos.current_tp and new_tp < pos.entry_price:
                    return TradeSignal(
                        action='adjust_tp',
                        price=new_tp,
                        reason=f'Adjusting TP down to {new_tp:.2f}'
                    )
        
        # 5. No action needed
        return TradeSignal(action='hold', reason='Position maintained')
    
    def _check_spike_and_reject(
        self,
        price_change: Optional[float],
        volume: Optional[float],
        recent_returns: Optional[List[float]],
        adverse_move: bool
    ) -> bool:
        """
        Detect "spike and reject" pattern (architecture doc Section 9.4 Path B).
        
        Pattern: Sharp move, high volume, immediate reversal attempt
        - For agent-based models: contested particles match -> potential exhaustion
        - For GARCH MVP: Heuristic based on volatility and volume
        
        Reference: monte_carlo_architecture.md Section 9.4.
        
        Note: Full implementation requires agent-based particle type information
        available in Phase 8 (Agent-Based Implementation).
        
        Returns:
            True if spike-and-reject pattern detected, False otherwise
        """
        # For MVP: Simple heuristics
        # Full implementation in Phase 8 will check particle types:
        # - If contested particles matched the move -> spike and reject
        # - If chartist-dominated particles matched -> grind (true trend)
        
        if price_change is None or recent_returns is None or len(recent_returns) < 3:
            return False
        
        # Check for sharp move followed by reversal attempt
        recent_vol = np.std(recent_returns)
        avg_vol = np.mean(np.abs(recent_returns[:-1])) if len(recent_returns) > 1 else recent_vol
        
        # Spike: large move with high volatility relative to recent average
        is_spike = abs(price_change) > 2 * avg_vol and recent_vol > 1.5 * avg_vol
        
        # Rejection: reversal in last return
        is_reject = False
        if len(recent_returns) >= 2:
            last_two = recent_returns[-2:]
            # Check for reversal: if move was adverse, check if last return reverses it
            if adverse_move:
                is_reject = (last_two[0] < 0 and last_two[1] > 0) or (last_two[0] > 0 and last_two[1] < 0)
        
        # High volume confirms spike (if available)
        high_volume = volume is not None and volume > np.percentile([v for v in [volume] if v], 75) if volume else False
        
        # Pattern detected if spike + (rejection or high volume)
        return is_spike and (is_reject or high_volume)
    
    def _calculate_position_size(
        self,
        current_price: float,
        stop_price: float,
        win_probability: float
    ) -> float:
        """
        Calculate position size using fractional Kelly.
        
        Kelly: f* = (p/a) * [1 - (q/p) * (a/b)]
        Where:
        - p = win probability
        - q = 1 - p
        - a = loss if wrong
        - b = gain if right
        
        Use half-Kelly for safety.
        """
        risk_per_unit = abs(current_price - stop_price)
        
        if risk_per_unit == 0:
            return 0
        
        # Assume 2:1 reward:risk for Kelly calculation
        reward_per_unit = risk_per_unit * 2
        
        p = win_probability
        q = 1 - p
        
        if p <= 0 or reward_per_unit <= 0:
            return 0
        
        kelly = (p / risk_per_unit) * (1 - (q / p) * (risk_per_unit / reward_per_unit))
        kelly = max(0, min(kelly, 0.25))  # Cap at 25%
        
        # Half-Kelly
        fraction = kelly * 0.5
        
        # Risk-based sizing
        account_risk = self.risk_per_trade  # 2% of account
        risk_based_size = account_risk / (risk_per_unit / current_price)
        
        # Take minimum of Kelly and risk-based
        size = min(fraction * self.max_position_size, risk_based_size)
        size = min(size, self.max_position_size)
        
        return size
    
    def open_position(
        self,
        side: PositionSide,
        price: float,
        size: float,
        tp: float,
        sl: float,
        confidence: float,
        std: float,
        timestamp: datetime
    ) -> Position:
        """Open new position."""
        self.current_position = Position(
            symbol=self.symbol,
            side=side,
            entry_price=price,
            entry_time=timestamp,
            size=size,
            initial_tp=tp,
            initial_sl=sl,
            current_tp=tp,
            current_sl=sl,
            entry_confidence=confidence,
            entry_std=std
        )
        
        logger.info(
            f"Opened {side.value} position: {self.symbol} @ {price}, "
            f"size={size}, TP={tp}, SL={sl}"
        )
        
        return self.current_position
    
    def close_position(
        self,
        price: float,
        timestamp: datetime,
        reason: str
    ) -> Dict:
        """Close current position and record trade."""
        if self.current_position is None:
            return {}
        
        pos = self.current_position
        
        # Calculate final PnL
        if pos.side == PositionSide.LONG:
            pnl = (price - pos.entry_price) * pos.size
        else:
            pnl = (pos.entry_price - price) * pos.size
        
        trade_record = {
            'symbol': pos.symbol,
            'side': pos.side.value,
            'entry_price': pos.entry_price,
            'exit_price': price,
            'entry_time': pos.entry_time,
            'exit_time': timestamp,
            'size': pos.size,
            'pnl': pnl,
            'max_favorable': pos.max_favorable,
            'max_adverse': pos.max_adverse,
            'reason': reason
        }
        
        self.trade_history.append(trade_record)
        self.current_position = None
        
        logger.info(
            f"Closed position: {pos.symbol} @ {price}, PnL={pnl:.2f}, "
            f"reason: {reason}"
        )
        
        return trade_record
```

---

# 9. Phase 7: Production Deployment

## 9.1 Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PRODUCTION ARCHITECTURE                          │
└─────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────┐
                    │      Load Balancer          │
                    │      (Health checks)        │
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
              ▼                    ▼                    ▼
    ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
    │   Instance A    │  │   Instance B    │  │   Instance C    │
    │   (Primary)     │  │   (Hot Standby) │  │   (Backup)      │
    │                 │  │                 │  │                 │
    │  ┌───────────┐  │  │  ┌───────────┐  │  │  ┌───────────┐  │
    │  │ ES Engine │  │  │  │ ES Engine │  │  │  │ ES Engine │  │
    │  │ NQ Engine │  │  │  │ NQ Engine │  │  │  │ NQ Engine │  │
    │  │ GC Engine │  │  │  │ GC Engine │  │  │  │ GC Engine │  │
    │  │ SI Engine │  │  │  │ SI Engine │  │  │  │ SI Engine │  │
    │  └───────────┘  │  │  └───────────┘  │  │  └───────────┘  │
    └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
             │                    │                    │
             └────────────────────┼────────────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │         Redis             │
                    │   (State, Pub/Sub)        │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │      TimescaleDB          │
                    │   (Historical, Logs)      │
                    └───────────────────────────┘
```

## 9.2 Docker Configuration

```yaml
# docker/docker-compose.yaml

version: '3.8'

services:
  synthetic-bar-generator:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    environment:
      - REDIS_HOST=redis
      - DB_HOST=timescale
      - LOG_LEVEL=INFO
    depends_on:
      - redis
      - timescale
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 32G
          cpus: '8'
    volumes:
      - ../config:/app/config:ro
      - ../data:/app/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes

  timescale:
    image: timescale/timescaledb:latest-pg15
    restart: unless-stopped
    environment:
      - POSTGRES_USER=synth
      - POSTGRES_PASSWORD=${DB_PASSWORD}
      - POSTGRES_DB=synthetic_bars
    volumes:
      - timescale-data:/var/lib/postgresql/data

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ../config/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:latest
    depends_on:
      - prometheus
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana

volumes:
  redis-data:
  timescale-data:
  prometheus-data:
  grafana-data:
```

## 9.3 Monitoring

```python
# monitoring/metrics.py

from prometheus_client import Counter, Gauge, Histogram, start_http_server
import logging

logger = logging.getLogger(__name__)

# Counters
BARS_PROCESSED = Counter(
    'bars_processed_total',
    'Total number of bars processed',
    ['symbol']
)

RESAMPLES = Counter(
    'particle_resamples_total',
    'Total number of particle resampling events',
    ['symbol']
)

TRADES_EXECUTED = Counter(
    'trades_executed_total',
    'Total number of trades executed',
    ['symbol', 'side', 'outcome']
)

# Gauges
CURRENT_PRICE = Gauge(
    'current_price',
    'Current market price',
    ['symbol']
)

EFFECTIVE_SAMPLE_SIZE = Gauge(
    'effective_sample_size',
    'Current ESS of particle filter',
    ['symbol']
)

DISTRIBUTION_STD = Gauge(
    'distribution_std',
    'Standard deviation of price distribution',
    ['symbol']
)

CONFIDENCE_LEVEL = Gauge(
    'confidence_level',
    'Current confidence level (0-4 scale)',
    ['symbol']
)

POSITION_PNL = Gauge(
    'position_pnl',
    'Current position PnL',
    ['symbol']
)

# Histograms
PROCESSING_LATENCY = Histogram(
    'bar_processing_seconds',
    'Time to process each bar',
    ['symbol'],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
)

def start_metrics_server(port: int = 8080):
    """Start Prometheus metrics HTTP server."""
    start_http_server(port)
    logger.info(f"Metrics server started on port {port}")
```

---

# 10. Phase 8: Agent-Based Implementation

## 10.1 Objectives

*Deferred until statistical model phases complete and deployed.*

1. Implement Chartist, Fundamentalist, Market Maker agents
2. Build synthetic order book
3. Emergent price dynamics from agent interaction
4. Extended particle filter with agent state
5. Richer signal extraction from agent mix inference
6. **Particle injection protocol on news detection** (architecture doc Section 5.3)

## 10.2 Reference

See `monte_carlo_architecture.md` Part I (Sections 1-3) for complete agent-based theoretical foundation.

## 10.3 News Detection and Particle Injection

**Particle Injection Protocol** (monte_carlo_architecture.md Section 5.3):

When news events are detected (high volume + contested flow pattern), inject fresh
particles with varied interpretations:

```
Bullish interpretation particles:
  - Fair value shifted UP by X%
  - Fundamentalists see current price as undervalued
  - Will generate buy-side flow

Bearish interpretation particles:
  - Fair value shifted DOWN by X%
  - Fundamentalists see current price as overvalued
  - Will generate sell-side flow

Contested interpretation particles:
  - Fair value unchanged or mixed beliefs
  - Two-sided flow, consolidation likely
```

**Detection Mechanism:**
- Volume spike without predetermined directional bias
- Forex Factory / economic calendar confirmation
- Pattern: high volume + contested flow = news event

**Implementation Notes:**
- Extend ParticleFilter class with `inject_particles()` method
- Create particles with varied fair value beliefs (agent-based state)
- Let reality filter which interpretation was correct
- See monte_carlo_architecture.md Section 5.3 for theoretical foundation

---

# 11. Testing Protocols

## 11.1 Unit Test Coverage Requirements

```
Minimum coverage targets:
- models/: 90%
- simulation/: 85%
- signals/: 85%
- data/: 80%
```

## 11.2 Integration Test Scenarios

```
□ End-to-end bar processing
□ Model hot-swap during runtime
□ Recovery from crash with state restoration
□ Handling of data gaps
□ Multi-instrument simultaneous processing
□ High-frequency bar arrival (stress test)
```

## 11.3 Validation Test Suite

**Reference:** monte_carlo_architecture.md Part VII for complete validation framework.

### 11.3.1 Probabilistic Forecast Metrics

**PIT (Probability Integral Transform) Uniformity Test** (architecture doc Section 14.2):
```
□ PIT histogram should be uniform (Anderson-Darling p > 0.10)
□ U-shaped histogram: Overconfident (underdispersed) - diagnostics required
□ Inverted-U: Underconfident (overdispersed) - increase uncertainty
□ Skewed: Systematic bias - model calibration issue
```

**Prediction Interval Coverage** (architecture doc Section 14.3):
```
□ 95% prediction intervals contain ~95% of observations
□ Christoffersen test: violations don't cluster (no consecutive failures)
□ Coverage between 94-96% for 95% intervals is acceptable
```

**CRPS (Continuous Ranked Probability Score)** (architecture doc Section 14.1):
```
□ CRPS = (1/N)Σ|x_i - y| - (1/2N²)Σ|x_i - x_j|
  Where: x_i = particle endpoints, y = realized value
□ CRPS improvement over naive baseline (>10%)
□ Lower CRPS = better forecast accuracy
```

### 11.3.2 Stylized Facts Validation

**Reference:** monte_carlo_architecture.md Section 15.

**Fat Tails** (Section 15.1):
```
□ Empirical kurtosis >> 3 (target: kurtosis > 3)
□ Hill tail-index in range 2-5
□ Extreme events occur more frequently than Gaussian predicts
```

**Volatility Clustering** (Section 15.2):
```
□ Significant positive autocorrelation of absolute/squared returns
□ Autocorrelation extends 20+ lags
□ ARCH-LM test: significant (p < 0.05)
□ Ljung-Box test on squared returns: significant (p < 0.05)
```

**Leverage Effect** (Section 15.3):
```
□ ES/NQ: Negative correlation between returns and future volatility
□ GC/SI: Near-zero correlation (symmetric vol response)
□ Correlation persists for several lags
```

**Return Autocorrelation**:
```
□ Near-zero autocorrelation in returns (no predictable patterns)
□ Ljung-Box test on returns: not significant (p > 0.05)
```

### 11.3.3 Walk-Forward Validation

**Reference:** monte_carlo_architecture.md Section 16.

**Protocol**:
```
□ Optimize parameters on in-sample window (e.g., 3 months)
□ Test out-of-sample (1 month)
□ Roll forward, repeat
□ Combine all out-of-sample periods for unbiased estimate
```

**Metrics**:
```
□ Walk-forward efficiency = OOS performance / IS performance
□ Target: Efficiency > 0.5 indicates acceptable robustness
□ Document parameter stability across windows
```

---

# 12. Monitoring and Maintenance

## 12.1 Daily Monitoring Checklist

```
□ All instruments processing (no stale data)
□ ESS above threshold for all instruments
□ Resampling frequency within normal bounds
□ No error spikes in logs
□ Latency within SLA
□ Model parameters within expected ranges
```

## 12.2 Weekly Maintenance

```
□ Review calibration parameter trends
□ Analyze forecast accuracy by instrument
□ Check for parameter drift requiring recalibration
□ Review trade performance if position management active
□ Backup state snapshots
```

## 12.3 Monthly Review

```
□ Full backtest validation with latest data
□ Compare model versions (GARCH vs A-GARCH vs Heston etc.)
□ Analyze prediction errors by market regime
□ Assess computational resource utilization
□ Plan capacity for any needed scaling
```

---

# Appendix: Configuration Templates

## instruments.yaml

**Reference:** monte_carlo_architecture.md Sections 10.3, 11.3, 12.3, 13.3 for instrument-specific session effects.

```yaml
instruments:
  ES:
    name: "E-mini S&P 500"
    exchange: "CME"
    tick_size: 0.25
    point_value: 50
    trading_hours:
      start: "18:00"
      end: "17:00"
      timezone: "America/New_York"
    observation_noise: 0.0001
    # Session effects (architecture doc Section 10.3):
    # - RTH open: Volatility spike, trend establishment
    # - European close: Often reversal/consolidation
    # - US close: Institutional rebalancing, mean-reversion to VWAP
    
  NQ:
    name: "E-mini NASDAQ-100"
    exchange: "CME"
    tick_size: 0.25
    point_value: 20
    trading_hours:
      start: "18:00"
      end: "17:00"
      timezone: "America/New_York"
    observation_noise: 0.0001
    # Session effects (architecture doc Section 11.3):
    # - Similar to ES but more volatile
    # - Tech earnings create outsized moves
    # - More sensitive to rate expectations
    
  GC:
    name: "Gold Futures"
    exchange: "COMEX"
    tick_size: 0.10
    point_value: 100
    trading_hours:
      start: "18:00"
      end: "17:00"
      timezone: "America/New_York"
    observation_noise: 0.0001
    # Session effects (architecture doc Section 12.3):
    # - London open: Volume builds
    # - PM Fix (10:00 AM ET): Peak volume/volatility
    # - Asian session: Quiet, Shanghai Gold Exchange influence
    
  SI:
    name: "Silver Futures"
    exchange: "COMEX"
    tick_size: 0.005
    point_value: 5000
    trading_hours:
      start: "18:00"
      end: "17:00"
      timezone: "America/New_York"
    observation_noise: 0.0003
    # Session effects (architecture doc Section 13.3):
    # - Similar to gold but more volatile
    # - Industrial demand adds complexity
    # - Most prone to flash crashes/spikes
```

## model_params.yaml

```yaml
# Default parameters - updated by calibration
# Reference: monte_carlo_architecture.md Appendix B for parameter specifications
garch:
  # GARCH parameters with Skewed GED innovation distribution
  # nu: GED shape parameter (lower = fatter tails, default 2.0 = normal)
  # skew: GED skewness parameter (0 = symmetric, default 0.0)
  ES: {omega: 0.00002, alpha: 0.10, beta: 0.88, nu: 2.0, skew: 0.0}
  NQ: {omega: 0.00002, alpha: 0.12, beta: 0.85, nu: 2.0, skew: 0.0}
  GC: {omega: 0.00002, alpha: 0.08, beta: 0.90, nu: 2.0, skew: 0.0}
  SI: {omega: 0.00003, alpha: 0.10, beta: 0.87, nu: 2.0, skew: 0.0}

agarch:
  # Asymmetric GARCH (GJR-GARCH) with leverage effect
  # Reference: monte_carlo_architecture.md Appendix B
  ES: {omega: 0.00002, alpha: 0.10, gamma: 0.12, beta: 0.88}
  NQ: {omega: 0.00002, alpha: 0.12, gamma: 0.14, beta: 0.85}
  GC: {omega: 0.00002, alpha: 0.08, gamma: 0.03, beta: 0.90}
  SI: {omega: 0.00003, alpha: 0.10, gamma: 0.04, beta: 0.87}

heston:
  # Heston stochastic volatility parameters
  # Reference: monte_carlo_architecture.md Appendix B
  # kappa: mean reversion speed, theta: long-run variance
  # xi: volatility of volatility, rho: correlation
  ES: {kappa: 3.0, theta: 0.04, xi: 0.50, rho: -0.80}
  NQ: {kappa: 2.5, theta: 0.05, xi: 0.55, rho: -0.85}
  GC: {kappa: 2.0, theta: 0.03, xi: 0.45, rho: -0.15}
  SI: {kappa: 1.8, theta: 0.05, xi: 0.60, rho: -0.10}

jumps:
  # Jump-diffusion parameters (per year)
  # Reference: monte_carlo_architecture.md Appendix B
  # lambda: jump intensity (jumps/year), mu_jump: mean jump size
  # sigma_jump: jump size volatility
  ES: {lambda: 2, mu_jump: -0.03, sigma_jump: 0.05}
  NQ: {lambda: 4, mu_jump: -0.03, sigma_jump: 0.06}
  GC: {lambda: 6, mu_jump: -0.02, sigma_jump: 0.08}
  SI: {lambda: 10, mu_jump: -0.02, sigma_jump: 0.12}

calibration:
  window_days: 252
  recalibration_frequency: daily
  min_observations: 100
```

## thresholds.yaml

```yaml
confidence:
  ES:
    very_high: 0.003
    high: 0.005
    moderate: 0.008
    low: 0.012
  NQ:
    very_high: 0.004
    high: 0.006
    moderate: 0.010
    low: 0.015
  GC:
    very_high: 0.003
    high: 0.005
    moderate: 0.008
    low: 0.012
  SI:
    very_high: 0.006
    high: 0.010
    moderate: 0.015
    low: 0.020

particle_filter:
  n_particles: 10000
  resample_threshold: 0.5
  
position_management:
  risk_per_trade: 0.02
  max_position_fraction: 0.10
  min_entry_confidence: high
  min_adjust_confidence: moderate
```

---

*Document Version: 1.0*
*Last Updated: January 2026*
*Reference: monte_carlo_architecture.md*
