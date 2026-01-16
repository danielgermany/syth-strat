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
│   ├── model_params.yaml         # Model hyperparameters (including regime-switching)
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
│   │   ├── regime_config.py      # Instrument, Timeframe, RegimeConfig enums/classes
│   │   ├── garch_params.py       # GARCHParams, RegimeGARCHParams dataclasses
│   │   ├── transition_matrix.py  # TransitionMatrix with TVTP
│   │   ├── lead_lag_estimator.py # Cross-instrument lead-lag estimation
│   │   ├── regime_calibration.py # RegimeSwitchingCalibrator
│   │   ├── garch.py              # Single-regime GARCH (legacy/reference)
│   │   ├── agarch.py             # Asymmetric GARCH
│   │   ├── heston.py             # Heston stochastic volatility
│   │   ├── jump_diffusion.py     # Merton jump-diffusion
│   │   ├── svcj.py               # Combined SVCJ model
│   │   └── calibration.py        # Parameter estimation utilities
│   │
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── regime_path_generator.py  # Regime-switching path generation
│   │   ├── path_generator.py         # Monte Carlo path generation (legacy)
│   │   ├── particle_filter.py        # Bayesian updating engine (with regime state)
│   │   └── resampler.py              # Systematic resampling
│   │
│   ├── signals/
│   │   ├── __init__.py
│   │   ├── distribution_analyzer.py  # Extract S/R zones
│   │   ├── confidence_calculator.py  # Std dev metrics
│   │   └── position_manager.py       # Entry/exit logic
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── time_utils.py             # Session handling, timezone, session effects
│   │   ├── timeframe_aggregator.py   # Multi-timeframe aggregation (1m → 5m, 15m, 1h)
│   │   ├── math_utils.py             # Statistical functions
│   │   └── validation.py             # Input validation
│   │
│   └── main.py                   # Application entry point
│
├── tests/
│   ├── unit/
│   │   ├── test_regime_garch.py
│   │   ├── test_transition_matrix.py
│   │   ├── test_lead_lag_estimator.py
│   │   ├── test_particle_filter.py
│   │   └── ...
│   ├── integration/
│   │   ├── test_full_pipeline.py
│   │   ├── test_cross_instrument.py
│   │   └── ...
│   └── validation/
│       ├── test_stylized_facts.py
│       ├── test_regime_detection.py
│       └── test_forecast_calibration.py
│
├── scripts/
│   ├── calibrate_models.py       # Offline calibration (including regime detection)
│   ├── estimate_lead_lag.py      # Cross-instrument lead-lag analysis
│   ├── backtest.py               # Historical validation
│   └── generate_reports.py       # Performance reports
│
├── data/
│   ├── historical/               # Historical bar data
│   ├── calibration/              # Saved model parameters (per-regime)
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

# 3. Phase 1: GARCH MVP (Updated with Regime-Switching)

**Reference:** monte_carlo_architecture.md Part IV (Development Roadmap) - Phase 1: GARCH MVP (Weeks 1-4)

**Update Note:** This section incorporates regime-switching GARCH with time-varying transition 
probabilities (TVTP) and symmetric bidirectional cross-instrument contagion based on empirical 
research findings. See "Research Verification" section below for evidence.

---

## 3.1 Objectives

1. Implement **regime-switching GARCH** with K regimes per instrument (architecture doc Section 4.2)
2. Implement **time-varying transition probabilities (TVTP)** for regime dynamics
3. Implement **symmetric bidirectional cross-instrument contagion** (GC↔SI, ES↔NQ)
4. Build particle filter infrastructure with extended state (architecture doc Part III, Section 6)
5. Establish signal extraction pipeline (architecture doc Part IV, Section 7)
6. Validate against stylized facts (architecture doc Part VII, Section 15)
7. Support **multi-timeframe generation** (1m base, aggregate to 5m, 15m, 1h)

---

## 3.2 Research Verification: Cross-Instrument Relationships

**CRITICAL:** Before implementation, we verified the lead-lag relationships between instrument pairs.
The research findings show **conflicting evidence** - neither instrument definitively leads the other.

### Gold-Silver (GC-SI) Relationship

**Studies finding GOLD leads SILVER:**
- "One-way directional spillover where gold price volatility significantly impacts silver" 
  (FSJ 2025 - Precious metals volatility study)
- "Gold is net contributor of shocks, silver is net receiver" 
  (ScienceDirect - Multiscale analysis)
- "Gold is the largest volatility transmitter" (Multiple sources)

**Studies finding SILVER leads GOLD:**
- "Silver is the net-contributor of spillover while gold is the net-receiver" 
  (Lau et al. 2017 - regime-switching cointegration)
- "Silver is always a net transmitter" (ScienceDirect - China metals study)

**Studies finding BIDIRECTIONAL relationship:**
- "Gold and silver are net transmitters of spillover regardless of time horizon" 
  (Barunik & Krehlik 2018 framework)
- "Bi-directional causality between gold and silver option-implied volatilities" 
  (ResearchGate - Volatility transmission study)

### ES-NQ Relationship

**Finding:** No strong academic evidence for either leading. They move together with NQ having 
higher volatility/beta, but this doesn't establish a lead-lag relationship. NQ reacts more 
strongly to shocks but doesn't systematically lead ES.

### Implementation Decision

**Given conflicting evidence, we implement SYMMETRIC BIDIRECTIONAL CONTAGION:**
- Equal contagion multipliers in both directions
- No hardcoded leader/follower assumptions
- Empirical lead-lag estimation from your data at calibration time
- Let TVTP handle cross-instrument dynamics naturally

---

## 3.3 Configuration and Data Structures

### 3.3.1 Enumerations and Configuration

```python
# models/regime_config.py

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np

class Timeframe(Enum):
    """Supported generation timeframes."""
    M1 = "1m"    # 1,380 bars/day (23h futures)
    M5 = "5m"    # 276 bars/day
    M15 = "15m"  # 92 bars/day
    H1 = "1h"    # 23 bars/day
    
    @property
    def minutes(self) -> int:
        """Return timeframe in minutes."""
        mapping = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}
        return mapping[self.value]
    
    @property
    def bars_per_day(self) -> int:
        """Approximate bars per trading day (23 hours)."""
        return (23 * 60) // self.minutes


class Instrument(Enum):
    """Supported instruments with pair relationships."""
    ES = "ES"
    NQ = "NQ"
    GC = "GC"
    SI = "SI"
    
    @property
    def pair(self) -> 'Instrument':
        """Return the paired instrument for cross-correlation."""
        pairs = {
            Instrument.ES: Instrument.NQ,
            Instrument.NQ: Instrument.ES,
            Instrument.GC: Instrument.SI,
            Instrument.SI: Instrument.GC,
        }
        return pairs[self]
    
    @property
    def is_equity_index(self) -> bool:
        """True for ES/NQ (equity indices)."""
        return self in [Instrument.ES, Instrument.NQ]
    
    @property
    def is_precious_metal(self) -> bool:
        """True for GC/SI (precious metals)."""
        return self in [Instrument.GC, Instrument.SI]
    
    @property
    def use_asymmetric_garch(self) -> bool:
        """
        ES/NQ: Use asymmetric (skewed-t) for leverage effect.
        GC/SI: Use symmetric (GED) - research shows no leverage effect.
        """
        return self.is_equity_index


@dataclass
class RegimeConfig:
    """Configuration for regime-switching model per instrument."""
    
    instrument: Instrument
    n_regimes: int                    # K regimes (3 for ES/NQ/GC, 4 for SI)
    regime_names: List[str]           # Human-readable names
    min_duration_bars: int            # Minimum bars before regime can switch
    
    # TVTP adjustment flags
    use_time_of_day: bool = True      # Adjust transitions by session
    use_cross_instrument: bool = True # Adjust based on partner regime
    use_lagged_rv: bool = True        # Adjust based on recent realized vol
    
    @classmethod
    def default_for_instrument(cls, instrument: Instrument) -> 'RegimeConfig':
        """Create default configuration for an instrument."""
        if instrument == Instrument.SI:
            # SI needs 4 regimes due to extreme tail events (flash crashes)
            return cls(
                instrument=instrument,
                n_regimes=4,
                regime_names=["Low", "Normal", "High", "Crisis"],
                min_duration_bars=5,
            )
        else:
            # ES, NQ, GC use 3 regimes
            return cls(
                instrument=instrument,
                n_regimes=3,
                regime_names=["Low", "Normal", "High"],
                min_duration_bars=5,
            )


@dataclass
class TimeframeConfig:
    """Configuration for a specific generation timeframe."""
    
    bar_size: Timeframe
    calibration_frequency: str    # '4h' | 'daily' | 'weekly'
    lookback_bars: int            # Historical bars for calibration
    min_regime_duration: int      # Minimum bars in regime
    
    @classmethod
    def default_for_timeframe(cls, tf: Timeframe) -> 'TimeframeConfig':
        """Create default configuration for a timeframe."""
        configs = {
            Timeframe.M1: cls(
                bar_size=tf,
                calibration_frequency='daily',
                lookback_bars=100_000,  # ~70 days of 1-min data
                min_regime_duration=5,
            ),
            Timeframe.M5: cls(
                bar_size=tf,
                calibration_frequency='daily',
                lookback_bars=20_000,
                min_regime_duration=5,
            ),
            Timeframe.M15: cls(
                bar_size=tf,
                calibration_frequency='daily',
                lookback_bars=6_000,
                min_regime_duration=3,
            ),
            Timeframe.H1: cls(
                bar_size=tf,
                calibration_frequency='weekly',
                lookback_bars=2_000,
                min_regime_duration=1,
            ),
        }
        return configs[tf]
```

### 3.3.2 Regime-Specific GARCH Parameters

```python
# models/garch_params.py

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class GARCHParams:
    """
    GARCH(1,1) parameters for a single regime.
    
    Reference: monte_carlo_architecture.md Section 4.2 for parameter specifications.
    
    Variance equation:
        σ²_t = ω + α * ε²_{t-1} + β * σ²_{t-1}
    
    For GJR-GARCH (ES/NQ only):
        σ²_t = ω + α * ε²_{t-1} + γ * ε²_{t-1} * I(ε_{t-1} < 0) + β * σ²_{t-1}
    """
    omega: float      # Baseline variance constant
    alpha: float      # Reaction to shocks (ARCH term)
    beta: float       # Persistence (GARCH term)
    gamma: float = 0.0  # Leverage effect (GJR-GARCH, ES/NQ only)
    mu: float = 0.0   # Drift term
    
    # Innovation distribution parameters
    # ES/NQ: skewed-t (leverage effect)
    # GC/SI: GED (symmetric fat tails)
    nu: float = 6.0      # Degrees of freedom (t-dist) or shape (GED)
    skew: float = 0.0    # Skewness (-1 to 1, 0 = symmetric)
    
    def validate(self) -> bool:
        """Check parameter constraints."""
        if self.omega <= 0:
            return False
        if self.alpha < 0 or self.beta < 0:
            return False
        # Stationarity: α + β + γ/2 < 1
        if self.alpha + self.beta + self.gamma / 2 >= 1:
            return False
        if self.nu <= 2:  # Need nu > 2 for finite variance
            return False
        return True
    
    @property
    def persistence(self) -> float:
        """Volatility persistence: α + β + γ/2."""
        return self.alpha + self.beta + self.gamma / 2
    
    @property
    def unconditional_variance(self) -> float:
        """Long-run variance: ω / (1 - α - β - γ/2)."""
        return self.omega / (1 - self.persistence)


@dataclass 
class RegimeGARCHParams:
    """
    Complete GARCH parameters for all regimes of an instrument.
    
    Example for ES (3 regimes):
        regime_params[0] = Low volatility regime parameters
        regime_params[1] = Normal volatility regime parameters  
        regime_params[2] = High volatility regime parameters
    """
    instrument: 'Instrument'
    regime_params: list  # List[GARCHParams], one per regime
    
    def get_regime_params(self, regime: int) -> GARCHParams:
        """Get parameters for a specific regime."""
        return self.regime_params[regime]
    
    @property
    def n_regimes(self) -> int:
        return len(self.regime_params)
    
    @classmethod
    def default_for_instrument(cls, instrument: 'Instrument') -> 'RegimeGARCHParams':
        """
        Create default regime-specific parameters.
        
        These are starting points - actual values should be calibrated from data.
        
        ES/NQ: Use GJR-GARCH (gamma > 0) with skewed-t innovations
        GC/SI: Use standard GARCH (gamma = 0) with GED innovations
        """
        from .regime_config import Instrument
        
        if instrument == Instrument.ES:
            return cls(
                instrument=instrument,
                regime_params=[
                    # Regime 0: Low volatility
                    GARCHParams(omega=0.00001, alpha=0.05, beta=0.92, gamma=0.08,
                               nu=8.0, skew=-0.1),
                    # Regime 1: Normal volatility
                    GARCHParams(omega=0.00002, alpha=0.10, beta=0.85, gamma=0.12,
                               nu=6.0, skew=-0.15),
                    # Regime 2: High volatility
                    GARCHParams(omega=0.00005, alpha=0.15, beta=0.78, gamma=0.15,
                               nu=4.0, skew=-0.2),
                ]
            )
        elif instrument == Instrument.NQ:
            return cls(
                instrument=instrument,
                regime_params=[
                    GARCHParams(omega=0.00001, alpha=0.06, beta=0.90, gamma=0.10,
                               nu=7.0, skew=-0.12),
                    GARCHParams(omega=0.00002, alpha=0.12, beta=0.82, gamma=0.14,
                               nu=5.0, skew=-0.18),
                    GARCHParams(omega=0.00006, alpha=0.18, beta=0.74, gamma=0.18,
                               nu=3.5, skew=-0.25),
                ]
            )
        elif instrument == Instrument.GC:
            # GC: Symmetric (no leverage effect), GED innovations
            return cls(
                instrument=instrument,
                regime_params=[
                    GARCHParams(omega=0.00001, alpha=0.06, beta=0.91, gamma=0.0,
                               nu=1.5, skew=0.0),  # GED shape
                    GARCHParams(omega=0.00002, alpha=0.09, beta=0.88, gamma=0.0,
                               nu=1.3, skew=0.0),
                    GARCHParams(omega=0.00005, alpha=0.12, beta=0.83, gamma=0.0,
                               nu=1.1, skew=0.0),
                ]
            )
        elif instrument == Instrument.SI:
            # SI: 4 regimes due to extreme kurtosis, symmetric GED
            return cls(
                instrument=instrument,
                regime_params=[
                    GARCHParams(omega=0.00002, alpha=0.06, beta=0.91, gamma=0.0,
                               nu=1.2, skew=0.0),
                    GARCHParams(omega=0.00003, alpha=0.10, beta=0.87, gamma=0.0,
                               nu=1.1, skew=0.0),
                    GARCHParams(omega=0.00008, alpha=0.15, beta=0.80, gamma=0.0,
                               nu=1.0, skew=0.0),
                    # Regime 3: Crisis (flash crash regime)
                    GARCHParams(omega=0.00020, alpha=0.25, beta=0.70, gamma=0.0,
                               nu=0.8, skew=0.0),
                ]
            )
        else:
            raise ValueError(f"Unknown instrument: {instrument}")
```

---

## 3.4 Transition Matrix with TVTP

### 3.4.1 Base Transition Matrix

```python
# models/transition_matrix.py

import numpy as np
from dataclasses import dataclass
from typing import Optional
from .regime_config import Instrument


@dataclass
class TransitionMatrix:
    """
    Markov transition matrix with time-varying transition probabilities (TVTP).
    
    The base matrix P[i,j] gives probability of transitioning from regime i to regime j.
    TVTP adjusts these probabilities based on:
        1. Time-of-day effects (session open/close)
        2. Cross-instrument contagion (partner's regime)
        3. Recent realized volatility ratio
    
    Research Note: Cross-instrument contagion uses SYMMETRIC bidirectional
    adjustments (equal multipliers both directions) because academic evidence
    shows conflicting results on which instrument leads.
    """
    
    base_matrix: np.ndarray  # K × K base transition probabilities
    instrument: Instrument
    
    # TVTP adjustment parameters
    tod_high_vol_boost: float = 1.3    # Multiplier for high-vol transitions at key times
    contagion_multiplier: float = 1.4  # SYMMETRIC - same both directions
    rv_adjustment_strength: float = 0.2  # How much RV ratio affects transitions
    
    def __post_init__(self):
        """Validate transition matrix."""
        assert self.base_matrix.shape[0] == self.base_matrix.shape[1], \
            "Transition matrix must be square"
        row_sums = self.base_matrix.sum(axis=1)
        assert np.allclose(row_sums, 1.0), \
            f"Rows must sum to 1, got {row_sums}"
    
    @property
    def n_regimes(self) -> int:
        return self.base_matrix.shape[0]
    
    def get_adjusted(
        self,
        time_of_day_hour: float,
        partner_regime: int,
        rv_ratio: float,
    ) -> np.ndarray:
        """
        Apply TVTP adjustments and return renormalized matrix.
        
        Args:
            time_of_day_hour: Current hour (0-23, fractional OK) in ET
            partner_regime: Current regime of paired instrument (for contagion)
            rv_ratio: RV_10bar / RV_100bar ratio (>1 means recent vol elevated)
            
        Returns:
            Adjusted and renormalized K × K transition matrix
        """
        P = self.base_matrix.copy()
        
        # 1. Time-of-day adjustments
        P = self._apply_tod_adjustment(P, time_of_day_hour)
        
        # 2. Cross-instrument contagion (SYMMETRIC)
        P = self._apply_contagion(P, partner_regime)
        
        # 3. Realized volatility adjustment
        P = self._apply_rv_adjustment(P, rv_ratio)
        
        # Renormalize rows to sum to 1
        row_sums = P.sum(axis=1, keepdims=True)
        P = P / row_sums
        
        return P
    
    def _apply_tod_adjustment(
        self, 
        P: np.ndarray, 
        hour: float
    ) -> np.ndarray:
        """
        Adjust transition probabilities based on time of day.
        
        Key volatility windows (hours in ET):
        - ES/NQ: RTH open (9:30), European close (11:30), US close (16:00)
        - GC/SI: London open (3:00), PM Fix (10:00)
        
        During these windows, increase probability of transitioning to higher regimes.
        """
        P = P.copy()
        boost = 1.0
        
        if self.instrument.is_equity_index:
            # ES/NQ volatility windows
            if 9.0 <= hour <= 10.0:     # RTH open
                boost = self.tod_high_vol_boost
            elif 11.0 <= hour <= 12.0:  # European close
                boost = self.tod_high_vol_boost * 0.8
            elif 15.5 <= hour <= 16.5:  # US close
                boost = self.tod_high_vol_boost * 0.9
        else:
            # GC/SI volatility windows
            if 2.5 <= hour <= 4.0:      # London open
                boost = self.tod_high_vol_boost
            elif 9.5 <= hour <= 10.5:   # PM Fix
                boost = self.tod_high_vol_boost
        
        if boost > 1.0:
            # Increase transitions TO higher regimes (last column = highest)
            for i in range(self.n_regimes):
                # Boost transitions to regimes higher than current
                for j in range(i + 1, self.n_regimes):
                    P[i, j] *= boost
        
        return P
    
    def _apply_contagion(
        self, 
        P: np.ndarray, 
        partner_regime: int
    ) -> np.ndarray:
        """
        Apply cross-instrument contagion effect.
        
        SYMMETRIC BIDIRECTIONAL CONTAGION:
        If partner is in a higher regime than us, increase our probability
        of transitioning UP to match. This applies equally in both directions
        (GC→SI and SI→GC get the same multiplier) because research shows
        no consistent leader.
        
        Research basis:
        - Some studies find gold leads silver (spillover direction)
        - Other studies find silver leads gold (Lau et al. 2017)
        - Studies find bidirectional causality at different time horizons
        - Solution: Use symmetric multipliers and let data drive dynamics
        """
        P = P.copy()
        
        # For each current regime, if partner is in higher regime,
        # boost our transitions toward higher regimes
        for i in range(self.n_regimes):
            if partner_regime > i:
                # Partner in higher vol regime → boost our upward transitions
                for j in range(i + 1, self.n_regimes):
                    P[i, j] *= self.contagion_multiplier
        
        return P
    
    def _apply_rv_adjustment(
        self, 
        P: np.ndarray, 
        rv_ratio: float
    ) -> np.ndarray:
        """
        Adjust based on recent realized volatility ratio.
        
        rv_ratio = RV_10bar / RV_100bar
        - If rv_ratio > 1.5: Recent vol elevated → boost upward transitions
        - If rv_ratio < 0.7: Recent vol depressed → boost downward transitions
        """
        P = P.copy()
        
        if rv_ratio > 1.5:
            # Elevated recent vol → increase upward transition probability
            adjustment = 1.0 + self.rv_adjustment_strength * (rv_ratio - 1.0)
            for i in range(self.n_regimes):
                for j in range(i + 1, self.n_regimes):
                    P[i, j] *= adjustment
                    
        elif rv_ratio < 0.7:
            # Depressed recent vol → increase downward transition probability
            adjustment = 1.0 + self.rv_adjustment_strength * (1.0 - rv_ratio)
            for i in range(1, self.n_regimes):
                for j in range(i):
                    P[i, j] *= adjustment
        
        return P
    
    @classmethod
    def default_for_instrument(cls, instrument: Instrument, n_regimes: int) -> 'TransitionMatrix':
        """
        Create default transition matrix with high self-transition probability.
        
        Diagonal elements ~0.95-0.98 ensures regimes are persistent.
        Off-diagonal transitions favor adjacent regimes.
        """
        P = np.zeros((n_regimes, n_regimes))
        
        # High self-transition probability (regime persistence)
        self_prob = 0.97
        
        for i in range(n_regimes):
            P[i, i] = self_prob
            remaining = 1.0 - self_prob
            
            # Distribute remaining probability to adjacent regimes
            neighbors = []
            if i > 0:
                neighbors.append(i - 1)
            if i < n_regimes - 1:
                neighbors.append(i + 1)
            
            if neighbors:
                for j in neighbors:
                    P[i, j] = remaining / len(neighbors)
        
        return cls(base_matrix=P, instrument=instrument)
```

---

## 3.5 Cross-Instrument Lead-Lag Estimation

### 3.5.1 Empirical Lead-Lag Estimator

```python
# models/lead_lag_estimator.py

import numpy as np
from scipy.signal import correlate
from statsmodels.tsa.stattools import grangercausalitytests
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class LeadLagResult:
    """Results from lead-lag estimation between two instruments."""
    
    instrument_1: str
    instrument_2: str
    
    # Cross-correlation results
    cross_corr_peak_lag: int      # Positive = instrument_1 leads instrument_2
    peak_correlation: float
    
    # Granger causality results (p-values)
    p_1_granger_causes_2: float   # p-value: does 1 Granger-cause 2?
    p_2_granger_causes_1: float   # p-value: does 2 Granger-cause 1?
    
    # Interpretation
    is_bidirectional: bool        # Both directions significant
    leader: Optional[str]         # None if bidirectional or unclear
    confidence: str               # "high", "moderate", "low"
    
    # Recommended contagion multipliers
    contagion_1_to_2: float       # Multiplier when 1's regime affects 2
    contagion_2_to_1: float       # Multiplier when 2's regime affects 1


def estimate_lead_lag(
    returns_1: np.ndarray,
    returns_2: np.ndarray,
    instrument_1: str,
    instrument_2: str,
    max_lag: int = 10,
    significance_level: float = 0.05
) -> LeadLagResult:
    """
    Estimate lead-lag relationship between two instruments.
    
    This function uses both cross-correlation and Granger causality to
    determine if one instrument systematically leads the other.
    
    Args:
        returns_1: Log returns of instrument 1
        returns_2: Log returns of instrument 2  
        instrument_1: Name of instrument 1 (e.g., "GC")
        instrument_2: Name of instrument 2 (e.g., "SI")
        max_lag: Maximum lag for Granger causality test
        significance_level: p-value threshold for significance
        
    Returns:
        LeadLagResult with cross-correlation, Granger causality, and
        recommended contagion multipliers.
        
    Research Note:
        Academic evidence on gold-silver and ES-NQ lead-lag is CONFLICTING.
        This function estimates from YOUR data rather than assuming a
        relationship. If results are unclear or bidirectional, we default
        to symmetric contagion multipliers.
    """
    # Ensure equal length
    min_len = min(len(returns_1), len(returns_2))
    r1 = returns_1[-min_len:]
    r2 = returns_2[-min_len:]
    
    # 1. Cross-correlation analysis
    cc = correlate(r1 - r1.mean(), r2 - r2.mean(), mode='full')
    cc = cc / (len(r1) * r1.std() * r2.std())  # Normalize
    
    lags = np.arange(-len(r1) + 1, len(r1))
    
    # Find peak within reasonable lag range
    center = len(r1) - 1
    search_range = slice(center - max_lag, center + max_lag + 1)
    local_cc = cc[search_range]
    local_lags = lags[search_range]
    
    peak_idx = np.argmax(np.abs(local_cc))
    peak_lag = local_lags[peak_idx]
    peak_corr = local_cc[peak_idx]
    
    # 2. Granger causality tests
    try:
        # Test: does instrument_1 Granger-cause instrument_2?
        data_12 = np.column_stack([r2, r1])  # [effect, cause]
        gc_1_causes_2 = grangercausalitytests(data_12, maxlag=max_lag, verbose=False)
        p_1_causes_2 = gc_1_causes_2[1][0]['ssr_ftest'][1]  # p-value at lag 1
        
        # Test: does instrument_2 Granger-cause instrument_1?
        data_21 = np.column_stack([r1, r2])  # [effect, cause]
        gc_2_causes_1 = grangercausalitytests(data_21, maxlag=max_lag, verbose=False)
        p_2_causes_1 = gc_2_causes_1[1][0]['ssr_ftest'][1]
        
    except Exception as e:
        logger.warning(f"Granger causality test failed: {e}. Using defaults.")
        p_1_causes_2 = 1.0
        p_2_causes_1 = 1.0
    
    # 3. Interpret results
    sig_1_causes_2 = p_1_causes_2 < significance_level
    sig_2_causes_1 = p_2_causes_1 < significance_level
    
    is_bidirectional = sig_1_causes_2 and sig_2_causes_1
    
    # Determine leader (if any)
    leader = None
    confidence = "low"
    
    if is_bidirectional:
        # Both directions significant - no clear leader
        leader = None
        confidence = "moderate"  # Bidirectional is a valid finding
    elif sig_1_causes_2 and not sig_2_causes_1:
        leader = instrument_1
        confidence = "high" if p_1_causes_2 < 0.01 else "moderate"
    elif sig_2_causes_1 and not sig_1_causes_2:
        leader = instrument_2
        confidence = "high" if p_2_causes_1 < 0.01 else "moderate"
    else:
        # Neither significant - independent or weak relationship
        leader = None
        confidence = "low"
    
    # 4. Determine contagion multipliers
    # Default: symmetric (1.4x both directions)
    base_multiplier = 1.4
    
    if leader == instrument_1:
        # 1 leads 2: 1's regime has stronger effect on 2
        contagion_1_to_2 = base_multiplier * 1.2  # 1.68x
        contagion_2_to_1 = base_multiplier * 0.8  # 1.12x
    elif leader == instrument_2:
        # 2 leads 1: 2's regime has stronger effect on 1
        contagion_1_to_2 = base_multiplier * 0.8
        contagion_2_to_1 = base_multiplier * 1.2
    else:
        # Symmetric (bidirectional or unclear)
        contagion_1_to_2 = base_multiplier
        contagion_2_to_1 = base_multiplier
    
    result = LeadLagResult(
        instrument_1=instrument_1,
        instrument_2=instrument_2,
        cross_corr_peak_lag=peak_lag,
        peak_correlation=peak_corr,
        p_1_granger_causes_2=p_1_causes_2,
        p_2_granger_causes_1=p_2_causes_1,
        is_bidirectional=is_bidirectional,
        leader=leader,
        confidence=confidence,
        contagion_1_to_2=contagion_1_to_2,
        contagion_2_to_1=contagion_2_to_1,
    )
    
    logger.info(
        f"Lead-lag estimation {instrument_1}-{instrument_2}: "
        f"leader={leader}, bidirectional={is_bidirectional}, "
        f"confidence={confidence}, "
        f"contagion multipliers: {contagion_1_to_2:.2f}/{contagion_2_to_1:.2f}"
    )
    
    return result
```

---

## 3.6 Regime-Switching GARCH Calibration

### 3.6.1 Complete Calibration Pipeline

```python
# models/regime_calibration.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
import statsmodels.api as sm
from arch import arch_model

from .regime_config import Instrument, RegimeConfig, TimeframeConfig, Timeframe
from .garch_params import GARCHParams, RegimeGARCHParams
from .transition_matrix import TransitionMatrix
from .lead_lag_estimator import estimate_lead_lag, LeadLagResult

logger = logging.getLogger(__name__)


class RegimeSwitchingCalibrator:
    """
    Calibrates regime-switching GARCH models with TVTP.
    
    Calibration proceeds in phases:
    1. Regime Detection: Use Markov-switching model to identify regimes
    2. Per-Regime GARCH: Estimate GARCH parameters within each regime
    3. Transition Matrix: Extract base transition probabilities
    4. Cross-Instrument: Estimate lead-lag for contagion multipliers
    
    The calibration is performed at 1-minute frequency, then parameters
    are used for all timeframes (aggregation handles the rest).
    """
    
    def __init__(
        self,
        instruments: List[Instrument],
        regime_configs: Optional[Dict[Instrument, RegimeConfig]] = None,
        timeframe_config: Optional[TimeframeConfig] = None,
    ):
        """
        Args:
            instruments: List of instruments to calibrate
            regime_configs: Per-instrument regime configuration (uses defaults if None)
            timeframe_config: Timeframe configuration (defaults to 1-minute)
        """
        self.instruments = instruments
        self.regime_configs = regime_configs or {
            inst: RegimeConfig.default_for_instrument(inst) 
            for inst in instruments
        }
        self.timeframe_config = timeframe_config or TimeframeConfig.default_for_timeframe(
            Timeframe.M1
        )
        
        # Calibration results
        self.regime_params: Dict[Instrument, RegimeGARCHParams] = {}
        self.transition_matrices: Dict[Instrument, TransitionMatrix] = {}
        self.lead_lag_results: Dict[Tuple[Instrument, Instrument], LeadLagResult] = {}
        self.calibration_date: Optional[datetime] = None
    
    def calibrate_all(
        self,
        data: Dict[Instrument, pd.DataFrame],
        as_of_date: Optional[datetime] = None
    ) -> None:
        """
        Run full calibration for all instruments.
        
        Args:
            data: Dict mapping Instrument to DataFrame with 'close' and 'volume' columns
            as_of_date: Calibration date (uses end of data if None)
        """
        self.calibration_date = as_of_date or datetime.now()
        
        # Phase 1 & 2: Regime detection and per-regime GARCH for each instrument
        for instrument in self.instruments:
            logger.info(f"Calibrating {instrument.value}...")
            
            df = data[instrument]
            config = self.regime_configs[instrument]
            
            # Calculate returns
            returns = np.log(df['close'] / df['close'].shift(1)).dropna().values
            
            # Detect regimes and estimate parameters
            regime_labels, transition_matrix = self._detect_regimes(
                returns, instrument, config.n_regimes
            )
            
            # Estimate per-regime GARCH parameters
            regime_garch = self._estimate_regime_garch(
                returns, regime_labels, instrument, config.n_regimes
            )
            
            self.regime_params[instrument] = regime_garch
            self.transition_matrices[instrument] = transition_matrix
        
        # Phase 3: Cross-instrument lead-lag estimation
        self._calibrate_cross_instrument(data)
        
        logger.info("Calibration complete")
    
    def _detect_regimes(
        self,
        returns: np.ndarray,
        instrument: Instrument,
        n_regimes: int
    ) -> Tuple[np.ndarray, TransitionMatrix]:
        """
        Detect volatility regimes using Markov-switching model.
        
        Uses statsmodels.tsa.regime_switching.MarkovRegression with
        switching_variance=True to identify different volatility states.
        """
        # Fit Markov-switching model
        try:
            mod = sm.tsa.MarkovRegression(
                returns * 100,  # Scale for numerical stability
                k_regimes=n_regimes,
                trend='n',
                switching_variance=True
            )
            result = mod.fit(disp=False)
            
            # Extract regime probabilities and most likely sequence
            regime_probs = result.smoothed_marginal_probabilities
            regime_labels = np.argmax(regime_probs, axis=1)
            
            # Extract transition matrix
            base_matrix = result.regime_transition
            
            # Sort regimes by variance (regime 0 = lowest vol)
            regime_variances = []
            for k in range(n_regimes):
                mask = regime_labels == k
                if mask.sum() > 0:
                    regime_variances.append(returns[mask].var())
                else:
                    regime_variances.append(0)
            
            sort_order = np.argsort(regime_variances)
            
            # Remap regime labels
            remap = {old: new for new, old in enumerate(sort_order)}
            regime_labels = np.array([remap[r] for r in regime_labels])
            
            # Reorder transition matrix
            base_matrix = base_matrix[sort_order][:, sort_order]
            
        except Exception as e:
            logger.warning(f"Markov switching failed for {instrument}: {e}. Using defaults.")
            # Fallback: simple variance-based regime assignment
            regime_labels = self._simple_regime_detection(returns, n_regimes)
            base_matrix = TransitionMatrix.default_for_instrument(
                instrument, n_regimes
            ).base_matrix
        
        transition_matrix = TransitionMatrix(
            base_matrix=base_matrix,
            instrument=instrument
        )
        
        return regime_labels, transition_matrix
    
    def _simple_regime_detection(
        self,
        returns: np.ndarray,
        n_regimes: int
    ) -> np.ndarray:
        """
        Fallback regime detection using rolling volatility percentiles.
        """
        # Calculate rolling volatility
        window = 60  # 1 hour of 1-min bars
        rolling_vol = pd.Series(returns).rolling(window).std().values
        
        # Assign regimes by percentile
        regime_labels = np.zeros(len(returns), dtype=int)
        percentiles = np.linspace(0, 100, n_regimes + 1)[1:-1]
        thresholds = np.nanpercentile(rolling_vol, percentiles)
        
        for i, threshold in enumerate(thresholds):
            regime_labels[rolling_vol > threshold] = i + 1
        
        return regime_labels
    
    def _estimate_regime_garch(
        self,
        returns: np.ndarray,
        regime_labels: np.ndarray,
        instrument: Instrument,
        n_regimes: int
    ) -> RegimeGARCHParams:
        """
        Estimate GARCH parameters for each regime separately.
        
        For ES/NQ: Use GJR-GARCH (asymmetric) with skewed-t distribution
        For GC/SI: Use standard GARCH (symmetric) with GED distribution
        """
        regime_params_list = []
        
        use_gjr = instrument.use_asymmetric_garch
        dist = 'skewt' if use_gjr else 'ged'
        vol_model = 'Garch' if not use_gjr else 'Garch'  # arch package handles GJR via o parameter
        
        for k in range(n_regimes):
            mask = regime_labels == k
            returns_k = returns[mask]
            
            if len(returns_k) < 100:
                # Insufficient data - use defaults scaled by regime
                logger.warning(f"Insufficient data for {instrument} regime {k}, using scaled defaults")
                default = RegimeGARCHParams.default_for_instrument(instrument)
                regime_params_list.append(default.regime_params[k])
                continue
            
            try:
                # Scale returns for numerical stability
                scale = 100
                
                if use_gjr:
                    # GJR-GARCH for ES/NQ
                    am = arch_model(
                        returns_k * scale,
                        vol='GARCH',
                        p=1, o=1, q=1,  # o=1 gives GJR term
                        dist='skewt'
                    )
                else:
                    # Standard GARCH for GC/SI
                    am = arch_model(
                        returns_k * scale,
                        vol='GARCH',
                        p=1, q=1,
                        dist='ged'
                    )
                
                result = am.fit(disp='off')
                
                # Extract parameters (scale back omega)
                omega = result.params.get('omega', 0.01) / (scale ** 2)
                alpha = result.params.get('alpha[1]', 0.1)
                beta = result.params.get('beta[1]', 0.85)
                gamma = result.params.get('gamma[1]', 0.0) if use_gjr else 0.0
                mu = result.params.get('mu', 0.0) / scale
                nu = result.params.get('nu', 6.0)
                skew = result.params.get('lambda', 0.0) if use_gjr else 0.0
                
                # Validate stationarity
                if alpha + beta + gamma / 2 >= 0.999:
                    total = alpha + beta + gamma / 2
                    factor = 0.99 / total
                    alpha *= factor
                    beta *= factor
                    gamma *= factor
                
                params = GARCHParams(
                    omega=max(omega, 1e-8),
                    alpha=max(alpha, 0.01),
                    beta=max(beta, 0.5),
                    gamma=max(gamma, 0.0),
                    mu=mu,
                    nu=max(nu, 2.5),
                    skew=np.clip(skew, -0.5, 0.5)
                )
                
                logger.info(
                    f"{instrument.value} Regime {k}: "
                    f"ω={params.omega:.6f}, α={params.alpha:.3f}, "
                    f"β={params.beta:.3f}, γ={params.gamma:.3f}"
                )
                
                regime_params_list.append(params)
                
            except Exception as e:
                logger.warning(f"GARCH estimation failed for {instrument} regime {k}: {e}")
                default = RegimeGARCHParams.default_for_instrument(instrument)
                regime_params_list.append(default.regime_params[k])
        
        return RegimeGARCHParams(
            instrument=instrument,
            regime_params=regime_params_list
        )
    
    def _calibrate_cross_instrument(
        self,
        data: Dict[Instrument, pd.DataFrame]
    ) -> None:
        """
        Estimate cross-instrument lead-lag relationships.
        
        Pairs: GC↔SI, ES↔NQ
        
        Research Note: We estimate from data rather than assuming a direction
        because academic evidence is conflicting.
        """
        pairs = [
            (Instrument.GC, Instrument.SI),
            (Instrument.ES, Instrument.NQ),
        ]
        
        for inst1, inst2 in pairs:
            if inst1 not in data or inst2 not in data:
                continue
            
            returns_1 = np.log(data[inst1]['close'] / data[inst1]['close'].shift(1)).dropna().values
            returns_2 = np.log(data[inst2]['close'] / data[inst2]['close'].shift(1)).dropna().values
            
            result = estimate_lead_lag(
                returns_1, returns_2,
                inst1.value, inst2.value,
                max_lag=10
            )
            
            self.lead_lag_results[(inst1, inst2)] = result
            
            # Update contagion multipliers in transition matrices
            if inst1 in self.transition_matrices:
                self.transition_matrices[inst1].contagion_multiplier = result.contagion_2_to_1
            if inst2 in self.transition_matrices:
                self.transition_matrices[inst2].contagion_multiplier = result.contagion_1_to_2
    
    def get_contagion_multiplier(
        self,
        from_instrument: Instrument,
        to_instrument: Instrument
    ) -> float:
        """
        Get the calibrated contagion multiplier for regime spillover.
        
        Args:
            from_instrument: Instrument whose regime affects the other
            to_instrument: Instrument being affected
            
        Returns:
            Contagion multiplier (default 1.4 if not calibrated)
        """
        # Check both orderings
        key1 = (from_instrument, to_instrument)
        key2 = (to_instrument, from_instrument)
        
        if key1 in self.lead_lag_results:
            return self.lead_lag_results[key1].contagion_1_to_2
        elif key2 in self.lead_lag_results:
            return self.lead_lag_results[key2].contagion_2_to_1
        else:
            return 1.4  # Default symmetric
```

---

## 3.7 Extended Particle State and Filter

### 3.7.1 Extended Particle with Regime

```python
# simulation/particle_filter.py (updated)

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class Particle:
    """
    Single particle representing one simulation path.
    
    Extended for regime-switching GARCH:
    - price: Current simulated price
    - variance: Current conditional variance (within-regime)
    - regime_probs: K-dimensional probability vector over regimes
    - current_regime: Most likely regime (argmax of regime_probs)
    - regime_duration: Bars spent in current regime (for min_duration constraint)
    """
    price: float
    variance: float
    weight: float = 1.0
    
    # Regime state
    regime_probs: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    current_regime: int = 0
    regime_duration: int = 0
    
    # Path history
    price_history: List[float] = field(default_factory=list)
    
    def update_history(self, max_history: int = 100) -> None:
        """Append current price to history, maintaining max length."""
        self.price_history.append(self.price)
        if len(self.price_history) > max_history:
            self.price_history.pop(0)
    
    def transition_regime(
        self,
        new_regime: int,
        min_duration: int = 5
    ) -> None:
        """
        Handle regime transition with minimum duration constraint.
        
        Args:
            new_regime: Proposed new regime
            min_duration: Minimum bars before regime can change
        """
        if new_regime == self.current_regime:
            self.regime_duration += 1
        elif self.regime_duration >= min_duration:
            # Allowed to transition
            self.current_regime = new_regime
            self.regime_duration = 1
        else:
            # Must stay in current regime
            self.regime_duration += 1
```

### 3.7.2 Regime-Switching Path Generator

```python
# simulation/regime_path_generator.py

import numpy as np
from typing import Dict, Tuple, Optional
from datetime import datetime
import logging

from ..models.regime_config import Instrument, RegimeConfig
from ..models.garch_params import RegimeGARCHParams
from ..models.transition_matrix import TransitionMatrix
from .particle_filter import Particle

logger = logging.getLogger(__name__)


class RegimeSwitchingPathGenerator:
    """
    Generate Monte Carlo price paths using regime-switching GARCH with TVTP.
    
    Algorithm:
    1. For each particle and each timestep:
       a. Get TVTP-adjusted transition matrix
       b. Sample regime transition
       c. Enforce min_duration constraint
       d. Generate return using regime-specific GARCH
       e. Update variance and price
    
    Cross-instrument contagion is incorporated through TVTP adjustments
    when the partner instrument's regime state is provided.
    """
    
    def __init__(
        self,
        instrument: Instrument,
        regime_params: RegimeGARCHParams,
        transition_matrix: TransitionMatrix,
        regime_config: RegimeConfig,
    ):
        self.instrument = instrument
        self.regime_params = regime_params
        self.transition_matrix = transition_matrix
        self.regime_config = regime_config
        
    def generate_paths(
        self,
        n_paths: int,
        horizon_bars: int,
        initial_price: float,
        initial_variance: float,
        initial_regime_probs: np.ndarray,
        time_of_day_start: float,
        partner_regimes: Optional[np.ndarray] = None,
        rv_ratio: float = 1.0,
        random_state: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate regime-switching GARCH paths.
        
        Args:
            n_paths: Number of simulation paths
            horizon_bars: Number of bars to simulate
            initial_price: Starting price
            initial_variance: Current conditional variance
            initial_regime_probs: K-dimensional initial regime probabilities
            time_of_day_start: Starting hour (ET, 0-23)
            partner_regimes: Optional array of partner instrument regimes per bar
                            (for cross-instrument contagion)
            rv_ratio: Initial RV_10bar / RV_100bar ratio
            random_state: Random seed
            
        Returns:
            Tuple of:
            - prices: (n_paths, horizon_bars + 1) price paths
            - variances: (n_paths, horizon_bars + 1) variance paths
            - regimes: (n_paths, horizon_bars + 1) regime labels
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        n_regimes = self.regime_config.n_regimes
        min_duration = self.regime_config.min_duration_bars
        
        # Initialize arrays
        prices = np.zeros((n_paths, horizon_bars + 1))
        variances = np.zeros((n_paths, horizon_bars + 1))
        regimes = np.zeros((n_paths, horizon_bars + 1), dtype=int)
        regime_durations = np.ones(n_paths, dtype=int)
        
        # Initial conditions
        prices[:, 0] = initial_price
        variances[:, 0] = initial_variance
        
        # Sample initial regimes from probabilities
        initial_regimes = np.random.choice(
            n_regimes, size=n_paths, p=initial_regime_probs
        )
        regimes[:, 0] = initial_regimes
        
        # Time tracking (minutes per bar)
        minutes_per_day = 23 * 60  # Futures trading hours
        
        for t in range(horizon_bars):
            # Current time of day
            current_minute = (time_of_day_start * 60 + t) % minutes_per_day
            current_hour = current_minute / 60
            
            # Partner regime for contagion (use mode if provided)
            if partner_regimes is not None and t < len(partner_regimes):
                partner_regime = int(partner_regimes[t])
            else:
                partner_regime = 1  # Default to "normal" regime
            
            # Get TVTP-adjusted transition matrix
            P = self.transition_matrix.get_adjusted(
                time_of_day_hour=current_hour,
                partner_regime=partner_regime,
                rv_ratio=rv_ratio,
            )
            
            # For each path, handle regime transition
            for i in range(n_paths):
                current_regime = regimes[i, t]
                
                # Sample next regime from transition probabilities
                proposed_regime = np.random.choice(n_regimes, p=P[current_regime])
                
                # Apply minimum duration constraint
                if proposed_regime != current_regime:
                    if regime_durations[i] >= min_duration:
                        regimes[i, t + 1] = proposed_regime
                        regime_durations[i] = 1
                    else:
                        regimes[i, t + 1] = current_regime
                        regime_durations[i] += 1
                else:
                    regimes[i, t + 1] = current_regime
                    regime_durations[i] += 1
                
                # Get regime-specific GARCH parameters
                regime_k = regimes[i, t + 1]
                params = self.regime_params.get_regime_params(regime_k)
                
                # GARCH variance update
                # σ²_t = ω + α*ε²_{t-1} + γ*ε²_{t-1}*I(ε<0) + β*σ²_{t-1}
                # For first step, use expected shock (variance)
                if t == 0:
                    last_shock_sq = variances[i, t]
                    leverage = 0.5 * params.gamma * variances[i, t]  # Expected leverage
                else:
                    last_return = np.log(prices[i, t] / prices[i, t - 1])
                    last_shock = last_return - params.mu
                    last_shock_sq = last_shock ** 2
                    leverage = params.gamma * last_shock_sq * (last_shock < 0)
                
                new_variance = (
                    params.omega +
                    params.alpha * last_shock_sq +
                    leverage +
                    params.beta * variances[i, t]
                )
                variances[i, t + 1] = max(new_variance, 1e-10)
                
                # Generate innovation
                z = self._generate_innovation(params.nu, params.skew, params.gamma > 0)
                
                # Generate return and update price
                sigma = np.sqrt(variances[i, t + 1])
                ret = params.mu + sigma * z
                prices[i, t + 1] = prices[i, t] * np.exp(ret)
        
        return prices, variances, regimes
    
    def _generate_innovation(
        self,
        nu: float,
        skew: float,
        use_skewed_t: bool
    ) -> float:
        """
        Generate a single innovation from the appropriate distribution.
        
        ES/NQ: Skewed-t distribution (leverage effect)
        GC/SI: Generalized Error Distribution (symmetric fat tails)
        """
        if use_skewed_t:
            # Skewed-t: Use t-distribution with skewness transformation
            z = np.random.standard_t(nu) if nu > 2 else np.random.standard_t(3)
            if abs(skew) > 1e-6:
                # Apply skewness via sinh transform
                z = np.sinh(skew * np.arcsinh(z) + np.arcsinh(skew))
            return z
        else:
            # GED: Approximate with scaled t-distribution
            # nu < 2 gives fatter tails
            if nu < 2:
                df = max(2.5, nu * 2)  # Map GED shape to t df
                z = np.random.standard_t(df)
            else:
                z = np.random.standard_normal()
            return z
```

---

## 3.8 Multi-Timeframe Aggregation

### 3.8.1 Bar Aggregator

```python
# utils/timeframe_aggregator.py

import numpy as np
import pandas as pd
from typing import Dict, Optional
from ..models.regime_config import Timeframe


class TimeframeAggregator:
    """
    Aggregate 1-minute generated paths to higher timeframes.
    
    Strategy: Generate at 1-minute resolution (most granular), then
    aggregate OHLCV to 5m, 15m, 1h as needed. This ensures consistency
    across timeframes while only calibrating once at 1-minute.
    """
    
    @staticmethod
    def aggregate_paths(
        prices_1m: np.ndarray,
        target_timeframe: Timeframe,
    ) -> np.ndarray:
        """
        Aggregate 1-minute price paths to target timeframe.
        
        Args:
            prices_1m: (n_paths, n_bars_1m) array of 1-minute prices
            target_timeframe: Target timeframe to aggregate to
            
        Returns:
            (n_paths, n_bars_target) array of close prices at target timeframe
        """
        if target_timeframe == Timeframe.M1:
            return prices_1m
        
        n_paths, n_bars_1m = prices_1m.shape
        agg_minutes = target_timeframe.minutes
        n_bars_target = n_bars_1m // agg_minutes
        
        # Take close price (last price in each aggregation window)
        prices_target = np.zeros((n_paths, n_bars_target))
        
        for i in range(n_bars_target):
            start_idx = i * agg_minutes
            end_idx = (i + 1) * agg_minutes
            # Close = last price in window
            prices_target[:, i] = prices_1m[:, end_idx - 1]
        
        return prices_target
    
    @staticmethod
    def aggregate_ohlcv(
        prices_1m: np.ndarray,
        target_timeframe: Timeframe,
    ) -> Dict[str, np.ndarray]:
        """
        Aggregate 1-minute price paths to OHLCV at target timeframe.
        
        Args:
            prices_1m: (n_paths, n_bars_1m) array of 1-minute prices
            target_timeframe: Target timeframe
            
        Returns:
            Dict with 'open', 'high', 'low', 'close' arrays
        """
        if target_timeframe == Timeframe.M1:
            return {
                'open': prices_1m[:, :-1],
                'high': prices_1m,
                'low': prices_1m,
                'close': prices_1m[:, 1:],
            }
        
        n_paths, n_bars_1m = prices_1m.shape
        agg_minutes = target_timeframe.minutes
        n_bars_target = (n_bars_1m - 1) // agg_minutes
        
        opens = np.zeros((n_paths, n_bars_target))
        highs = np.zeros((n_paths, n_bars_target))
        lows = np.zeros((n_paths, n_bars_target))
        closes = np.zeros((n_paths, n_bars_target))
        
        for i in range(n_bars_target):
            start_idx = i * agg_minutes
            end_idx = (i + 1) * agg_minutes + 1  # +1 because we need close of last bar
            
            window = prices_1m[:, start_idx:end_idx]
            
            opens[:, i] = window[:, 0]
            highs[:, i] = window.max(axis=1)
            lows[:, i] = window.min(axis=1)
            closes[:, i] = window[:, -1]
        
        return {
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
        }
```

---

## 3.9 Updated Configuration Templates

### model_params.yaml (Updated)

```yaml
# Regime-Switching GARCH Configuration
# Reference: Phase 1 GARCH MVP with TVTP and Cross-Instrument Contagion

regime_switching:
  # Number of regimes per instrument
  n_regimes:
    ES: 3
    NQ: 3
    GC: 3
    SI: 4  # Extra "crisis" regime for flash crashes
  
  # Regime names (sorted by volatility, low to high)
  regime_names:
    ES: ["Low", "Normal", "High"]
    NQ: ["Low", "Normal", "High"]
    GC: ["Low", "Normal", "High"]
    SI: ["Low", "Normal", "High", "Crisis"]
  
  # Minimum bars before regime can switch (prevents noise-driven micro-switches)
  min_regime_duration:
    M1: 5    # 5 minutes
    M5: 3    # 15 minutes
    M15: 2   # 30 minutes
    H1: 1    # 1 hour

tvtp:
  # Time-of-day high-volatility boost factor
  tod_high_vol_boost: 1.3
  
  # Cross-instrument contagion multiplier (SYMMETRIC - same both directions)
  # Actual values may be calibrated differently if lead-lag is detected
  default_contagion_multiplier: 1.4
  
  # Realized volatility adjustment strength
  rv_adjustment_strength: 0.2
  
  # High-volatility time windows (hours in ET)
  volatility_windows:
    ES_NQ:
      - {start: 9.0, end: 10.0, boost: 1.3}    # RTH open
      - {start: 11.0, end: 12.0, boost: 1.1}   # European close
      - {start: 15.5, end: 16.5, boost: 1.2}   # US close
    GC_SI:
      - {start: 2.5, end: 4.0, boost: 1.3}     # London open
      - {start: 9.5, end: 10.5, boost: 1.3}    # PM Fix

cross_instrument:
  # Instrument pairs for contagion modeling
  pairs:
    - [GC, SI]
    - [ES, NQ]
  
  # Lead-lag estimation settings
  lead_lag:
    max_lag: 10              # Maximum lag for Granger causality
    significance_level: 0.05 # p-value threshold
    
    # NOTE: Lead-lag is ESTIMATED from data, not assumed
    # Research shows conflicting evidence on GC/SI and ES/NQ leadership
    # Default to symmetric if estimation is inconclusive

# Per-regime GARCH parameters (starting points - calibrated from data)
# ES/NQ: Use GJR-GARCH (gamma > 0) with skewed-t
# GC/SI: Use standard GARCH (gamma = 0) with GED
regime_garch:
  ES:
    regime_0:  # Low volatility
      omega: 0.00001
      alpha: 0.05
      beta: 0.92
      gamma: 0.08
      nu: 8.0
      skew: -0.1
    regime_1:  # Normal
      omega: 0.00002
      alpha: 0.10
      beta: 0.85
      gamma: 0.12
      nu: 6.0
      skew: -0.15
    regime_2:  # High volatility
      omega: 0.00005
      alpha: 0.15
      beta: 0.78
      gamma: 0.15
      nu: 4.0
      skew: -0.2
  
  NQ:
    regime_0:
      omega: 0.00001
      alpha: 0.06
      beta: 0.90
      gamma: 0.10
      nu: 7.0
      skew: -0.12
    regime_1:
      omega: 0.00002
      alpha: 0.12
      beta: 0.82
      gamma: 0.14
      nu: 5.0
      skew: -0.18
    regime_2:
      omega: 0.00006
      alpha: 0.18
      beta: 0.74
      gamma: 0.18
      nu: 3.5
      skew: -0.25
  
  GC:
    # Symmetric (gamma = 0), GED distribution (nu = shape parameter)
    regime_0:
      omega: 0.00001
      alpha: 0.06
      beta: 0.91
      gamma: 0.0
      nu: 1.5  # GED shape (lower = fatter tails)
      skew: 0.0
    regime_1:
      omega: 0.00002
      alpha: 0.09
      beta: 0.88
      gamma: 0.0
      nu: 1.3
      skew: 0.0
    regime_2:
      omega: 0.00005
      alpha: 0.12
      beta: 0.83
      gamma: 0.0
      nu: 1.1
      skew: 0.0
  
  SI:
    # 4 regimes for silver (extreme kurtosis, flash crashes)
    regime_0:
      omega: 0.00002
      alpha: 0.06
      beta: 0.91
      gamma: 0.0
      nu: 1.2
      skew: 0.0
    regime_1:
      omega: 0.00003
      alpha: 0.10
      beta: 0.87
      gamma: 0.0
      nu: 1.1
      skew: 0.0
    regime_2:
      omega: 0.00008
      alpha: 0.15
      beta: 0.80
      gamma: 0.0
      nu: 1.0
      skew: 0.0
    regime_3:  # Crisis regime
      omega: 0.00020
      alpha: 0.25
      beta: 0.70
      gamma: 0.0
      nu: 0.8
      skew: 0.0

calibration:
  # Calibration always done at 1-minute, aggregated for other timeframes
  base_timeframe: M1
  
  # Calibration frequency by generation timeframe
  frequency:
    M1: daily
    M5: daily
    M15: daily
    H1: weekly
  
  # Lookback periods (in bars at base timeframe)
  lookback_bars: 100000  # ~70 days of 1-minute data
  min_observations_per_regime: 100
  
  # Model selection
  use_bic_for_n_regimes: true
  min_regimes: 2
  max_regimes: 6  # Test 2-6 regimes, select by BIC
```

---

## 3.10 Validation Checklist (Updated)

### 3.10.1 Regime-Switching Specific Validation

```
□ Regime detection produces K distinct volatility levels
□ Regime persistence: self-transition probability > 0.9
□ Regime durations: average duration > min_duration_bars
□ Cross-instrument contagion: regimes correlate between pairs
□ TVTP effects: transition probabilities vary by time-of-day
□ Lead-lag estimation: Granger causality tests run without errors
```

### 3.10.2 Stylized Facts (Extended)

```
□ Fat tails preserved across all regimes
□ Volatility clustering within and across regimes
□ ES/NQ: Leverage effect (negative return-volatility correlation)
□ GC/SI: Symmetric volatility response (no leverage effect)
□ Regime-conditional distributions match empirical
```

### 3.10.3 Cross-Instrument Validation

```
□ GC-SI correlation: 0.6-0.9 range
□ ES-NQ correlation: 0.8-0.95 range
□ Regime synchronization: pairs tend to be in similar regimes
□ Contagion effect: high-vol regime in one increases transition prob in other
```

---

*Section Updated: January 2026*
*Changes: Added regime-switching GARCH, TVTP, symmetric cross-instrument contagion*
*Research basis: Academic literature review on gold-silver and ES-NQ lead-lag relationships*
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
    # Regime-switching config
    n_regimes: 3
    use_asymmetric_garch: true  # GJR-GARCH for leverage effect
    innovation_distribution: skewed_t
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
    # Regime-switching config
    n_regimes: 3
    use_asymmetric_garch: true  # GJR-GARCH for leverage effect
    innovation_distribution: skewed_t
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
    # Regime-switching config
    n_regimes: 3
    use_asymmetric_garch: false  # Symmetric - no leverage effect in gold
    innovation_distribution: ged
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
    # Regime-switching config
    n_regimes: 4  # Extra "crisis" regime for flash crashes
    use_asymmetric_garch: false  # Symmetric - no leverage effect in silver
    innovation_distribution: ged
    # Session effects (architecture doc Section 13.3):
    # - Similar to gold but more volatile
    # - Industrial demand adds complexity
    # - Most prone to flash crashes/spikes

# Cross-instrument pair configuration
instrument_pairs:
  - pair: [GC, SI]
    contagion_type: symmetric_bidirectional
    default_multiplier: 1.4
    # Research Note: Academic evidence is CONFLICTING on GC-SI lead-lag
    # Some studies find gold leads, others find silver leads
    # We use symmetric contagion and estimate from data
    
  - pair: [ES, NQ]
    contagion_type: symmetric_bidirectional
    default_multiplier: 1.4
    # Research Note: No clear evidence either leads the other
    # NQ has higher beta but doesn't systematically lead ES
```

## model_params.yaml

```yaml
# Regime-Switching GARCH Configuration
# Reference: monte_carlo_architecture.md and Phase 1 GARCH MVP

# ============================================================================
# REGIME-SWITCHING CONFIGURATION
# ============================================================================

regime_switching:
  # Number of regimes per instrument (selected by BIC or specified)
  n_regimes:
    ES: 3
    NQ: 3
    GC: 3
    SI: 4  # Extra "crisis" regime for flash crashes
  
  # Regime names (sorted by volatility, low to high)
  regime_names:
    ES: ["Low", "Normal", "High"]
    NQ: ["Low", "Normal", "High"]
    GC: ["Low", "Normal", "High"]
    SI: ["Low", "Normal", "High", "Crisis"]
  
  # Minimum bars before regime can switch (prevents noise-driven micro-switches)
  min_regime_duration:
    M1: 5    # 5 minutes
    M5: 3    # 15 minutes
    M15: 2   # 30 minutes
    H1: 1    # 1 hour
  
  # Model selection
  use_bic_for_n_regimes: true
  min_regimes: 2
  max_regimes: 6

# ============================================================================
# TIME-VARYING TRANSITION PROBABILITIES (TVTP)
# ============================================================================

tvtp:
  # Time-of-day high-volatility boost factor
  tod_high_vol_boost: 1.3
  
  # Cross-instrument contagion multiplier
  # SYMMETRIC BY DEFAULT - same multiplier both directions
  # Actual values calibrated from data if lead-lag is detected
  default_contagion_multiplier: 1.4
  
  # Realized volatility adjustment strength
  rv_adjustment_strength: 0.2
  
  # High-volatility time windows (hours in ET)
  volatility_windows:
    ES_NQ:
      - {start: 9.0, end: 10.0, boost: 1.3}    # RTH open
      - {start: 11.0, end: 12.0, boost: 1.1}   # European close
      - {start: 15.5, end: 16.5, boost: 1.2}   # US close
    GC_SI:
      - {start: 2.5, end: 4.0, boost: 1.3}     # London open
      - {start: 9.5, end: 10.5, boost: 1.3}    # PM Fix

# ============================================================================
# CROSS-INSTRUMENT RELATIONSHIPS
# ============================================================================

cross_instrument:
  # Instrument pairs for contagion modeling
  pairs:
    - [GC, SI]
    - [ES, NQ]
  
  # Lead-lag estimation settings
  lead_lag:
    max_lag: 10              # Maximum lag for Granger causality
    significance_level: 0.05 # p-value threshold
    
    # IMPORTANT: Lead-lag is ESTIMATED from data, not hardcoded
    # Research shows conflicting evidence on GC/SI and ES/NQ leadership:
    # - Some studies: gold leads silver (spillover direction)
    # - Other studies: silver leads gold (Lau et al. 2017)
    # - ES/NQ: No clear evidence either leads
    # Solution: Default to symmetric, let data decide

# ============================================================================
# PER-REGIME GARCH PARAMETERS
# ============================================================================

# ES/NQ: Use GJR-GARCH (gamma > 0) with skewed-t distribution
# GC/SI: Use standard GARCH (gamma = 0) with GED distribution

regime_garch:
  ES:
    regime_0:  # Low volatility
      omega: 0.00001
      alpha: 0.05
      beta: 0.92
      gamma: 0.08    # Leverage effect
      nu: 8.0        # t-distribution df
      skew: -0.1     # Negative skew
    regime_1:  # Normal
      omega: 0.00002
      alpha: 0.10
      beta: 0.85
      gamma: 0.12
      nu: 6.0
      skew: -0.15
    regime_2:  # High volatility
      omega: 0.00005
      alpha: 0.15
      beta: 0.78
      gamma: 0.15
      nu: 4.0
      skew: -0.2
  
  NQ:
    regime_0:
      omega: 0.00001
      alpha: 0.06
      beta: 0.90
      gamma: 0.10
      nu: 7.0
      skew: -0.12
    regime_1:
      omega: 0.00002
      alpha: 0.12
      beta: 0.82
      gamma: 0.14
      nu: 5.0
      skew: -0.18
    regime_2:
      omega: 0.00006
      alpha: 0.18
      beta: 0.74
      gamma: 0.18
      nu: 3.5
      skew: -0.25
  
  GC:
    # Symmetric (gamma = 0), GED distribution
    regime_0:
      omega: 0.00001
      alpha: 0.06
      beta: 0.91
      gamma: 0.0      # No leverage effect in gold
      nu: 1.5         # GED shape (lower = fatter tails)
      skew: 0.0       # Symmetric
    regime_1:
      omega: 0.00002
      alpha: 0.09
      beta: 0.88
      gamma: 0.0
      nu: 1.3
      skew: 0.0
    regime_2:
      omega: 0.00005
      alpha: 0.12
      beta: 0.83
      gamma: 0.0
      nu: 1.1
      skew: 0.0
  
  SI:
    # 4 regimes for silver (extreme kurtosis, flash crashes)
    # Symmetric (gamma = 0), GED distribution
    regime_0:
      omega: 0.00002
      alpha: 0.06
      beta: 0.91
      gamma: 0.0
      nu: 1.2
      skew: 0.0
    regime_1:
      omega: 0.00003
      alpha: 0.10
      beta: 0.87
      gamma: 0.0
      nu: 1.1
      skew: 0.0
    regime_2:
      omega: 0.00008
      alpha: 0.15
      beta: 0.80
      gamma: 0.0
      nu: 1.0
      skew: 0.0
    regime_3:  # Crisis regime (flash crashes)
      omega: 0.00020
      alpha: 0.25
      beta: 0.70
      gamma: 0.0
      nu: 0.8
      skew: 0.0

# ============================================================================
# CALIBRATION SETTINGS
# ============================================================================

calibration:
  # Base timeframe for calibration (generate at 1m, aggregate up)
  base_timeframe: M1
  
  # Calibration frequency by generation timeframe
  frequency:
    M1: daily
    M5: daily
    M15: daily
    H1: weekly
  
  # Lookback periods
  lookback_bars: 100000  # ~70 days of 1-minute data
  min_observations_per_regime: 100
  
  # Exponential weighting for parameter estimation
  use_exponential_weighting: true
  halflife_days: 30

# ============================================================================
# LEGACY SINGLE-REGIME GARCH (for comparison/fallback)
# ============================================================================

garch:
  # Single-regime GARCH parameters (fallback if regime detection fails)
  ES: {omega: 0.00002, alpha: 0.10, beta: 0.88, nu: 6.0, skew: -0.1}
  NQ: {omega: 0.00002, alpha: 0.12, beta: 0.85, nu: 5.0, skew: -0.15}
  GC: {omega: 0.00002, alpha: 0.08, beta: 0.90, nu: 1.3, skew: 0.0}
  SI: {omega: 0.00003, alpha: 0.10, beta: 0.87, nu: 1.1, skew: 0.0}

agarch:
  # Asymmetric GARCH (GJR-GARCH) with leverage effect
  ES: {omega: 0.00002, alpha: 0.10, gamma: 0.12, beta: 0.88}
  NQ: {omega: 0.00002, alpha: 0.12, gamma: 0.14, beta: 0.85}
  GC: {omega: 0.00002, alpha: 0.08, gamma: 0.03, beta: 0.90}
  SI: {omega: 0.00003, alpha: 0.10, gamma: 0.04, beta: 0.87}

# ============================================================================
# FUTURE PHASES (unchanged)
# ============================================================================

heston:
  ES: {kappa: 3.0, theta: 0.04, xi: 0.50, rho: -0.80}
  NQ: {kappa: 2.5, theta: 0.05, xi: 0.55, rho: -0.85}
  GC: {kappa: 2.0, theta: 0.03, xi: 0.45, rho: -0.15}
  SI: {kappa: 1.8, theta: 0.05, xi: 0.60, rho: -0.10}

jumps:
  ES: {lambda: 2, mu_jump: -0.03, sigma_jump: 0.05}
  NQ: {lambda: 4, mu_jump: -0.03, sigma_jump: 0.06}
  GC: {lambda: 6, mu_jump: -0.02, sigma_jump: 0.08}
  SI: {lambda: 10, mu_jump: -0.02, sigma_jump: 0.12}
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
  # Extended state for regime-switching
  track_regime_probabilities: true
  
position_management:
  risk_per_trade: 0.02
  max_position_fraction: 0.10
  min_entry_confidence: high
  min_adjust_confidence: moderate
```

---

*Document Version: 2.0*
*Last Updated: January 2026*
*Changes: Updated Phase 1 with regime-switching GARCH, TVTP, symmetric cross-instrument contagion*
*Research Basis: Academic literature review on gold-silver and ES-NQ lead-lag relationships*
*Reference: monte_carlo_architecture.md*
