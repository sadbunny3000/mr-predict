# 🎯 Mr Predict

> **AI-powered sports prediction platform** with real-time Telegram alerts for football and tennis matches.

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue?logo=postgresql)](https://postgresql.org)
[![Railway](https://img.shields.io/badge/Deployed-Railway-purple?logo=railway)](https://railway.app)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange)](https://xgboost.readthedocs.io)

---

## 📌 What It Does

Mr Predict is a full-stack machine learning platform that:

- **Predicts football match outcomes** (winner, goals, corners) using Elo ratings and 84 engineered features
- **Predicts tennis match winners and total games** using surface-specific Elo, serve statistics, fatigue metrics, and head-to-head history
- **Sends automated Telegram alerts** 3 hours before kick-off / match start
- **Tracks betting value** by comparing model probabilities against real bookmaker odds
- **Runs fully autonomously** on a cloud deployment with an hourly scheduler

---

## 🏗️ Architecture

```
mr-predict/
├── backend/
│   ├── app/
│   │   ├── api/routes/          # FastAPI endpoints
│   │   ├── services/
│   │   │   ├── alert_engine.py          # Football alert engine
│   │   │   └── tennis_alert_engine.py   # Tennis alert engine v2
│   │   ├── config.py
│   │   ├── database.py
│   │   └── main.py
│   ├── tennis/
│   │   ├── ingestion/
│   │   │   ├── sackmann_ingestion.py    # ATP match data loader
│   │   │   └── odds_ingestion.py        # Bookmaker odds loader
│   │   └── features/
│   │       └── feature_builder.py       # Elo + serve stats + fatigue
│   ├── data_pipeline/                   # Football data ingestion
│   ├── ml/
│   │   └── saved_models/
│   │       ├── tennis_winner_model.pkl
│   │       └── tennis_total_games_model.pkl
│   └── migrations/
├── docker-compose.yml
└── railway.toml
```

---

## 🤖 Machine Learning Models

### Football
| Model | Algorithm | Accuracy / MAE |
|-------|-----------|----------------|
| Match Outcome (1X2) | XGBoost + Calibration | 45.8% (3-class) |
| Goals | Gradient Boosting | MAE 1.38 |
| Corners | Gradient Boosting | Real corner data, PL/Championship/La Liga |

**Features:** 84 features including Elo ratings, rest days, fixture congestion, team form, head-to-head

### Tennis
| Model | Algorithm | Accuracy / MAE |
|-------|-----------|----------------|
| Match Winner | XGBoost + Calibration | **65.4%** (vs 63.9% Elo baseline) |
| Total Games | XGBoost Regression | **MAE 5.37** (vs 6.65 baseline) |

**Features:** Overall Elo, surface-specific Elo (Hard/Clay/Grass), serve/return statistics, fatigue metrics (matches in last 14 days, days since last match), head-to-head history, career match count

**Training data:** 106,716 ATP matches (1991–2024) from Jeff Sackmann's dataset via Kaggle

---

## 📡 Alert System

Telegram alerts fire automatically **3 hours before each match**, containing:

**Football alerts:**
- Predicted outcome with probabilities
- Goals and corners predictions
- Value bet flags where model edge > market

**Tennis alerts (v2):**
- Match winner prediction with confidence %
- Model edge vs bookmaker implied probability
- Total games prediction with 90%/95% confidence intervals
- Over/Under recommendation with Kelly stake sizing

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| API | FastAPI + Uvicorn |
| Database | PostgreSQL 16 |
| Cache | Redis 7 |
| ML | XGBoost, Scikit-learn, Pandas, NumPy |
| Scheduler | APScheduler (hourly) |
| Deployment | Railway (auto-deploy from GitHub) |
| Containerisation | Docker + Docker Compose |
| Data sources | API-Football, Jeff Sackmann ATP dataset, tennis-data.co.uk, The Odds API |
| Alerts | Telegram Bot API |

---

## 🚀 Getting Started

### Prerequisites
- Docker Desktop
- Python 3.12
- A free [API-Football](https://api-football.com) account
- A free [The Odds API](https://the-odds-api.com) account
- A Telegram bot token (create via [@BotFather](https://t.me/botfather))

### Local Setup

**1. Clone the repo:**
```bash
git clone https://github.com/sadbunny3000/mr-predict.git
cd mr-predict
```

**2. Create your environment file:**
```bash
cp backend/.env.example backend/.env
# Fill in your real API keys in backend/.env
```

**3. Start all services:**
```bash
docker-compose up -d
```

**4. Run database migrations:**
```bash
docker exec -it football_backend alembic upgrade head
```

**5. Ingest football data:**
```bash
docker exec -it football_backend python3 data_pipeline/ingestion/match_ingestion.py
```

**6. Train models:**
```bash
docker exec -it football_backend python3 ml/train_models.py
```

The API will be available at `http://localhost:8000/docs`

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/api/v1/matches` | Upcoming matches |
| GET | `/api/v1/predictions` | Current predictions |
| POST | `/api/v1/alerts/run` | Trigger alert engine manually |
| POST | `/api/v1/tennis/predict` | Tennis match prediction |

---

## 📊 Data Sources

| Source | Data | Coverage |
|--------|------|----------|
| [API-Football](https://api-football.com) | Live matches, odds, stats | Premier League, Championship, La Liga |
| [Jeff Sackmann ATP Dataset](https://github.com/JeffSackmann/tennis_atp) | Historical ATP matches | 1968–2024 |
| [tennis-data.co.uk](http://www.tennis-data.co.uk) | ATP bookmaker odds | 2000–2024 |
| [The Odds API](https://the-odds-api.com) | Live tennis odds | Real-time |

---

## ⚙️ Environment Variables

Copy `backend/.env.example` to `backend/.env` and fill in:

```env
ODDS_API_KEY=            # The Odds API key
TENNIS_ODDS_API_KEY=     # Separate key for tennis (or same key)
TELEGRAM_BOT_TOKEN=      # From @BotFather
TELEGRAM_CHAT_ID=        # Your Telegram chat/channel ID
DATABASE_URL=            # PostgreSQL connection string
```

---

## 📈 Backtesting Results

Tennis match winner model backtested against real Bet365 closing odds (2020–2023, 10,075 matched records):

- Model consistently identifies value opportunities where predicted probability exceeds market-implied probability
- Calibration score (Brier): **0.213** — well-calibrated probabilities
- Full year 2020–2023 coverage at **96–98% match rate** against bookmaker records

---

## 🗺️ Roadmap

- [ ] WTA tennis support
- [ ] Tennis total games model v2 (quantile regression)
- [ ] Closing Line Value (CLV) tracking
- [ ] Web dashboard for prediction history
- [ ] Bankroll management / Kelly criterion staking

---

## 👤 Author

**Natangwe Martin** — [@sadbunny3000](https://github.com/sadbunny3000)

Built as a solo project combining sports analytics, machine learning, and quantitative betting research.

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. Predictions are probabilistic and not financial advice. Always gamble responsibly.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
