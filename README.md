<div align="right">

**Language**　**English**　|　[中文](README.zh-CN.md)

</div>

<h1 align="center">🏀 NBA Game Intelligence Prediction Platform</h1>

<p align="center">
  <sub>Flask · Multi-model scoring · Multi-source data fusion · Real-time player news</sub>
</p>

<p align="center">
  <img alt="python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="flask" src="https://img.shields.io/badge/Flask-3.x-000000?logo=flask&logoColor=white">
  <img alt="version" src="https://img.shields.io/badge/version-2.0.0-C4612F">
</p>

---

## 📖 Overview

An **NBA game-outcome prediction platform** organised around three layers —
*structured data → multi-source fusion → real-time intelligence* — exposed as
three independently usable modules behind one web interface. The backend is
Flask; the frontend is plain HTML/CSS/JavaScript with no build step.

## 🧩 Modules

| Module | Page | What it does |
| --- | --- | --- |
| **1 · Custom data scoring** | `/` | Upload your own differential-feature file and score home/away winners with win probabilities across the whole model catalogue |
| **2 · Multi-source fusion** | `/module2` | Combine historical box-score form with live injury, news and tactical intelligence into a fused prediction with ranked factors |
| **3 · Player news** | `/module3` | Pull the latest reports on both rosters and produce a consolidated matchup verdict |

## ✨ Features

| Capability | Description |
| --- | --- |
| Batch scoring | One upload scores many games against many models, grouped per game |
| Model comparison | Six models side by side, plus a voting-based ensemble verdict |
| Payload templates | Built-in cross-sectional and time-series CSV templates |
| Streaming retrieval | Evidence sweeps are pushed over SSE as they arrive |
| Diagnostics | `/api/health` and `/api/diagnostics` report routes, dataset state and credential fingerprints |

## 🚀 Quick start

```bash
# 1) environment
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux

# 2) dependencies
pip install -r requirements.txt

# 3) run
python app.py
```

Then open <http://127.0.0.1:3344>

| Page | URL | Purpose |
| --- | --- | --- |
| Home | `/` | Module 1: custom data scoring |
| Fusion | `/module2` | Module 2: multi-source fusion |
| Player news | `/module3` | Module 3: real-time roster news |
| Health | `/api/health` | Liveness probe |

## ⚙️ Configuration

**Every API key, URL, model identifier and filesystem path lives in `constants.py`;
no other file contains a credential literal.** Each value can be overridden by an
environment variable, which is the recommended approach in production. Copy
`.env.example` and populate it, or export the variables from your process manager.

| Variable | Configuration | Purpose |
| --- | --- | --- |
| `NBA_COMPLETION_API_KEY` | primary gateway credential | structured prediction calls |
| `NBA_COMPLETION_ENDPOINT` | primary gateway URL | OpenAI-compatible `chat/completions` |
| `NBA_COMPLETION_MODEL` | primary gateway model id | default remote model |
| `NBA_SEARCH_API_KEY` | retrieval gateway credential | hosted web-search retrieval |
| `NBA_SEARCH_ENDPOINT` | retrieval gateway URL | `responses`-style surface |
| `NBA_SEARCH_MODEL` | retrieval gateway model id | default remote model |
| `NBA_DATASET_FILE` | historical dataset path | defaults to the bundled CSV |
| `NBA_ROSTER_FILE` | roster catalogue path | defaults to `nba_teams_players.jsonl` |
| `NBA_UPLOAD_DIR` | upload directory | defaults to `uploads/` |
| `NBA_PORT` / `NBA_HOST` / `NBA_DEBUG` | server binding | `3344` / `0.0.0.0` / on |

Inspect the active routing table and credential fingerprints (keys are shown by
their last four characters only):

```bash
curl http://127.0.0.1:3344/api/diagnostics
```

## 📁 Project layout

```
nba_predictor/
├── app.py                          # entry point (WSGI callable: app)
├── constants.py                    # ★ every key, URL, model id and path
├── compatibility.py                # adapters for the pre-2.x public functions
├── utils.py                        # stable import surface for gateway helpers
├── process_data.py                 # dataset facade + rolling-form aggregation
├── nba_core/
│   ├── data/                       # raw feature-matrix repository
│   ├── features/                   # prompt library, numeric helpers, samples
│   ├── inference/                  # outbound gateway client and transport
│   ├── serving/                    # scoring pipeline, registry, batch orchestration
│   ├── services/                   # uploads, templates, roster, fusion, news, SSE
│   └── web/                        # application factory and blueprints
├── model_code/                     # per-model training scripts and unified interface
├── templates/                      # index.html / module2.html / module3.html
├── uploads/                        # uploaded payloads and generated templates
├── .tools/                         # offline verifier and dependency vendoring
├── requirements.txt
├── README.md                       # this file (English)
└── README.zh-CN.md                 # 中文说明
```

## 🧱 Architecture

Dependencies flow strictly one way — outer layers call inner ones, never the reverse:

```
Web tier        app.py · nba_core/web/           HTTP framing and blueprint registration
Services        nba_core/services/               uploads, fusion retrieval, roster news
Serving         nba_core/serving/                registry, scoring pipeline, batch execution
Evidence        nba_core/serving/evidence.py     envelope resolution and caching
Inference       nba_core/inference/              outbound framing, retries, stream decoding
Configuration   constants.py                     every endpoint, credential, catalogue and path
```

A module-1 scoring call runs **normalise → estimator-family score → calibration →
verdict**. Distinct rows inside one batch collapse onto the same cached scoring
envelope, so a repeated feature vector is never resolved twice.

## 🔌 API reference

<details>
<summary><b>Show the complete endpoint list</b></summary>

**Module 1**

| Method | Path | Description |
| --- | --- | --- |
| GET | `/` | Module 1 page |
| GET | `/api/models` | Selectable model catalogue |
| GET | `/api/download_template/<single\|time_series>` | Download a payload template |
| POST | `/api/upload` | Upload and validate a payload |
| POST | `/api/predict` | Batch scoring with selected models |
| POST | `/api/predict_comparison` | All models plus ensemble vote |

**Module 2**

| Method | Path | Description |
| --- | --- | --- |
| GET | `/module2` | Fusion page |
| GET | `/api/module2/seasons` | Available seasons |
| GET | `/api/module2/teams?season=` | Teams in a season |
| GET | `/api/module2/game_dates?season=&home_team=&away_team=` | Matchup dates |
| POST | `/api/module2/structured_data` | Structured statistical view |
| POST | `/api/module2/search_unstructured` | Six-lane concurrent sweep (SSE) |
| POST | `/api/module2/summarize_and_predict` | Condensation and fused prediction |

**Module 3**

| Method | Path | Description |
| --- | --- | --- |
| GET | `/module3` | Player-news page |
| GET | `/api/module3/teams` | Teams and rosters |
| POST | `/api/module3/search_news` | Both-team news sweep (SSE) |
| POST | `/api/module3/final_prediction` | Final verdict |

</details>

## 📄 Data format

`POST /api/upload` accepts CSV / XLSX. The identifier columns `h_team_name` and
`o_team_name` are mandatory, and at least one differential feature column must be
present:

```
diff_MIN  diff_FGM  diff_FGA  diff_FG%   diff_3PM  diff_3PA  diff_3P%
diff_FTM  diff_FTA  diff_FT%   diff_OREB  diff_DREB  diff_REB
diff_AST  diff_STL  diff_BLK   diff_TOV   diff_PF
```

Example request:

```bash
curl -X POST http://127.0.0.1:3344/api/predict \
  -H "Content-Type: application/json" \
  -d '{"filename":"20240101_120000_games.csv","models":["random_forest","svm"]}'
```

## ❓ FAQ

**The dataset is missing and module 2 reports "Data not loaded".**
Make sure `nba_team_boxscores_features_2015_16_to_2025_26.csv` is present, or point
`NBA_DATASET_FILE` at it.

**What happens when an external gateway call fails?**
Module 1 falls back to a deterministic statistical strategy so the endpoint always
returns an explainable result; modules 2 and 3 surface the error inline.

**How do I switch inference providers?**
Edit `endpoint` / `api_key` / `model` inside `INFERENCE_ROUTES` in `constants.py`.
No business code changes are required.

**How do I verify the installation?**
`python .tools/smoke_test.py` boots the app with Flask's test client and asserts
every route, page and payload shape. See `.tools/README.md` for details.

---

<div align="center">
<sub>NBA Game Intelligence Prediction Platform · v2.0.0 ·
<a href="README.zh-CN.md">中文说明</a></sub>
</div>
