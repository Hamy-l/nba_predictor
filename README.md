<div align="right">

**🌐 Language / 语言**　[**中文**](#)　|　[**English**](#)

</div>

<h1 align="center">🏀 NBA Game Intelligence Prediction Platform</h1>

<p align="center">
  <b>NBA 比赛智能预测平台</b><br>
  <sub>Flask · Multi-model scoring · Multi-source data fusion · Real-time player news</sub>
</p>

<p align="center">
  <img alt="python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="flask" src="https://img.shields.io/badge/Flask-3.x-000000?logo=flask&logoColor=white">
  <img alt="version" src="https://img.shields.io/badge/version-2.0.0-C4612F">
  <img alt="license" src="https://img.shields.io/badge/usage-course%20project-5C635D">
</p>

---

<!-- ══════════════════════════════ 中文 ══════════════════════════════ -->
<div data-lang="zh">

## 📖 项目简介

本项目是一个 **NBA 比赛胜负预测平台**，围绕「结构化数据 → 多源信息融合 → 实时情报」三个层次，
提供三个可独立使用的功能模块与统一的 Web 界面。平台后端基于 Flask，前端为原生
HTML/CSS/JavaScript，无需构建步骤即可运行。

- **模块 1 · 自定义数据在线预测**：上传自己的比赛差分特征文件，选择模型批量预测主/客队胜负与胜率。
- **模块 2 · 多源数据融合预测**：结合历史技术统计与实时伤停、新闻、战术情报，输出融合预测与 Top 5 影响因素。
- **模块 3 · 球员实时新闻预测**：拉取双方球队核心球员的最新动态，生成综合分析与赛果判断。

## ✨ 功能特性

| 能力 | 说明 |
| --- | --- |
| 批量预测 | 一次上传可对多场比赛、多个模型并发打分，结果按比赛分组展示 |
| 多模型对比 | 六个模型同场竞技，附带投票法集成预测结果 |
| 数据模板 | 内置横截面与时间序列两套 CSV 模板，下载即用 |
| 流式检索 | 情报检索采用 SSE 真流式推送，边检索边展示 |
| 诊断接口 | `/api/health`、`/api/diagnostics` 用于运行时自检（路由、数据集、凭证指纹） |

## 🧱 系统架构

平台采用严格单向依赖的分层结构，外层只调用内层，内层不感知外层：

```
Web 层        app.py · nba_core/web/           仅负责 HTTP 编解码与蓝图注册
服务层        nba_core/services/               工作流编排：上传、融合检索、名单新闻
评分层        nba_core/serving/                模型注册表、评分流水线、批量执行
证据层        nba_core/serving/evidence.py     评分信封解析与缓存
推理层        nba_core/inference/              出站请求封装、重试、流式解码
配置层        constants.py                     全部端点、密钥、模型目录与路径
```

模块 1 的一次打分请求依次经过：**特征归一化 → 估计器族打分 → 概率校准 → 裁定**，
同一批次内重复的特征向量会命中信封缓存，避免重复解析。

## 📁 目录结构

```
nba_predictor/
├── app.py                          # 应用入口（WSGI callable: app）
├── constants.py                    # ★ 所有 API Key / URL / 模型目录 / 路径
├── compatibility.py                # 旧版公开函数适配层
├── utils.py                        # 网关工具的稳定导入面
├── process_data.py                 # 数据集门面（滚动状态聚合）
├── nba_core/
│   ├── data/                       # 原始特征矩阵仓库
│   ├── features/                   # 提示词库、数值归一化、模板样本
│   ├── inference/                  # 出站网关客户端与传输工具
│   ├── serving/                    # 评分流水线、模型注册表、批量编排
│   ├── services/                   # 上传、模板、名单、融合、新闻、SSE
│   └── web/                        # 应用工厂与蓝图
├── model_code/                     # 各模型训练脚本与统一预测接口
├── templates/                      # index.html / module2.html / module3.html
├── uploads/                        # 用户上传数据与生成的数据模板
├── requirements.txt
└── README.md
```

## 🚀 快速开始

```bash
# 1) 环境准备
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux

# 2) 安装依赖
pip install -r requirements.txt

# 3) 启动服务
python app.py
```

打开浏览器访问 <http://127.0.0.1:3344>

| 页面 | 地址 | 用途 |
| --- | --- | --- |
| 首页 | `/` | 模块 1：自定义数据在线预测 |
| 融合预测 | `/module2` | 模块 2：多源数据融合 |
| 球员情报 | `/module3` | 模块 3：实时球员新闻预测 |
| 健康检查 | `/api/health` | 服务存活探针 |

## ⚙️ 配置说明

**所有 API Key、URL、模型名与路径集中在 `constants.py`，其它文件不出现任何密钥字面量。**
每个配置项都支持环境变量覆盖，生产环境建议通过环境变量注入：

| 环境变量 | 对应配置 | 说明 |
| --- | --- | --- |
| `NBA_COMPLETION_API_KEY` | 主推理网关密钥 | 结构化预测调用 |
| `NBA_COMPLETION_ENDPOINT` | 主推理网关地址 | OpenAI 兼容 `chat/completions` |
| `NBA_COMPLETION_MODEL` | 主推理网关模型名 | 默认模型标识 |
| `NBA_SEARCH_API_KEY` | 检索网关密钥 | 联网情报检索 |
| `NBA_SEARCH_ENDPOINT` | 检索网关地址 | `responses` 形态接口 |
| `NBA_SEARCH_MODEL` | 检索网关模型名 | 默认模型标识 |
| `NBA_DATASET_FILE` | 历史数据集路径 | 默认使用仓库内 CSV |
| `NBA_UPLOAD_DIR` | 上传目录 | 默认 `uploads/` |
| `NBA_PORT` / `NBA_HOST` / `NBA_DEBUG` | 服务监听配置 | 默认 `3344` / `0.0.0.0` / 开启 |

查看当前生效的路由与凭证指纹（密钥仅显示尾四位）：

```bash
curl http://127.0.0.1:3344/api/diagnostics
```

## 🔌 API 一览

<details>
<summary><b>展开完整接口列表</b></summary>

**模块 1**

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 模块 1 页面 |
| GET | `/api/models` | 可选模型目录 |
| GET | `/api/download_template/<single\|time_series>` | 下载数据模板 |
| POST | `/api/upload` | 上传并校验数据文件 |
| POST | `/api/predict` | 指定模型批量预测 |
| POST | `/api/predict_comparison` | 全模型对比 + 集成投票 |

**模块 2**

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/module2` | 融合预测页面 |
| GET | `/api/module2/seasons` | 可选赛季 |
| GET | `/api/module2/teams?season=` | 赛季球队列表 |
| GET | `/api/module2/game_dates?season=&home_team=&away_team=` | 交手日期 |
| POST | `/api/module2/structured_data` | 结构化统计视图 |
| POST | `/api/module2/search_unstructured` | 六路并发情报检索（SSE 流式） |
| POST | `/api/module2/summarize_and_predict` | 情报归纳 + 融合预测 |

**模块 3**

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/module3` | 球员情报页面 |
| GET | `/api/module3/teams` | 球队与球员名单 |
| POST | `/api/module3/search_news` | 双方新闻检索（SSE 流式） |
| POST | `/api/module3/final_prediction` | 最终赛果分析 |

</details>

## 📄 数据格式

`POST /api/upload` 接受 CSV / XLSX，必须包含标识列 `h_team_name`、`o_team_name`，
并至少包含一个差分特征列：

```
diff_MIN  diff_FGM  diff_FGA  diff_FG%   diff_3PM  diff_3PA  diff_3P%
diff_FTM  diff_FTA  diff_FT%   diff_OREB  diff_DREB  diff_REB
diff_AST  diff_STL  diff_BLK   diff_TOV   diff_PF
```

请求示例：

```bash
curl -X POST http://127.0.0.1:3344/api/predict \
  -H "Content-Type: application/json" \
  -d '{"filename":"20240101_120000_games.csv","models":["random_forest","svm"]}'
```

## ❓ 常见问题

**Q：数据集没加载，模块 2 报 “Data not loaded”？**
检查 `nba_team_boxscores_features_2015_16_to_2025_26.csv` 是否存在，或用 `NBA_DATASET_FILE` 指定路径。

**Q：外部网关调用失败会怎样？**
模块 1 会退回确定性的统计回退策略，保证接口始终返回可解释的结果；模块 2/3 会在页面内提示错误信息。

**Q：如何换一个推理服务商？**
只需修改 `constants.py` 中 `INFERENCE_ROUTES` 的 `endpoint` / `api_key` / `model`，业务代码无需改动。

</div>

<!-- ══════════════════════════════ English ══════════════════════════════ -->
<div data-lang="en">

## 📖 Overview

An **NBA game-outcome prediction platform** organised around three layers —
*structured data → multi-source fusion → real-time intelligence* — exposed as
three independently usable modules behind one web interface. The backend is
Flask; the frontend is plain HTML/CSS/JavaScript with no build step.

- **Module 1 · Custom data scoring** — upload your own differential-feature file and score
  home/away winners with win probabilities across the model catalogue.
- **Module 2 · Multi-source fusion** — combine historical box-score form with live injury,
  news and tactical intelligence into a fused prediction with ranked factors.
- **Module 3 · Player news** — pull the latest reports on both rosters and produce a
  consolidated matchup verdict.

## ✨ Features

| Capability | Description |
| --- | --- |
| Batch scoring | One upload scores many games against many models, grouped per game |
| Model comparison | Six models side by side, plus a voting-based ensemble verdict |
| Payload templates | Built-in cross-sectional and time-series CSV templates |
| Streaming retrieval | Evidence sweeps are pushed over SSE as they arrive |
| Diagnostics | `/api/health` and `/api/diagnostics` report routes, dataset state and credential fingerprints |

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
├── requirements.txt
└── README.md
```

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
environment variable, which is the recommended approach in production:

| Variable | Configuration | Purpose |
| --- | --- | --- |
| `NBA_COMPLETION_API_KEY` | primary gateway credential | structured prediction calls |
| `NBA_COMPLETION_ENDPOINT` | primary gateway URL | OpenAI-compatible `chat/completions` |
| `NBA_COMPLETION_MODEL` | primary gateway model id | default remote model |
| `NBA_SEARCH_API_KEY` | retrieval gateway credential | hosted web-search retrieval |
| `NBA_SEARCH_ENDPOINT` | retrieval gateway URL | `responses`-style surface |
| `NBA_SEARCH_MODEL` | retrieval gateway model id | default remote model |
| `NBA_DATASET_FILE` | historical dataset path | defaults to the bundled CSV |
| `NBA_UPLOAD_DIR` | upload directory | defaults to `uploads/` |
| `NBA_PORT` / `NBA_HOST` / `NBA_DEBUG` | server binding | `3344` / `0.0.0.0` / on |

Inspect the active routing table and credential fingerprints (keys are shown by
their last four characters only):

```bash
curl http://127.0.0.1:3344/api/diagnostics
```

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

</div>

---

<div align="center">
<sub>NBA Game Intelligence Prediction Platform · v2.0.0</sub>
</div>

<script>
(function () {
  var nodes = document.querySelectorAll('[data-lang]');
  var links = document.querySelectorAll('a[href="#"]');

  function apply(lang) {
    for (var i = 0; i < nodes.length; i += 1) {
      nodes[i].style.display = nodes[i].getAttribute('data-lang') === lang ? '' : 'none';
    }
    try { window.localStorage.setItem('readme-lang', lang); } catch (e) { /* ignore */ }
  }

  var stored = 'zh';
  try { stored = window.localStorage.getItem('readme-lang') || 'zh'; } catch (e) { /* ignore */ }
  apply(stored);

  for (var j = 0; j < links.length; j += 1) {
    links[j].addEventListener('click', function (event) {
      event.preventDefault();
      apply(this.textContent.trim().toLowerCase().indexOf('english') === 0 ? 'en' : 'zh');
    });
  }
})();
</script>
