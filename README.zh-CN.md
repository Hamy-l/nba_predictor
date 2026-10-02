<div align="right">

**语言**　[English](README.md)　|　**中文**

</div>

<h1 align="center">🏀 NBA 比赛智能预测平台</h1>

<p align="center">
  <sub>Flask · 多模型评分 · 多源数据融合 · 实时球员情报</sub>
</p>

<p align="center">
  <img alt="python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="flask" src="https://img.shields.io/badge/Flask-3.x-000000?logo=flask&logoColor=white">
  <img alt="version" src="https://img.shields.io/badge/version-2.0.0-C4612F">
</p>

---

## 📖 项目简介

本项目是一个 **NBA 比赛胜负预测平台**，围绕「结构化数据 → 多源信息融合 → 实时情报」
三个层次，提供三个可独立使用的功能模块与统一的 Web 界面。后端基于 Flask，前端为原生
HTML/CSS/JavaScript，无需构建步骤即可运行。

## 🧩 功能模块

| 模块 | 页面 | 说明 |
| --- | --- | --- |
| **模块 1 · 自定义数据在线预测** | `/` | 上传自己的比赛差分特征文件，选择模型批量预测主/客队胜负与胜率 |
| **模块 2 · 多源数据融合预测** | `/module2` | 结合历史技术统计与实时伤停、新闻、战术情报，输出融合预测与 Top 5 影响因素 |
| **模块 3 · 球员实时新闻预测** | `/module3` | 拉取双方球队核心球员的最新动态，生成综合分析与赛果判断 |

## ✨ 功能特性

| 能力 | 说明 |
| --- | --- |
| 批量预测 | 一次上传可对多场比赛、多个模型并发打分，结果按比赛分组展示 |
| 多模型对比 | 六个模型同场竞技，附带投票法集成预测结果 |
| 数据模板 | 内置横截面与时间序列两套 CSV 模板，下载即用 |
| 流式检索 | 情报检索采用 SSE 真流式推送，边检索边展示 |
| 诊断接口 | `/api/health`、`/api/diagnostics` 用于运行时自检（路由、数据集、凭证指纹） |

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
每个配置项都支持环境变量覆盖，生产环境建议通过环境变量注入。可直接复制 `.env.example`
填写，或在进程管理器中导出这些变量。

| 环境变量 | 对应配置 | 说明 |
| --- | --- | --- |
| `NBA_COMPLETION_API_KEY` | 主推理网关密钥 | 结构化预测调用 |
| `NBA_COMPLETION_ENDPOINT` | 主推理网关地址 | OpenAI 兼容 `chat/completions` |
| `NBA_COMPLETION_MODEL` | 主推理网关模型名 | 默认模型标识 |
| `NBA_SEARCH_API_KEY` | 检索网关密钥 | 联网情报检索 |
| `NBA_SEARCH_ENDPOINT` | 检索网关地址 | `responses` 形态接口 |
| `NBA_SEARCH_MODEL` | 检索网关模型名 | 默认模型标识 |
| `NBA_DATASET_FILE` | 历史数据集路径 | 默认使用仓库内 CSV |
| `NBA_ROSTER_FILE` | 名单目录路径 | 默认 `nba_teams_players.jsonl` |
| `NBA_UPLOAD_DIR` | 上传目录 | 默认 `uploads/` |
| `NBA_PORT` / `NBA_HOST` / `NBA_DEBUG` | 服务监听配置 | 默认 `3344` / `0.0.0.0` / 开启 |

查看当前生效的路由与凭证指纹（密钥仅显示尾四位）：

```bash
curl http://127.0.0.1:3344/api/diagnostics
```

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
├── .tools/                         # 离线校验脚本与依赖离线安装工具
├── requirements.txt
├── README.md                       # English
└── README.zh-CN.md                 # 本文件（中文）
```

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

**数据集没加载，模块 2 报 “Data not loaded”？**
检查 `nba_team_boxscores_features_2015_16_to_2025_26.csv` 是否存在，或用
`NBA_DATASET_FILE` 指定路径。

**外部网关调用失败会怎样？**
模块 1 会退回确定性的统计回退策略，保证接口始终返回可解释的结果；模块 2/3 会在
页面内提示错误信息。

**如何换一个推理服务商？**
只需修改 `constants.py` 中 `INFERENCE_ROUTES` 的 `endpoint` / `api_key` / `model`，
业务代码无需改动。

**如何验证安装是否正常？**
执行 `python .tools/smoke_test.py`，它会用 Flask 测试客户端启动应用，逐项校验所有
路由、页面与响应结构。详见 `.tools/README.md`。

---

<div align="center">
<sub>NBA 比赛智能预测平台 · v2.0.0 ·
<a href="README.md">English</a></sub>
</div>
