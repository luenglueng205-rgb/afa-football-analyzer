# AFA MCP Server — 竞彩足球+北京单场量化分析引擎

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![MCP](https://img.shields.io/badge/MCP-v1.27-green)](https://modelcontextprotocol.io/)
[![Version](https://img.shields.io/badge/version-2.0.0-orange)](https://github.com/luenglueng205-rgb/afa-football-analyzer)

LobeHub 通用 MCP 工具。提供 **19 个足球量化分析端点**，覆盖竞彩足球+北京单场 12 种玩法。

---

## 安装

```bash
# 从 GitHub 安装
pip install git+https://github.com/luenglueng205-rgb/afa-football-analyzer.git

# 或本地开发模式
git clone https://github.com/luenglueng205-rgb/afa-football-analyzer.git
cd afa-football-analyzer
pip install -e .
```

---

## LobeHub 配置

在 LobeChat → 设置 → MCP 服务器 → 添加自定义 MCP：

```json
{
  "mcpServers": {
    "afa-football-analyzer": {
      "command": "python3",
      "args": ["-m", "afa_mcp.server"],
      "env": {
        "PYTHONPATH": "/Users/你的用户名/Desktop/afa-mcp-package/src",
        "AFA_DATA_DIR": "/Users/你的用户名/Desktop/afa-mcp-package/data"
      }
    }
  }
}
```

> ⚠️ 替换 `PYTHONPATH` 和 `AFA_DATA_DIR` 为你的实际路径

---

## 19 个工具

### 🧮 计算引擎（5个）
| 工具 | 用途 |
|------|------|
| `elo_calculate` | ELO评分计算与赛后更新 |
| `dixon_coles_predict` | Dixon-Coles双变量泊松预测 |
| `kelly_analyze` | Kelly公式 · 竞彩/北单阈值自动区分 |
| `odds_implied_probabilities` | 赔率→市场隐含概率 |
| `comprehensive_match_analysis` | 一键综合：ELO+赔率+Kelly |

### 📊 概率建模（3个）
| 工具 | 用途 |
|------|------|
| `score_probability_matrix` | 完整比分概率矩阵+总进球+上下单双 |
| `monte_carlo_simulator` | 蒙特卡洛模拟(5000次迭代) |
| `beidan_sxds_analyzer` | 北单上下单双+联赛因子修正 |

### 🎲 投注优化（4个）
| 工具 | 用途 |
|------|------|
| `mxn_calculator` | M串N组合+奖金(含封顶扣税) |
| `parlay_optimizer` | 智能串关推荐(2串1到N串1) |
| `smart_bet_selector` | 价值投注筛选+排序 |
| `bankroll_calculator` | 资金管理(单场/日/周限度) |

### 📡 数据采集（2个）
| 工具 | 用途 |
|------|------|
| `scrape_500_jczq` | 500.com竞彩实时SP爬取 |
| `scrape_500_beidan` | 500.com北单实时SP爬取 |

### 🧬 自我进化（3个）
| 工具 | 用途 |
|------|------|
| `evolution_feedback` | 赛后复盘+偏差分析+权重调整建议 |
| `evolution_status` | 进化引擎状态(129次回测参数) |
| `odds_calibration_lookup` | 赔率实测胜率查询(8.9万场) |

### 🏷️ 辅助（2个）
| 工具 | 用途 |
|------|------|
| `get_league_factor` | 联赛量化因子(15.9万场实测) |
| `get_team_elo` | 1062队ELO查询 |

---

## 数据源

- **500.com** — 竞彩+北单实时SP值
- **15.9万场历史数据** — 赔率校准+联赛因子
- **1062队ELO** — 球队实力评级
- **129次进化回测** — 权重参数优化

---

## 需求

- Python 3.10+
- `mcp>=1.0` · `numpy` · `requests`
