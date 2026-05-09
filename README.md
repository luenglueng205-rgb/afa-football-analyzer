# AFA MCP Server — 竞彩足球+北京单场量化分析引擎

LobeHub 通用 MCP 工具。提供 15 个足球量化分析端点。

## 安装
```bash
pip install .
```

## 使用
```bash
afa-mcp-server --port 18900
```

## 工具列表
| 工具 | 用途 |
|------|------|
| elo_calculate | ELO评分 |
| dixon_coles_predict | Dixon-Coles泊松 |
| kelly_analyze | 凯利公式 |
| odds_implied_probabilities | 赔率隐含概率 |
| comprehensive_match_analysis | 综合比赛分析 |
| get_league_factor | 联赛因子 |
| get_team_elo | ELO查询 |
| score_probability_matrix | 比分全景矩阵 |
| mxn_calculator | M串N计算 |
| parlay_optimizer | 串关优化 |
| evolution_feedback | 进化反馈 |
| odds_calibration_lookup | 赔率校准 |
| beidan_sxds_analyzer | 上下单双 |
| bankroll_calculator | 资金管理 |
| scrape_500_jczq | 500.com爬取 |
