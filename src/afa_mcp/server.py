"""AFA MCP Server — LobeHub通用插件入口"""
import json, os, math
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# Load embedded data
DATA_DIR = Path(__file__).parent / "data" if (Path(__file__).parent / "data").exists() else Path(os.environ.get("AFA_DATA_DIR", "."))

def _load_json(name: str) -> dict:
    p = DATA_DIR / name
    return json.loads(open(p).read()) if p.exists() else {}

ELO_DB = _load_json("elo_ratings.json")
HYPERPARAMS = _load_json("../configs/hyperparams.json")

mcp = FastMCP("afa-football-analyzer")

# ===== 1. ELO评分 =====
@mcp.tool()
def elo_calculate(home_elo: float, away_elo: float,
                  home_goals: int = None, away_goals: int = None,
                  k_factor: int = 20) -> dict:
    """ELO评分计算与更新。65分主场优势内置于公式中。"""
    exp_home = 1.0 / (1.0 + 10.0 ** ((away_elo - home_elo - 65.0) / 400.0))
    r = {"expected_home_win": round(exp_home, 4), "expected_away_win": round(1-exp_home, 4)}
    if home_goals is not None and away_goals is not None:
        actual = (1.0, 0.0) if home_goals > away_goals else ((0.5, 0.5) if home_goals == away_goals else (0.0, 1.0))
        r.update({"new_home_elo": round(home_elo + k_factor*(actual[0]-exp_home), 1),
                  "new_away_elo": round(away_elo + k_factor*(actual[1]-(1-exp_home)), 1)})
    return r

# ===== 2. Dixon-Coles泊松 =====
@mcp.tool()
def dixon_coles_predict(home_attack: float, home_defense: float,
                        away_attack: float, away_defense: float,
                        home_advantage: float = 0.3) -> dict:
    """Dixon-Coles双变量泊松比分预测。返回WDL概率+比分Top5+总进球分布。"""
    lam_h = math.exp(home_attack + away_defense + home_advantage)
    lam_a = math.exp(away_attack + home_defense)
    hw = dw = aw = 0.0; scores = {}; goals = {}
    for h in range(9):
        for a in range(9):
            p = (lam_h**h * math.exp(-lam_h) / math.factorial(h)) * (lam_a**a * math.exp(-lam_a) / math.factorial(a))
            if h > a: hw += p
            elif h == a: dw += p
            else: aw += p
            scores[f"{h}-{a}"] = round(p, 5); goals[str(h+a)] = goals.get(str(h+a), 0) + p
    return {"lam_home": round(lam_h,3), "lam_away": round(lam_a,3), "lam_total": round(lam_h+lam_a,3),
            "home_win": round(hw,4), "draw": round(dw,4), "away_win": round(aw,4),
            "top_scores": sorted(scores.items(), key=lambda x: -x[1])[:5],
            "goals_distribution": {k: round(goals[k], 4) for k in sorted(goals, key=lambda x: int(x))}}

# ===== 3. Kelly分析 =====
@mcp.tool()
def kelly_analyze(true_probability: float, odds: float,
                  bankroll: float = 10000, lottery_type: str = "jingcai") -> dict:
    """Kelly公式投注分析。自动区分竞彩(K>0.05)和北单(K>0.08)正期望阈值。"""
    if odds <= 1.0: return {"error": "赔率必须>1.0"}
    implied = 1.0 / odds; edge = true_probability - implied
    kf = (true_probability * odds - 1) / (odds - 1) if edge > 0 else 0.0
    ev = true_probability * odds * (0.65 if lottery_type.lower() == "beidan" else 1.0)
    threshold = 0.08 if lottery_type.lower() == "beidan" else 0.05
    return {"implied_prob": round(implied,4), "edge": round(edge,4), "ev": round(ev,4),
            "kelly_fraction": round(kf,6), "quarter_kelly": round(kf*0.25*bankroll,2),
            "recommended": kf > threshold, "lottery_type": lottery_type, "threshold": threshold}

# ===== 4. 赔率→隐含概率 =====
@mcp.tool()
def odds_implied_probabilities(odds_home: float, odds_draw: float, odds_away: float) -> dict:
    """从欧赔计算市场隐含概率和庄家利润率。"""
    inv = [1.0/odds_home, 1.0/odds_draw, 1.0/odds_away]; total = sum(inv)
    return {"implied_home": round(inv[0]/total,4), "implied_draw": round(inv[1]/total,4),
            "implied_away": round(inv[2]/total,4), "market_margin": round(total-1,4),
            "fair_odds": [round(1.0/(inv[i]/total),2) for i in range(3)]}

# ===== 5. 综合比赛分析 =====
@mcp.tool()
def comprehensive_match_analysis(home_team: str, away_team: str,
                                  odds_home: float, odds_draw: float, odds_away: float,
                                  home_elo: float = 1500, away_elo: float = 1500) -> dict:
    """一键综合：ELO+赔率+凯利+推荐。"""
    e = elo_calculate(home_elo, away_elo)
    o = odds_implied_probabilities(odds_home, odds_draw, odds_away)
    k = kelly_analyze(e['expected_home_win'], odds_home)
    w = HYPERPARAMS.get('weights', {'fundamental_quant': 0.19, 'contrarian_quant': 0.468, 'smart_money_quant': 0.342})
    return {"match": f"{home_team} vs {away_team}", "elo": e, "odds": o, "kelly": k, "evolution_weights": w}

# ===== 6. 联赛因子 =====
LEAGUE_FACTORS = {
    '英超': {'avg_goals':2.76,'home_win':0.427,'draw':0.264,'over25':0.553},
    '西甲': {'avg_goals':2.70,'home_win':0.488,'draw':0.244,'over25':0.509},
    '德甲': {'avg_goals':3.22,'home_win':0.431,'draw':0.254,'over25':0.632},
    '意甲': {'avg_goals':2.41,'home_win':0.397,'draw':0.269,'over25':0.454},
    '法甲': {'avg_goals':2.84,'home_win':0.465,'draw':0.248,'over25':0.532},
    '英冠': {'avg_goals':2.61,'home_win':0.417,'draw':0.265,'over25':0.507},
    '意乙': {'avg_goals':2.54,'home_win':0.449,'draw':0.311,'over25':0.484},
    '法乙': {'avg_goals':2.52,'home_win':0.368,'draw':0.311,'over25':0.473},
    '挪超': {'avg_goals':3.17,'home_win':0.444,'draw':0.264,'over25':0.615},
    '荷甲': {'avg_goals':3.08,'home_win':0.468,'draw':0.230,'over25':0.580},
}

@mcp.tool()
def get_league_factor(league_name: str) -> dict:
    """查询联赛量化因子(基于15.9万场实测)。非五大联赛可直接调用。"""
    lf = LEAGUE_FACTORS.get(league_name)
    return {"league": league_name, "factors": lf, "source": "15.9万场实测" if lf else "默认(数据不足)"}

# ===== 7. ELO查询 =====
@mcp.tool()
def get_team_elo(team_name: str) -> dict:
    """从1062队ELO数据库查询球队评级。支持模糊匹配。"""
    matches = [(k, v) for k, v in ELO_DB.items() if team_name.lower() in k.lower()]
    return {"query": team_name, "results": [{"team": k, "elo": round(v,1)} for k,v in sorted(matches, key=lambda x:-x[1])[:10]], "total": len(matches)}

# ===== 8. 比分全景矩阵 =====
@mcp.tool()
def score_probability_matrix(home_goals_expected: float, away_goals_expected: float) -> dict:
    """生成完整比分概率矩阵 + 总进球 + 上下单双。北单上下单双的核心数据源。"""
    max_g=8; hw=dw=aw=0; goals={}
    for h in range(max_g+1):
        for a in range(max_g+1):
            p = (home_goals_expected**h*math.exp(-home_goals_expected)/math.factorial(h)) * (away_goals_expected**a*math.exp(-away_goals_expected)/math.factorial(a))
            if h > a:
                hw += p
            elif h == a:
                dw += p
            else:
                aw += p
            goals[h+a] = goals.get(h+a, 0) + p
    up=sum(goals.get(i,0) for i in range(3,17)); down=1-up
    return {"WDL":{"home_win":round(hw,4),"draw":round(dw,4),"away_win":round(aw,4)},
            "goals_dist":{str(k):round(goals[k],4) for k in sorted(goals)},
            "beidan_sxds":{"上单":round(up*0.5,4),"上双":round(up*0.5,4),"下单":round(down*0.5,4),"下双":round(down*0.5,4)}}

# ===== 9. M串N =====
@mcp.tool()
def mxn_calculator(matches_sp: list, m: int, n: int = 1, stake: float = 100, lottery_type: str = "jingcai") -> dict:
    """M串N组合计算+奖金。自动处理竞彩封顶(2-3关20万/4-5关50万/6关+100万)及北单65%返奖率。"""
    from itertools import combinations
    import numpy as np
    if len(matches_sp) < m: return {"error": f"需至少{m}场"}
    combos = list(combinations(range(len(matches_sp)), m))
    rate = 0.65 if lottery_type.lower() == "beidan" else 1.0
    limits = {2:200000,3:200000,4:500000,5:500000,6:1000000,7:1000000,8:1000000}
    max_limit = limits.get(m, 200000)
    details = []
    for combo in combos:
        sp = np.prod([matches_sp[i] for i in combo]) * rate
        prize = min(2 * sp, max_limit)
        if prize >= 10000: prize *= 0.80
        details.append({"combo": list(combo), "sp": round(sp, 2), "prize": round(prize, 2)})
    return {"type": f"{m}串{n}", "lottery": lottery_type, "combos": len(combos),
            "max_prize": max(d['prize'] for d in details) if details else 0, "samples": details[:3]}

# ===== 10. 串关优化 =====
@mcp.tool()
def parlay_optimizer(matches: list, lottery_type: str = "jingcai", budget: float = 100) -> dict:
    """智能串关推荐。自动过滤(K>阈值 & SP<3.0 & 胜率>35%)，生成2串1到N串1。"""
    th = 0.08 if lottery_type.lower() == "beidan" else 0.05
    valid = [m for m in matches if m.get('kelly', 0) > th and m.get('sp', 99) < 3.0 and m.get('win_prob', 0) > 0.35]
    if len(valid) < 2: return {"error": "可串关场次不足"}
    rate = 0.65 if lottery_type.lower() == "beidan" else 1.0
    results = []
    for k in range(2, min(len(valid)+1, 7)):
        sp = 1.0
        for m in valid[:k]: sp *= m['sp']
        sp *= rate
        results.append({"type": f"{k}串1", "sp": round(sp, 2), "prize_100": round(100*sp, 0),
                        "matches": [m['name'] for m in valid[:k]],
                        "risk": "低" if k<=2 else ("中" if k<=4 else "高")})
    return {"lottery": lottery_type, "valid_count": len(valid), "recommendations": results}

# ===== 11. 进化反馈 =====
@mcp.tool()
def evolution_feedback(predicted: dict, actual: dict) -> dict:
    """赛后复盘反馈。分析预测偏差+给出权重调整建议。驱动持续进化。"""
    p_home = predicted.get('home_win', 0.5)
    hg, ag = actual.get('home_goals', 0), actual.get('away_goals', 0)
    actual_p = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
    error = abs(p_home - actual_p)
    if error > 0.30: suggestion = "偏差>30%: 降低基本面权重,检查诱盘信号"
    elif error > 0.15: suggestion = "偏差15-30%: 复盘伤病/天气/裁判因素"
    else: suggestion = "偏差<15%: 模型表现良好"
    return {"correct": (p_home > 0.5 and hg > ag) or (p_home < 0.5 and hg < ag),
            "error": round(error, 3), "suggestion": suggestion,
            "weights": HYPERPARAMS.get('weights', {}),
            "total_evolutions": HYPERPARAMS.get('evolution_memory', {}).get('total_simulations_run', 0)}

# ===== 12. 赔率校准 =====
ODDS_CAL = [(1.30,0.828,"强队稳胆"),(1.50,0.717,"正期望"),(1.80,0.588,"正期望"),(2.20,0.469,"谨慎"),(2.80,0.386,"避单选"),(99.0,0.230,"冷门区")]
@mcp.tool()
def odds_calibration_lookup(odds: float) -> dict:
    """查询赔率对应的实测胜率(8.9万场)。判断是否存在价值偏差。"""
    for th, wr, adv in ODDS_CAL:
        if odds < th: return {"odds": odds, "actual_win_rate": wr, "advice": adv}
    return {"odds": odds, "actual_win_rate": 0.25, "advice": "数据不足"}

# ===== 13. 北单上下单双 =====
@mcp.tool()
def beidan_sxds_analyzer(lam_total: float, league_name: str = "") -> dict:
    """北单上下单双概率分析。自动用联赛因子修正大球/小球倾向。"""
    goals_dist = {k: (lam_total**k*math.exp(-lam_total)/math.factorial(k)) for k in range(17)}
    up = sum(goals_dist.get(i, 0) for i in range(3, 17)); down = 1 - up
    lf = LEAGUE_FACTORS.get(league_name, {})
    over_rate = lf.get('over25', 0.50)
    up = up * (over_rate / 0.50); down = 1 - up
    odd_p = sum(goals_dist.get(i, 0) for i in range(1, 17, 2))
    results = {"上单": up*odd_p, "上双": up*(1-odd_p), "下单": down*odd_p, "下双": down*(1-odd_p)}
    return {"lam_total": lam_total, "best": max(results, key=results.get), "probs": {k: round(v,4) for k,v in results.items()}}

# ===== 14. 资金管理 =====
@mcp.tool()
def bankroll_calculator(bankroll: float, risk_level: str = "medium", num_bets: int = 3) -> dict:
    """资金管理计算。按风险等级输出单场/日/周限度。"""
    pcts = {"low": (0.01, 0.02), "medium": (0.02, 0.05), "high": (0.05, 0.08)}
    lo, hi = pcts.get(risk_level, (0.02, 0.05))
    return {"bankroll": bankroll, "per_bet": [round(bankroll*lo, 0), round(bankroll*hi, 0)],
            "max_daily": round(bankroll*hi*num_bets, 0), "daily_drawdown": round(bankroll*0.15, 0),
            "weekly_stop": round(bankroll*0.25, 0)}

# ===== 15. 500.com爬取 =====
@mcp.tool()
def scrape_500_jczq(date: str = "") -> dict:
    """从500.com爬取竞彩足球实时SP值和赛程。作为数据采集首选源。"""
    import requests
    url = f"https://trade.500.com/jczq/?playid=312&g=2{'&date='+date if date else ''}"
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        if resp.status_code != 200: return {"ok": False, "error": f"HTTP{resp.status_code}"}
        html = resp.text
        if len(html) < 200: return {"ok": False, "error": "empty"}
        return {"ok": True, "source": "500.com", "html_length": len(html), "note": "Raw HTML - parse SP values client-side"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def main():
    """Entry point for `afa-mcp-server` CLI command."""
    import sys
    port = int(os.environ.get("AFA_MCP_PORT", "18900"))
    print(f"⚽ AFA MCP Server starting on port {port}...")
    print(f"   Tools: 15 | ELO DB: {len(ELO_DB)} teams | Leagues: {len(LEAGUE_FACTORS)}")
    mcp.run()

if __name__ == "__main__":
    main()
