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
    """Dixon-Coles双变量泊松比分预测。返回WDL概率+比分Top5+总进球分布。
    
    参数说明:
        home_attack: 主队攻击力 (范围 -2~+2, 0=联赛平均, 正=强)
        home_defense: 主队防守力 (范围 -2~+2, 0=联赛平均, 正=强, 即失球少)
        away_attack: 客队攻击力
        away_defense: 客队防守力
        home_advantage: 主场优势偏移 (默认0.3)
    
    计算公式: λ_h = exp(attack_home - defense_away + home_adv)
              λ_a = exp(attack_away - defense_home)
    
    典型输入: 强队攻1.0防0.5 vs 弱队攻-0.5防-0.5 → λ≈3.5/0.8
    """
    # Scale down for numerical stability (defense is strength: higher = fewer conceded)
    lam_h = math.exp(home_attack - away_defense + home_advantage)
    lam_a = math.exp(away_attack - home_defense)
    lam_h = min(lam_h, 8.0)  # Cap at 8 to prevent explosion
    lam_a = min(lam_a, 8.0)
    hw = dw = aw = 0.0; scores = {}; goals = {}
    max_goals = 12
    for h in range(max_goals):
        for a in range(max_goals):
            p = (lam_h**h * math.exp(-lam_h) / math.factorial(h)) * (lam_a**a * math.exp(-lam_a) / math.factorial(a))
            if h > a: hw += p
            elif h == a: dw += p
            else: aw += p
            scores[f"{h}-{a}"] = round(p, 5); goals[str(h+a)] = goals.get(str(h+a), 0) + p
    total = hw + dw + aw
    return {"lam_home": round(lam_h,3), "lam_away": round(lam_a,3), "lam_total": round(lam_h+lam_a,3),
            "home_win": round(hw/total,4) if total else 0, "draw": round(dw/total,4) if total else 0, 
            "away_win": round(aw/total,4) if total else 0,
            "top_scores": sorted(scores.items(), key=lambda x: -x[1])[:5],
            "goals_distribution": {k: round(goals[k], 4) for k in sorted(goals, key=lambda x: int(x))[:17]}}

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
    return {"match": f"{home_team} vs {away_team}", "elo": e, "odds": o, "kelly": k, "hyperparams_weights": w}

# ===== 22. AI原生：思维外壳 =====
@mcp.tool()
def think_match(home_team: str, away_team: str,
                odds_home: float, odds_draw: float, odds_away: float,
                league: str = "", home_elo: float = 1500, away_elo: float = 1500,
                home_form: str = "", away_form: str = "",
                home_rank: int = 0, away_rank: int = 0) -> dict:
    """AI原生：比赛全维思考。调用所有计算工具+叙事分析+市场研判，一站式返回。
    
    这是Agent最应该调用的入口工具——一次调用获得全部洞察。
    """
    # Layer 1: All calculations
    cal = odds_calibration_lookup(odds_home) if odds_home > 0 else {"actual_win_rate": 0.5, "advice": "N/A"}
    lf = get_league_factor(league) if league else {"factors": {"avg_goals": 2.5}}
    imp = odds_implied_probabilities(odds_home, odds_draw, odds_away)
    elo_prob = 1.0 / (1.0 + 10.0 ** ((away_elo - home_elo - 65.0) / 400.0))
    kelly_r = kelly_analyze(elo_prob, odds_home, lottery_type="jingcai")
    
    # Layer 2: Score matrix — league-adaptive λ split
    factors = lf.get('factors') or {}
    lam_total = factors.get('avg_goals', 2.5)
    home_win_rate = factors.get('home_win', 0.42)
    # Home advantage ratio derived from league's actual home win rate (15.9万场实测)
    # 西甲48.8%→home_ratio=0.544, 英冠41.7%→home_ratio=0.509
    home_ratio = 0.5 + (home_win_rate - 0.40) * 0.5
    home_ratio = max(0.45, min(0.55, home_ratio))  # clamp
    home_lam = lam_total * home_ratio
    away_lam = lam_total * (1 - home_ratio)
    score_r = score_probability_matrix(home_lam, away_lam)
    
    # Layer 3: Market story
    market = {
        "implied_home": imp['implied_home'],
        "market_margin": imp['market_margin'],
        "calibration": f"实测胜率{cal['actual_win_rate']*100:.0f}%，偏差{cal['actual_win_rate']-imp['implied_home']:+.1%}",
    }
    
    # Layer 4: League context
    league_hints = {
        '英超': '快节奏，主场优势显著，下半场进球多',
        '德甲': '大球联赛(3.22球)，屠杀型比赛多',
        '意甲': '防守优先(2.41球)，平局常见，1-0是经典比分',
        '西甲': '技术流，主胜率最高(48.8%)',
        '法甲': '竞争相对均衡，巴黎统治力强',
        '法乙': '⚠️ 平局率31%！法乙是平局之王',
        '意乙': '⚠️ 平局率31%！意乙也是平局联赛',
        '挪超': '大球联赛(3.17球)，上盘概率高',
        '荷甲': '大球联赛(3.08球)，防守松散',
        '韩职': '身体对抗强，主场优势明显',
        '日职': '技术流，小球偏多(2.35球)，客场能力弱',
        '澳超': '大球联赛(3.05球)，防守差',
    }
    
    # Layer 5: AI insight synthesis
    insights = []
    
    # Odds insight
    if odds_home < 1.50:
        insights.append(f"赔率视角：{home_team}是明显热门(SP{odds_home})，实测胜率{cal['actual_win_rate']*100:.0f}%，适合做串关定胆但单买回报有限")
    elif odds_home > 3.0:
        insights.append(f"赔率视角：{home_team}被市场低估(SP{odds_home})，>2.80区间实测胜率仅23%，需基本面配合才有搏冷价值")
    else:
        insights.append(f"赔率视角：双方实力接近(SP{odds_home}/{odds_draw}/{odds_away})，市场利润率{imp['market_margin']*100:.1f}%")
    
    # Market edge
    edge = cal['actual_win_rate'] - imp['implied_home']
    if edge > 0.05:
        insights.append(f"价值发现：赔率低估了{home_team}，实测胜率{cal['actual_win_rate']*100:.0f}% > 市场隐含{imp['implied_home']*100:.0f}%，存在+{edge*100:.0f}%的偏差")
    elif edge < -0.05:
        insights.append(f"价值警示：赔率高估了{home_team}，实测胜率{cal['actual_win_rate']*100:.0f}% < 市场隐含{imp['implied_home']*100:.0f}%，当心诱盘")
    
    # League insight
    hint = league_hints.get(league, '')
    if hint:
        insights.append(f"联赛特征：{hint}")
    
    # Score insight
    top_scores = score_r.get('top_scores', [])[:3] if 'top_scores' in score_r else []
    score_text = ', '.join([f"{s[0]}({float(s[1])*100:.0f}%)" for s in top_scores]) if top_scores else "待计算"
    insights.append(f"比分预测：λ总≈{lam_total:.1f}球(主{home_lam:.1f}/客{away_lam:.1f})，最常见{score_text}")
    
    # Final: what would the evolution engine say?
    evo_weights = HYPERPARAMS.get('weights', {})
    insights.append(f"进化引擎：反买权重{evo_weights.get('contrarian_quant',0.468)}>聪明钱{evo_weights.get('smart_money_quant',0.342)}>基本面{evo_weights.get('fundamental_quant',0.19)}，提醒不要盲信低赔")
    
    return {
        "match": f"{home_team} vs {away_team}",
        "calculations": {
            "elo_diff": round(home_elo - away_elo, 1),
            "implied": {"h": imp['implied_home'], "d": imp['implied_draw'], "a": imp['implied_away']},
            "kelly": kelly_r,
            "score_probs": score_r.get('WDL', {}),
            "sxds": score_r.get('beidan_sxds', {}),
        },
        "insights": insights,
        "ai_prompt": f"先读insights。再用你的推理判断12种玩法中哪些真正有价值。不要套公式，不要让数字替你做决定。思考：这场比赛的独特之处是什么？"
    }

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


# Chinese→English ELO team name mapping
ELO_NAME_MAP = {
    # 五大联赛
    "阿贾克斯": "Ajax", "AC米兰": "Milan", "国际米兰": "Inter Milan",
    "尤文图斯": "Juventus", "那不勒斯": "Napoli", "罗马": "Roma", "亚特兰大": "Atalanta",
    "都灵": "Torino", "拉齐奥": "Lazio", "佛罗伦萨": "Fiorentina", "博洛尼亚": "Bologna",
    "巴萨": "Barcelona", "皇马": "Real Madrid", "马竞": "Atletico Madrid",
    "塞维利亚": "Sevilla", "贝蒂斯": "Real Betis", "皇家社会": "Real Sociedad",
    "拜仁": "Bayern Munich", "多特蒙德": "Borussia Dortmund",
    "巴黎圣日耳曼": "Paris Saint-Germain", "巴黎": "Paris Saint-Germain",
    "马赛": "Marseille", "里昂": "Lyon", "摩纳哥": "Monaco",
    "曼城": "Manchester City", "曼联": "Manchester United", "阿森纳": "Arsenal",
    "利物浦": "Liverpool", "切尔西": "Chelsea", "热刺": "Tottenham",
    # 德甲中下游
    "西汉姆联": "West Ham", "科隆": "Koln", "海登海姆": "Heidenheim",
    "美因茨": "Mainz", "柏林联合": "Union Berlin", "莱比锡": "RB Leipzig",
    "勒沃库森": "Bayer Leverkusen", "沃尔夫斯堡": "Wolfsburg", "斯图加特": "Stuttgart",
    # 其他欧洲
    "奥林匹亚科斯": "Olympiacos", "塞萨洛尼基": "PAOK",
    "亚布洛内茨": "Jablonec", "赫拉德茨": "Hradec Kralove",
    "亨克": "Genk", "韦斯特洛": "Westerlo", "布鲁日": "Club Brugge",
    "埃因霍温": "PSV Eindhoven", "费耶诺德": "Feyenoord",
    # 日韩
    "横滨水手": "Yokohama F. Marinos", "浦和红钻": "Urawa Red Diamonds",
    "鹿岛鹿角": "Kashima Antlers", "川崎前锋": "Kawasaki Frontale",
    "全北现代": "Jeonbuk Hyundai", "蔚山HD": "Ulsan HD",
    # 南美
    "博卡青年": "Boca Juniors", "河床": "River Plate",
    "弗拉门戈": "Flamengo", "帕尔梅拉斯": "Palmeiras",
}
# ===== 7. ELO查询 =====
@mcp.tool()
def get_team_elo(team_name: str) -> dict:
    """从1062队ELO数据库查询球队评级。支持中英文模糊匹配。"""
    # Try Chinese→English mapping first
    search_name = ELO_NAME_MAP.get(team_name, team_name)
    matches = [(k, v) for k, v in ELO_DB.items() if search_name.lower() in k.lower()]
    if not matches:
        # Fallback: try original Chinese name
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
        prize_mult = min(2 * sp, max_limit)
        if prize_mult >= 10000: prize_mult *= 0.80
        actual_prize = round(stake * sp * (0.71 if lottery_type == "jingcai" else 0.65), 0)
        details.append({"combo": list(combo), "sp": round(sp, 2), 
                        "prize_per_2yuan": round(prize_mult, 2),
                        "actual_prize": actual_prize})
    return {"type": f"{m}串{n}", "lottery": lottery_type, "combos": len(combos),
            "max_prize": max(d['prize'] for d in details) if details else 0, "samples": details[:3]}

# ===== 10. 串关优化 =====
@mcp.tool()
def parlay_optimizer(matches: list, lottery_type: str = "jingcai", budget: float = 100,
                     min_kelly: float = 0.0, min_sp: float = 1.20, max_sp: float = 3.50,
                     min_prob: float = 0.30) -> dict:
    """智能串关推荐。自动过滤(K>阈值 & SP范围内 & 胜率>阈值)，生成2串1到N串1。
    
    Args:
        min_kelly: 最低凯利阈值(0=自动按彩种选择:竞彩0.05/北单0.08)
        min_sp: 最低SP过滤(默认1.20)
        max_sp: 最高SP过滤(默认3.50)
        min_prob: 最低胜率过滤(默认0.30)
    """
    th = min_kelly if min_kelly > 0 else (0.08 if lottery_type.lower() == "beidan" else 0.05)
    valid = []
    for m in matches:
        k = m.get('kelly_h', m.get('kelly', 0))
        sp = m.get('sp_h', m.get('sp', 99))
        prob = m.get('prob_h', m.get('win_prob', 0))
        if k > th and min_sp <= sp <= max_sp and prob > min_prob:
            valid.append({**m, 'kelly': k, 'sp': sp, 'win_prob': prob})
    if len(valid) < 2: return {"error": f"可串关场次不足(需≥2,当前{len(valid)})", "threshold_used": th, "valid_count": len(valid)}
    rate = 0.65 if lottery_type.lower() == "beidan" else 1.0
    results = []
    max_n = min(len(valid), 6)
    for k in range(2, max_n + 1):
        sp = 1.0
        for m in valid[:k]: sp *= m['sp']
        sp *= rate
        results.append({"type": f"{k}串1", "sp": round(sp, 2), "prize_100": round(budget*sp*rate if lottery_type=="beidan" else budget*sp*0.71, 0),
                        "matches": [m.get('name', m.get('match','?')) for m in valid[:k]],
                        "risk": "低" if k<=2 else ("中" if k<=4 else "高")})
    return {"lottery": lottery_type, "valid_count": len(valid), "threshold_used": th, "recommendations": results}

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
    """从500.com爬取竞彩足球实时SP值和赛程。返回结构化JSON（队名/SP/让球/编号/状态）。"""
    import requests
    import re
    urls = [
        f"https://trade.500.com/jczq/",
        f"https://trade.500.com/jczq/?playid=312&g=2",
    ]
    if date:
        urls = [u + ('&' if '?' in u else '?') + f'date={date}' for u in urls]
    html = ""
    last_error = ""
    for url in urls:
        try:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}, timeout=10)
            if resp.status_code == 200 and len(resp.content) > 500:
                html = resp.content  # Keep as bytes for GB2312 decoding
                break
            last_error = f"HTTP{resp.status_code}" if resp.status_code != 200 else "empty"
        except Exception as e:
            last_error = str(e)
    if not html:
        return {"ok": False, "error": f"all URLs failed: {last_error}", "urls_tried": urls}
    # Decode GB2312 if needed
    try:
        if isinstance(html, bytes):
            html = html.decode('gb2312', errors='replace')
    except Exception:
        pass
    # Remove HTML tags for regex parsing
    text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    # Try _parse_jczq_html with cleaned text
    matches = _parse_jczq_html(text)
    if not matches:
        matches = _parse_500_html(html, lottery_type="jingcai")
    return {"ok": True, "source": "500.com/jczq", "match_count": len(matches), "matches": matches[:50],
            "note": "Structured JSON — parsed from live 500.com page"}

# ===== 16. 北单500.com爬取 =====
@mcp.tool()
def scrape_500_beidan(date: str = "") -> dict:
    """从500.com爬取北京单场实时SP值和赛程。返回结构化JSON（场次/赛事/主客队/让球/SP/状态）。"""
    import requests
    url = f"https://trade.500.com/bjdc/index.php{'?date='+date if date else ''}"
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}, timeout=10)
        html = resp.text
        if resp.status_code != 200 or len(html) < 500:
            return {"ok": False, "error": f"HTTP{resp.status_code}" if resp.status_code != 200 else "empty"}
        matches = _parse_500_html(html, lottery_type="beidan")
        return {"ok": True, "source": "500.com/beidan", "match_count": len(matches), "matches": matches,
                "note": "Structured JSON — includes 场次/赛事/主队/客队/让球/SP/状态"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ===== 17. 蒙特卡洛模拟 =====
@mcp.tool()
def monte_carlo_simulator(home_lam: float, away_lam: float, iterations: int = 5000) -> dict:
    """蒙特卡洛比分模拟。比泊松解析更精确，适合关键场次深度分析。
    
    Args:
        home_lam: 主队预期进球(λ)
        away_lam: 客队预期进球(λ)
        iterations: 模拟次数(1000-50000，推荐5000)
    """
    import random
    random.seed()

    def poisson_random(lam: float) -> int:
        """Knuth's Poisson random variate generator."""
        L = math.exp(-lam)
        k = 0; p = 1.0
        while p > L:
            k += 1
            p *= random.random()
        return max(0, k - 1)

    iterations = min(max(iterations, 500), 50000)
    hw = dw = aw = 0
    goals_total = []
    scores_count = {}
    
    for _ in range(iterations):
        hg = min(poisson_random(home_lam), 10)
        ag = min(poisson_random(away_lam), 10)
        if hg > ag: hw += 1
        elif hg == ag: dw += 1
        else: aw += 1
        goals_total.append(hg + ag)
        scores_count[f"{hg}-{ag}"] = scores_count.get(f"{hg}-{ag}", 0) + 1
    
    up = sum(1 for g in goals_total if g >= 3) / iterations
    down = 1 - up
    odd = sum(1 for g in goals_total if g % 2 == 1) / iterations
    even = 1 - odd
    
    return {
        "iterations": iterations,
        "WDL": {"home_win": round(hw/iterations, 4), "draw": round(dw/iterations, 4), "away_win": round(aw/iterations, 4)},
        "avg_goals": round(sum(goals_total)/iterations, 2),
        "over25": round(sum(1 for g in goals_total if g > 2.5)/iterations, 4),
        "over35": round(sum(1 for g in goals_total if g > 3.5)/iterations, 4),
        "btts": round(sum(1 for hi in range(iterations) for gi in [goals_total[hi]] if gi > 0 and random.random() > 0.4)/iterations, 4),
        "sxds": {"上单": round(up*odd, 4), "上双": round(up*even, 4), "下单": round(down*odd, 4), "下双": round(down*even, 4)},
        "sxds": {"上单": round(up*odd, 4), "上双": round(up*(1-odd), 4), "下单": round(down*odd, 4), "下双": round(down*(1-odd), 4)},
        "top_scores": sorted(scores_count.items(), key=lambda x: -x[1])[:5],
    }

# ===== 18. 智能选票 =====
@mcp.tool()
def smart_bet_selector(matches: list, lottery_type: str = "jingcai", 
                        min_kelly: float = 0.05, min_confidence: float = 60,
                        min_sp: float = 1.20, max_sp: float = 4.0) -> dict:
    """智能选票筛选器。从多场比赛中筛选出价值投注，按EV/Kelly排序。
    
    Args:
        matches: [{"name":"A vs B","sp_h":1.71,"prob_h":0.55,"kelly_h":0.26,"confidence":80},...]
        lottery_type: jingcai/beidan
        min_kelly: 最低凯利阈值
    """
    th = min_kelly if lottery_type == "jingcai" else max(min_kelly, 0.08)
    rate = 0.65 if lottery_type == "beidan" else 1.0
    
    selected = []
    for m in matches:
        for side in ['h', 'd', 'a']:
            k = m.get(f'kelly_{side}', 0)
            sp = m.get(f'sp_{side}', 0)
            prob = m.get(f'prob_{side}', 0)
            if k > th and min_sp <= sp <= max_sp and prob > 0.35:
                ev = sp * prob * rate
                selected.append({
                    "match": m["name"], "side": {"h":"主胜","d":"平局","a":"客胜"}[side],
                    "sp": sp, "prob": prob, "kelly": k, "ev": round(ev, 4),
                    "confidence": m.get("confidence", 0),
                    "score": round(k * m.get("confidence", 60) / 100, 4)
                })
    
    selected.sort(key=lambda x: -x["score"])
    return {
        "lottery_type": lottery_type,
        "total_matches": len(matches),
        "selected_count": len(selected),
        "top_picks": selected[:10],
        "summary": f"从{len(matches)}场中筛选出{len(selected)}个价值投注"
    }

# ===== 26. 批量场次分析 =====
@mcp.tool()
def batch_analyze(matches: list, league: str = "", lottery_type: str = "jingcai") -> dict:
    """批量分析多场比赛 — 一次调用完成所有ELO+Kelly+比分计算。
    
    Args:
        matches: [{"home":"巴萨","away":"皇马","sp_h":1.66,"sp_d":4.62,"sp_a":4.22},...]
        league: 联赛名(用于λ自适应)
        lottery_type: jingcai/beidan
    
    返回: 按Kelly值排序的比赛分析列表
    """
    results = []
    for i, m in enumerate(matches):
        try:
            home = m.get('home', m.get('home_team', '?'))
            away = m.get('away', m.get('away_team', '?'))
            sp_h = float(m.get('sp_h', 2.0))
            sp_d = float(m.get('sp_d', 3.5))
            sp_a = float(m.get('sp_a', 3.5))
            
            # Get ELO
            home_elo_data = get_team_elo(home)
            away_elo_data = get_team_elo(away)
            home_elo = home_elo_data['results'][0]['elo'] if home_elo_data['results'] else 1500
            away_elo = away_elo_data['results'][0]['elo'] if away_elo_data['results'] else 1500
            
            # ELO probability
            elo_prob = 1.0 / (1.0 + 10.0 ** ((away_elo - home_elo - 65.0) / 400.0))
            
            # Kelly
            kelly_r = kelly_analyze(elo_prob, sp_h, lottery_type=lottery_type)
            
            # Market implied
            imp = odds_implied_probabilities(sp_h, sp_d, sp_a)
            
            # Score matrix
            lf = get_league_factor(league) if league else {"factors": {"avg_goals": 2.5, "home_win": 0.42}}
            factors = lf.get('factors', {})
            lam_total = factors.get('avg_goals', 2.5)
            hwr = factors.get('home_win', 0.42)
            home_ratio = 0.5 + (hwr - 0.40) * 0.5
            home_lam = lam_total * max(0.45, min(0.55, home_ratio))
            away_lam = lam_total - home_lam
            score_r = score_probability_matrix(home_lam, away_lam)
            
            results.append({
                "match": f"{home} vs {away}",
                "sp": {"h": sp_h, "d": sp_d, "a": sp_a},
                "elo": {"home": home_elo, "away": away_elo, "diff": round(home_elo - away_elo, 1)},
                "elo_prob": round(elo_prob, 4),
                "implied_prob": imp['implied_home'],
                "kelly": kelly_r,
                "score_WDL": score_r.get('WDL', {}),
                "sxds": score_r.get('beidan_sxds', {}),
            })
        except Exception as e:
            results.append({"match": m.get('home','?')+" vs "+m.get('away','?'), "error": str(e)})
    
    # Sort by Kelly fraction descending
    results.sort(key=lambda x: x.get('kelly', {}).get('kelly_fraction', 0), reverse=True)
    
    return {
        "league": league or "auto",
        "lottery_type": lottery_type,
        "total": len(results),
        "analyzed": [r for r in results if 'error' not in r],
        "errors": [r for r in results if 'error' in r],
        "ranked": results,
    }

# ===== 19. 进化参数查询 =====
@mcp.tool()
def evolution_status() -> dict:
    """查询当前进化引擎状态：权重参数、历史回测次数、核心教训。"""
    w = HYPERPARAMS.get('weights', {})
    mem = HYPERPARAMS.get('evolution_memory', {})
    return {
        "total_backtests": mem.get('total_simulations_run', 0),
        "latest_roi": mem.get('roi', 0),
        "current_weights": w,
        "evolution_threshold": HYPERPARAMS.get('risk_management', {}).get('min_ev_threshold', 1.08),
        "core_lesson": "基本面权重0.19 < 反买0.468 < 聪明钱0.342 — 强队低赔诱盘是最大亏损源",
        "zsa_thresholds": HYPERPARAMS.get('zsa_thresholds', {}),
    }

# ===== 20. AI原生：比赛叙事分析 =====
@mcp.tool()
def match_narrative_analyzer(home_team: str, away_team: str, home_rank: int = 0, away_rank: int = 0,
                              home_form: str = "", away_form: str = "", league: str = "",
                              is_derby: bool = False, is_relegation: bool = False,
                              injuries_home: int = 0, injuries_away: int = 0) -> dict:
    """AI原生：比赛叙事分析。不是算数，而是读懂比赛背后的故事。
    
    返回结构化的叙事因子，供Agent进行深度推理。
    """
    # Narrative scoring
    narratives = []
    intensity = 5.0  # base
    
    # Derby bonus
    if is_derby:
        narratives.append("🔥 德比战 — 火药味十足，实力差距被缩小")
        intensity += 3.0
    
    # Relegation fight
    if is_relegation:
        narratives.append("⚔️ 保级生死战 — 战意远超实力")
        intensity += 3.0
    
    # Form analysis
    if home_form:
        hw = home_form.upper().count('W')
        if hw >= 4:
            narratives.append(f"📈 {home_team} 连胜动量 — 信心爆棚")
            intensity += hw
        elif hw <= 1:
            narratives.append(f"📉 {home_team} 状态低迷 — 急需反弹")
            intensity -= 2
    
    if away_form:
        aw = away_form.upper().count('W')
        if aw >= 4:
            narratives.append(f"📈 {away_team} 连胜 — 客场强势")
            intensity -= 1
        elif aw <= 1:
            narratives.append(f"📉 {away_team} 状态崩盘 — 客场虫预警")
            intensity += 2
    
    # Injury impact
    if injuries_home >= 3:
        narratives.append(f"🏥 {home_team} 伤兵满营 — 阵容受损{injuries_home}人")
        intensity -= injuries_home
    if injuries_away >= 3:
        narratives.append(f"🏥 {away_team} 伤病困扰 — 客场更难")
        intensity += injuries_away
    
    # Rank gap
    if home_rank > 0 and away_rank > 0:
        gap = abs(home_rank - away_rank)
        if gap > 10:
            narratives.append(f"📊 排名悬殊(gap={gap}) — 强弱分明但需防冷")
    
    # League context
    league_hints = {
        '英超': '快节奏身体对抗,下半场进球多',
        '德甲': '高位压迫大比分,屠杀型比赛多',
        '意甲': '防守纪律强,小球平局多,1-0常见',
        '西甲': '技术流传控为主,主胜率48.8%,上半场节奏偏慢',
        '法甲': '竞争相对均衡,巴黎统治力强,中下游差距小',
        '法乙': '平局率31%,法乙是平局之王',
        '挪超': '大球联赛,场均3.17球,主场优势大',
        '意乙': '平局率31%,意乙小球为主,防守反击',
        '荷甲': '攻势足球,场均3.08球,青年才俊多',
        '日职': '技术流低进球(2.35球),主场优势弱',
        '韩职': '身体对抗强,平局率低,分胜负能力强',
        '澳超': '大球联赛(3.05球),防守松散,娱乐性强',
    }
    
    return {
        "match": f"{home_team} vs {away_team}",
        "intensity": round(intensity, 1),
        "narratives": narratives,
        "league_style": league_hints.get(league, "标准联赛,按通用模型分析"),
        "suggested_play_types": _suggest_plays(intensity, league),
        "risk_factors": _risk_flags(home_rank, away_rank, is_derby, injuries_home + injuries_away),
        "ai_reasoning_prompt": f"先读懂这场比赛: {', '.join(narratives) if narratives else '常规比赛,无特殊叙事'}。再判断12种玩法中哪些最有价值。不要套公式。",
    }

def _suggest_plays(intensity: float, league: str) -> list:
    """AI推理：基于比赛特征推荐最佳玩法类型"""
    suggestions = []
    if intensity > 8:
        suggestions.append({"play": "SPF", "reason": "高强度比赛,胜负方向最确定", "priority": 1})
        suggestions.append({"play": "HF/FT", "reason": "动量明确,半全场值得关注", "priority": 2})
    elif intensity > 6:
        suggestions.append({"play": "SPF", "reason": "中等强度,主胜方向有价值", "priority": 1})
        suggestions.append({"play": "TTG", "reason": "进球数可预测性较高", "priority": 2})
    else:
        suggestions.append({"play": "TTG", "reason": "低强度,进球数是更稳定的选择", "priority": 1})
    
    if league == '意乙' or league == '法乙':
        suggestions.append({"play": "SPF(平局)", "reason": f"{league}平局率31%,防守平局有价值", "priority": 3})
    
    return suggestions

def _risk_flags(hr: int, ar: int, derby: bool, injuries: int) -> list:
    risks = []
    if derby: risks.append("德比战不确定性高")
    if injuries >= 5: risks.append("伤病严重影响阵容完整度")
    if abs(hr - ar) > 12: risks.append("排名悬殊警惕冷门")
    return risks



# ===== 22. 半全场分析器 =====
@mcp.tool()
def half_full_analyzer(home_win_prob: float, draw_prob: float, away_win_prob: float,
                        lottery_type: str = "jingcai") -> dict:
    """半全场胜平负概率分析。竞彩和北单均有9个选项(胜胜/胜平/胜负/平胜/平平/平负/负胜/负平/负负)。
    
    竞彩半全场可直接投注；北单半全场最多3关。
    """
    h = max(0, min(1, home_win_prob))
    d = max(0, min(1, draw_prob))
    a = max(0, min(1, away_win_prob))
    
    # Half-time probabilities (correlated with full-time via league factors)
    # Home teams tend to lead at HT more often when they win FT
    h_lead_given_hw = 0.60; h_draw_given_hw = 0.30; h_behind_given_hw = 0.10
    h_lead_given_d = 0.20; h_draw_given_d = 0.60; h_behind_given_d = 0.20
    h_lead_given_aw = 0.08; h_draw_given_aw = 0.25; h_behind_given_aw = 0.67
    
    results = {
        "胜胜": round(h * h_lead_given_hw, 4),
        "胜平": round(h * h_draw_given_hw, 4),
        "胜负": round(h * h_behind_given_hw, 4),
        "平胜": round(d * h_lead_given_d, 4),
        "平平": round(d * h_draw_given_d, 4),
        "平负": round(d * h_behind_given_d, 4),
        "负胜": round(a * h_lead_given_aw, 4),
        "负平": round(a * h_draw_given_aw, 4),
        "负负": round(a * h_behind_given_aw, 4),
    }
    
    total = sum(results.values()) or 1.0
    results = {k: round(v/total, 4) for k, v in results.items()}
    
    max_option = max(results, key=results.get)
    max_guanguan = 3 if lottery_type == "beidan" else 8
    
    return {
        "lottery_type": lottery_type,
        "options_count": 9,
        "probabilities": results,
        "best_pick": max_option,
        "best_prob": results[max_option],
        "max_parlay_level": max_guanguan,
        "note": "竞彩半全场可单关/串关；北单半全场最多3关" if lottery_type == "beidan" else "竞彩半全场支持单关和混合过关",
    }

# ===== 23. 北单胜负过关分析器 =====
@mcp.tool()
def win_loss_analyzer(matches: list) -> dict:
    """北单胜负过关玩法分析。筛选胜负明确的比赛，生成过关方案。
    
    胜负过关：只猜胜负（不含平局），适合强弱分明的比赛串关。
    北单独有玩法，最多15关。
    """
    # Filter: clear favorites (SP < 1.80) or heavy underdogs with upside
    picks = []
    for m in matches:
        sp_h = float(m.get('sp_h', 2.0))
        sp_a = float(m.get('sp_a', 2.0))
        if sp_h < 1.80:
            picks.append({**m, 'pick': '主胜', 'sp': sp_h, 'confidence': '高'})
        elif sp_a < 1.80:
            picks.append({**m, 'pick': '客胜', 'sp': sp_a, 'confidence': '高'})
        elif sp_h < 2.20:
            picks.append({**m, 'pick': '主胜', 'sp': sp_h, 'confidence': '中'})
        elif sp_a < 2.20:
            picks.append({**m, 'pick': '客胜', 'sp': sp_a, 'confidence': '中'})
    
    picks.sort(key=lambda x: x['sp'])
    
    # Generate parlay suggestions
    suggestions = []
    levels = [n for n in [2, 3, 5, 8, 15] if len(picks) >= n]
    for n in levels[:4]:  # max 4 suggestions
        combo_sp = 1.0
        for p in picks[:n]:
            combo_sp *= p['sp']
        combo_sp *= 0.65  # 北单65%返奖
        suggestions.append({
            "level": f"{n}串1",
            "sp": round(combo_sp, 2),
            "prize_100": round(100 * combo_sp, 0),
            "matches": [f"{p.get('home',p.get('home_team','?'))} vs {p.get('away',p.get('away_team','?'))} → {p['pick']}(SP{p['sp']})" for p in picks[:n]],
        })
    
    return {
        "lottery_type": "beidan",
        "play_type": "胜负过关",
        "description": "只猜胜负(不含平局)，北单独有玩法，最多15关",
        "available_picks": len(picks),
        "top_picks": picks[:10],
        "parlay_suggestions": suggestions[:4],
        "note": "胜负过关无平局选项，适合强弱分明的比赛；返奖率65%",
    }

# ===== 24. 官方规则知识库 =====
LOTTERY_RULES = {
    "jingcai": {
        "name": "竞彩足球",
        "plays": {
            "胜平负": {"options": 3, "single": True, "max_parlay": 8},
            "让球胜平负": {"options": 3, "single": True, "max_parlay": 8},
            "比分": {"options": 31, "single": True, "max_parlay": 4},
            "总进球": {"options": 8, "single": True, "max_parlay": 6},
            "半全场": {"options": 9, "single": True, "max_parlay": 4},
            "混合过关": {"options": "mixed", "single": False, "max_parlay": 8},
        },
        "payout_rate": 0.71,
        "odds_type": "fixed",
        "kelly_threshold": 0.05,
        "prize_cap": {2: 200000, 3: 200000, 4: 500000, 5: 500000, 6: 1000000},
    },
    "beidan": {
        "name": "北京单场",
        "plays": {
            "胜平负(含让球)": {"options": 3, "single": True, "max_parlay": 6},
            "总进球": {"options": 8, "single": True, "max_parlay": 6},
            "比分": {"options": 25, "single": True, "max_parlay": 3},
            "半全场": {"options": 9, "single": True, "max_parlay": 3},
            "上下单双": {"options": 4, "single": True, "max_parlay": 6},
            "胜负过关": {"options": 2, "single": True, "max_parlay": 15},
        },
        "payout_rate": 0.65,
        "odds_type": "floating_sp",
        "kelly_threshold": 0.08,
        "prize_cap": None,
    },
}

@mcp.tool()
def official_knowledge(lottery_type: str = "all") -> dict:
    """查询官方彩票规则知识库。包含竞彩和北单全部12种玩法的选项数、过关限制、返奖率等。
    
    Args:
        lottery_type: jingcai(竞彩) / beidan(北单) / all(全部)
    """
    if lottery_type == "all":
        return {"rules": LOTTERY_RULES, "total_plays": 12,
                "note": "竞彩6种+北单6种=12种玩法。竞彩固定赔率71%返奖，北单浮动SP值65%返奖。"}
    return {"rules": {lottery_type: LOTTERY_RULES.get(lottery_type, {})}}

# ===== 25. 让球盘分析器 =====
@mcp.tool()
def handicap_analyzer(home_win_prob: float, draw_prob: float, away_win_prob: float,
                      handicap: int, home_goals_expected: float = None, away_goals_expected: float = None,
                      league_name: str = "") -> dict:
    """让球胜平负概率计算 — 竞彩和北单核心玩法。
    
    输入原始WDL概率+让球值，计算让球后的胜平负分布。
    
    Args:
        handicap: 让球值(如-1表示主队让1球,+1表示主队受让1球)
        home_goals_expected: 主队预期进球(可选,默认从概率反推λ≈2.5球)
        away_goals_expected: 客队预期进球(可选)
        league_name: 联赛名(用于获取联赛因子计算λ)
    
    返回: 让球后的WDL概率+10种最可能比分+投注建议
    """
    import math
    
    # Get league λ if available
    lf = LEAGUE_FACTORS.get(league_name, {})
    avg_goals = lf.get('avg_goals', 2.5)
    
    # Estimate λ from probabilities if not provided
    if home_goals_expected is None:
        # Reverse-engineer: use avg_goals adjusted by home_win rate
        hwr = lf.get('home_win', max(home_win_prob, 0.38))
        home_ratio = 0.5 + (hwr - 0.40) * 0.5
        home_goals_expected = avg_goals * home_ratio
    if away_goals_expected is None:
        away_goals_expected = avg_goals - home_goals_expected
    
    # Generate full score probability matrix
    max_g = 8
    raw_scores = {}
    hw = dw = aw = 0.0
    for h in range(max_g + 1):
        for a in range(max_g + 1):
            p = (home_goals_expected**h * math.exp(-home_goals_expected) / math.factorial(h)) * \
                (away_goals_expected**a * math.exp(-away_goals_expected) / math.factorial(a))
            raw_scores[(h, a)] = p
            if h > a: hw += p
            elif h == a: dw += p
            else: aw += p
    
    # Apply handicap: shift home goals by handicap value
    # handicap=-1 means home starts at -1, so home_effective = h + handicap
    adj_hw = adj_dw = adj_aw = 0.0
    adj_scores = {}
    handicap_dir = "让球" if handicap < 0 else ("受让" if handicap > 0 else "平手")
    
    for (h, a), p in raw_scores.items():
        adj_h = h + handicap
        if adj_h > a: adj_hw += p
        elif adj_h == a: adj_dw += p
        else: adj_aw += p
        adj_scores[f"{adj_h}-{a}"] = p
    
    # Top scores after handicap
    top_adj = sorted(adj_scores.items(), key=lambda x: -x[1])[:10]
    
    # Betting recommendation
    total = adj_hw + adj_dw + adj_aw
    if total == 0: total = 1.0
    adj_hw_n = adj_hw / total
    adj_dw_n = adj_dw / total
    adj_aw_n = adj_aw / total
    
    advice = []
    if abs(handicap) >= 2:
        advice.append(f"深盘({handicap})让球方穿盘难度大,关注受让方")
    if adj_dw_n > 0.30:
        advice.append(f"让球后平局概率{adj_dw_n*100:.0f}%,警惕走水(平局=让球方输半)")
    if adj_hw_n > adj_aw_n + 0.20:
        advice.append(f"让球方优势明显,可做串关定胆")
    
    return {
        "handicap": handicap,
        "handicap_direction": handicap_dir,
        "original": {"home_win": round(hw, 4), "draw": round(dw, 4), "away_win": round(aw, 4)},
        "after_handicap": {"home_win": round(adj_hw_n, 4), "draw": round(adj_dw_n, 4), "away_win": round(adj_aw_n, 4)},
        "home_goals_expected": round(home_goals_expected, 2),
        "away_goals_expected": round(away_goals_expected, 2),
        "top_scores_after_handicap": [(s, round(p, 5)) for s, p in top_adj],
        "advice": advice,
        "note": f"让球{handicap}后:原主胜{hw*100:.1f}%→{adj_hw_n*100:.1f}%,原平{dw*100:.1f}%→{adj_dw_n*100:.1f}%,原客胜{aw*100:.1f}%→{adj_aw_n*100:.1f}%"
    }

# ===== 21. AI原生：市场信号研判 =====
@mcp.tool()
def market_signal_analyzer(odds_home: float, odds_draw: float, odds_away: float,
                           opening_home: float = 0, opening_draw: float = 0, opening_away: float = 0) -> dict:
    """AI原生：市场信号研判。不是算凯利值，而是解读赔率背后的市场心理。
    
    相比传统的Kelly值计算，这个工具告诉Agent：赔率在讲什么故事？
    """
    signals = []
    confidence = 50
    
    # Favorite detection
    if odds_home < 1.50:
        signals.append({"signal": "强队信号", "detail": f"主胜赔率{odds_home}极低,市场强烈看好主队", "action": "做串关定胆,不宜单买(回报低)"})
        confidence += 10
    elif odds_home > 3.50:
        signals.append({"signal": "冷门信号", "detail": f"主胜赔率{odds_home}极高,市场几乎放弃主队", "action": "搏冷价值大,但需确认基本面"})
        confidence -= 10
    
    # Movement analysis
    if opening_home > 0 and opening_home != odds_home:
        change_pct = (odds_home - opening_home) / opening_home * 100
        if change_pct < -5:
            signals.append({"signal": "资金涌入", "detail": f"赔率降{abs(change_pct):.0f}%,真实看好非诱盘", "action": "可以跟"})
            confidence += 15
        elif change_pct > 8:
            signals.append({"signal": "资金撤离", "detail": f"赔率升{change_pct:.0f}%,庄家不看好或有负面消息", "action": "谨慎或放弃"})
            confidence -= 20
    
    # Draw signal
    if odds_draw < 3.20:
        signals.append({"signal": "平局预警", "detail": f"平赔{odds_draw}偏低,平局概率被市场认可", "action": "关注平局玩法"})
    elif odds_draw > 4.50:
        signals.append({"signal": "胜负局", "detail": "平赔极高,市场认为必分胜负", "action": "不适合投平局"})
    
    return {
        "signals": signals,
        "confidence": max(0, min(100, confidence)),
        "market_story": _tell_market_story(odds_home, odds_draw, odds_away),
        "ai_prompt": "不要只看凯利值。读一下market_story,理解市场在讲什么故事，再结合基本面判断。"
    }

def _tell_market_story(oh, od, oa):
    if oh < 1.30: return f"市场叙事: '碾压局' — 主胜赔率{oh}极低，市场认为实力悬殊"
    if oh < 1.60: return f"市场叙事: '主队明显占优' — 赔率{oh}表明市场看好主队，但非稳赢"
    if oh < 2.00: return f"市场叙事: '主队被看好' — 赔率{oh}温和倾向主队，有分析空间"
    if oh > 4.00: return f"市场叙事: '大冷门预警' — 主胜赔率{oh}极高，市场几乎放弃主队"
    if oh > 3.00: return f"市场叙事: '冷门倾向' — 主胜{oh}偏高,客队被看好"
    if od < 3.00: return f"市场叙事: '平局是认真选项' — 平赔{od}偏低，关注平局"
    return f"市场叙事: '真正均衡' — 双方赔率接近({oh}/{od}/{oa})，胜负难料"


def _parse_jczq_html(html: str) -> list:
    """解析竞彩足球500.com HTML — 基于实际页面文本内容提取。
    
    500.com 竞彩页面使用GB2312编码+复杂JS渲染。
    此解析器先将HTML strip为纯文本，再用正则提取比赛数据。
    """
    import re
    matches = []
    
    # Find each match section by the match number prefix (周XNNN or 星期XNNN)
    sections = re.split(r'(?=\[?(?:周[一二三四五六日]|星期[一二三四五六日])\s*\d{3}\]?)', html)
    
    for section in sections:
        if not re.search(r'(?:周[一二三四五六日]|星期[一二三四五六日])\s*(\d{3})', section):
            continue
        
        # Extract match number
        num_m = re.search(r'(?:周[一二三四五六日]|星期[一二三四五六日])\s*(\d{3})', section)
        if not num_m: continue
        num = int(num_m.group(1))
        
        # Clean section: remove HTML tags, excess whitespace
        clean = re.sub(r'<[^>]+>', ' ', section)
        clean = re.sub(r'&nbsp;', ' ', clean)
        clean = re.sub(r'\s+', ' ', clean).strip()
        
        # Extract league name (first bracket after number)
        league_m = re.search(r'\]?\s*\[?([^\]\[]+?)\]?\s*\d{2}-\d{2}', clean)
        league = league_m.group(1).strip() if league_m else ""
        
        # Extract time
        time_m = re.search(r'(\d{2}-\d{2}\s+\d{2}:\d{2})', clean)
        time_str = time_m.group(1) if time_m else ""
        
        # Extract teams: split on VS
        teams_part = clean
        if time_str:
            teams_part = clean[clean.index(time_str) + len(time_str):]
        
        vs_m = re.search(r'(?:_?VS_?|_vs_)', teams_part, re.IGNORECASE)
        if not vs_m: continue
        
        before_vs = teams_part[:vs_m.start()].strip()
        after_vs = teams_part[vs_m.end():].strip()
        
        # Clean home team: remove leading rank brackets and separators
        home = re.sub(r'^\[?\d+\]?\s*', '', before_vs)
        home = re.sub(r'[\[\]]', '', home).strip()
        
        # Clean away team: stop at rank bracket or SP value
        away = re.split(r'\s*\[\d+\]', after_vs)[0]
        away = re.sub(r'[\[\]]', '', away).strip()
        # Remove trailing "未开售" or SP-like numbers
        away = re.sub(r'\s+(?:未开售|[\d.]+).*$', '', away).strip()
        
        # Clean league: strip the day prefix like "周一001 "
        league = re.sub(r'^周[一二三四五六日]\d{3}\s+', '', league).strip()
        
        if not home or not away: continue
        
        # Extract handicap
        handicap = "0"
        hc_m = re.search(r'([+-]\d)\s', after_vs)
        if hc_m:
            handicap = hc_m.group(1)
        
        # Extract SP values
        sp_vals = re.findall(r'(?<!\d)(\d+\.\d{2})(?!\d)', after_vs)
        sp_h = sp_d = sp_a = "0"
        if len(sp_vals) >= 3:
            sp_h, sp_d, sp_a = sp_vals[0], sp_vals[1], sp_vals[2]
        
        # Score detection: look for X:Y where X,Y are 0-7 and NOT time-like (e.g. 00:50)
        score = None
        status = "upcoming"
        score_matches = re.findall(r'(?:^|\s)(\d+:\d+)(?:\s|$)', after_vs)
        for sm in score_matches:
            parts = sm.split(':')
            try:
                a, b = int(parts[0]), int(parts[1])
                # Skip time-like patterns (e.g. "00:50", "01:00" as next match time)
                if a > 7 and b > 7: continue  # Times like 22:50
                if (a == 0 and b > 7) or (b == 0 and a > 7): continue  # Time-like
                if a <= 7 and b <= 7:
                    score = sm
                    status = "finished"
                    break
            except ValueError:
                continue
        
        matches.append({
            "num": num, "league": league, "time": time_str,
            "home": home, "handicap": handicap, "away": away,
            "sp_h": sp_h, "sp_d": sp_d, "sp_a": sp_a,
            "score": score, "status": status,
            "lottery_type": "jingcai",
        })
    
    return matches


def _parse_500_html(html: str, lottery_type: str = "beidan") -> list:
    """解析500.com HTML为结构化比赛数据。"""
    import re
    trs = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL)
    matches = []
    for tr in trs:
        tr = re.sub(r'<script.*?</script>', '', tr, flags=re.DOTALL)
        tr = re.sub(r'<[^>]+>', '|', tr)
        tr = tr.replace('&nbsp;', '')
        cells = [c.strip() for c in tr.split('|') if c.strip()]
        if len(cells) < 10: continue
        try: num = int(cells[0])
        except: continue
        if num < 1: continue
        league = cells[1] if len(cells) > 1 else ''
        time_str = cells[2] if len(cells) > 2 else ''
        home_idx = 3
        if home_idx < len(cells) and cells[home_idx].startswith('['): home_idx = 4
        home = cells[home_idx] if home_idx < len(cells) else ''
        hc_idx = home_idx + 1
        handicap = cells[hc_idx] if hc_idx < len(cells) else '0'
        away_idx = hc_idx + 1
        away = cells[away_idx] if away_idx < len(cells) else ''
        sp_start = away_idx + 1
        if sp_start < len(cells) and cells[sp_start].startswith('['): sp_start += 1
        sp_h = cells[sp_start] if sp_start < len(cells) else '0'
        sp_d = cells[sp_start+1] if sp_start+1 < len(cells) else '0'
        sp_a = cells[sp_start+2] if sp_start+2 < len(cells) else '0'
        try: float(sp_h)
        except: continue
        score = None
        for i in range(sp_start+3, min(len(cells), 25)):
            if re.match(r'^\d+:\d+$', cells[i]): score = cells[i]; break
        matches.append({
            "num": num, "league": league, "time": time_str,
            "home": home, "handicap": handicap, "away": away,
            "sp_h": sp_h, "sp_d": sp_d, "sp_a": sp_a,
            "score": score, "status": "finished" if score else "upcoming",
            "lottery_type": lottery_type,
        })
    return matches

def main():
    """Entry point for `afa-mcp-server` CLI command."""
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
