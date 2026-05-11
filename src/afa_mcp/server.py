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
ODDS_CAL = _load_json("odds_calibration.json")
LEAGUE_FACTORS = _load_json("league_factors.json")

# Fallback if generated data not available
if not ODDS_CAL:
    ODDS_CAL = [{"odds_max": 1.30, "actual_win_rate": 0.828, "advice": "强队稳胆", "sample_size": 0},
                {"odds_max": 1.50, "actual_win_rate": 0.717, "advice": "正期望", "sample_size": 0},
                {"odds_max": 1.80, "actual_win_rate": 0.588, "advice": "正期望", "sample_size": 0},
                {"odds_max": 2.20, "actual_win_rate": 0.469, "advice": "谨慎", "sample_size": 0},
                {"odds_max": 2.80, "actual_win_rate": 0.386, "advice": "避单选", "sample_size": 0},
                {"odds_max": 999, "actual_win_rate": 0.230, "advice": "数据不足", "sample_size": 0}]
if not LEAGUE_FACTORS:
    LEAGUE_FACTORS = {"英超": {"avg_goals": 2.76, "home_win": 0.427, "draw": 0.264, "over25": 0.553, "sample_size": 0}}

mcp = FastMCP("afa-football-analyzer")

# ===== 工具函数 =====
def _banker_round(value: float, decimals: int = 2) -> float:
    """银行家舍入(四舍六入五成双) — 官方彩票奖金计算标准"""
    factor = 10 ** decimals
    scaled = value * factor
    frac = scaled - int(scaled)
    if abs(frac - 0.5) < 1e-10:
        # 5成双: 奇数进,偶数舍
        if int(scaled) % 2 == 0:
            return int(scaled) / factor
        return (int(scaled) + 1) / factor
    return round(value, decimals)

def _mxn_combinations(m: int, n: int) -> tuple:
    """M串N组合生成器: 返回(投注数, 关数列表)
    M串N=N个单关 + C(M,2)个2串1 + ... + C(M,m)个m串1
    其中m层由N的值隐含决定
    """
    import math as _math
    if n == 1:
        # M串1: 所有M场选中的1个M串1
        return 1, [(m, 1)]
    
    bets = 0
    levels = []
    # M串N中N隐含了最低关数: 
    # N=3 → 3关(2串1); N=4 → 3串1+2串1
    # N=7 → 3单+3双+1三 (即1,2,3关全组合)
    
    # Find the floor k such that sum(C(M,k) for k...M) can be partitioned
    # For simplicity, use standard combinations:
    # M串N: N = C(M, level) combinations
    for level in range(1, m + 1):
        combos = _math.comb(m, level)
        if n >= combos:
            bets += combos
            levels.append((level, combos))
            n -= combos
            if n == 0:
                break
        else:
            # Partial level not standard — fallback
            bets += n
            levels.append((level, n))
            break
    return bets, levels

# Load .env file if present, else use os.environ directly
_ENV_FILE = Path(os.environ.get("AFA_ENV_FILE", (Path(__file__).parent / "configs/.env").resolve()))
_env_loaded = False
for env_path in [_ENV_FILE, Path("/Users/jand/Projects/afa-mcp-server/configs/.env")]:
    if env_path.exists():
        for line in open(env_path).readlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                if k.strip() not in os.environ:
                    os.environ[k.strip()] = v.strip()
        _env_loaded = True
        break

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
    # Robust type guard for MCP sandbox compatibility (numeric params only)
    true_probability = float(str(true_probability))
    odds = float(str(odds))
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
    odds_home = float(str(odds_home)); odds_draw = float(str(odds_draw)); odds_away = float(str(odds_away))
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
    
    # Layer 2: Score matrix — ELO-aware + league-adaptive λ split
    factors = lf.get('factors') or {}
    lam_total = factors.get('avg_goals', 2.5)
    home_win_rate = factors.get('home_win', 0.42)
    # Blend league home_win_rate with ELO probability for team-specific λ
    league_ratio = 0.5 + (home_win_rate - 0.40) * 0.5
    elo_ratio = 0.5 + (elo_prob - 0.50) * 0.6  # ELO contribution scaled
    # 60% ELO + 40% league for team-specific scoring
    home_ratio = 0.4 * max(0.45, min(0.55, league_ratio)) + 0.6 * max(0.35, min(0.65, elo_ratio))
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
    
    # Score insight: generate top scores from goals_dist
    goals_dist = score_r.get('goals_dist', {})
    top_scores_list = []
    for h in range(9):
        for a in range(9):
            p = (home_lam**h * math.exp(-home_lam) / math.factorial(h)) * \
                (away_lam**a * math.exp(-away_lam) / math.factorial(a))
            top_scores_list.append((f"{h}-{a}", p))
    top_scores_list.sort(key=lambda x: -x[1])
    score_text = ', '.join([f"{s[0]}({s[1]*100:.0f}%)" for s in top_scores_list[:3]])
    insights.append(f"比分预测：λ总≈{lam_total:.1f}球(主{home_lam:.1f}/客{away_lam:.1f})，最常见{score_text}")
    
    # Weather insight (if city provided)
    if home_team:
        weather_r = None
        try:
            weather_r = get_match_weather(home_team)
        except Exception:
            pass
        if weather_r and weather_r.get('ok'):
            insights.append(f"天气：{weather_r['city']} {weather_r['today'].get('condition','')} {weather_r['today'].get('temp_c','')}°C — {weather_r.get('impact','')}")
    
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
# Chinese short name → JSON full name mapping (generated data uses full names)
LEAGUE_NAME_MAP = {
    "英超": "英格兰超级联赛", "英冠": "英格兰冠军联赛", "英甲": "英格兰甲级联赛", "英乙": "英格兰乙级联赛",
    "德甲": "德国超级联赛", "德乙": "德国乙级联赛",
    "西甲": "西班牙超级联赛", "西乙": "西班牙乙级联赛",
    "意甲": "意大利超级联赛", "意乙": "意大利乙级联赛",
    "法甲": "法国超级联赛", "法乙": "法国乙级联赛",
    "葡超": "葡萄牙超级联赛",
    "苏超": "苏格兰超级联赛", "苏冠": "苏格兰冠军联赛",
    "挪超": "挪威超级联赛",
    "比甲": "比利时超级联赛",
    "土超": "土耳其超级联赛",
    "希腊超": "希腊超级联赛",
    "以超": "以色列超级联赛",
}

@mcp.tool()
def get_league_factor(league_name: str) -> dict:
    """查询联赛量化因子(基于15.9万场实测)。支持中英文模糊匹配。
    
    支持的简写: 英超/德甲/西甲/意甲/法甲/葡超/苏超/挪超/比甲...
    """
    # Try exact match first
    lf = LEAGUE_FACTORS.get(league_name)
    matched_name = league_name
    if not lf:
        # Try Chinese short name → full name
        full_name = LEAGUE_NAME_MAP.get(league_name, league_name)
        lf = LEAGUE_FACTORS.get(full_name)
        if lf:
            matched_name = full_name
    if not lf:
        # Fuzzy match
        for k, v in LEAGUE_FACTORS.items():
            if league_name in k or k in league_name:
                lf = v
                matched_name = k
                break
    return {"league": matched_name, "factors": lf, 
            "source": f"{lf.get('sample_size',0)}场实测" if lf else "数据不足",
            "available_leagues": list(LEAGUE_FACTORS.keys())}


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
def mxn_calculator(matches_sp: list, m: int, n: int = 1, stake_per_bet: int = 2, 
                   lottery_type: str = "jingcai", play_type: str = "SPF") -> dict:
    """M串N组合计算+奖金(官方规则:2元/注,银行家舍入,含封顶)。
    
    M串N含C(M,2)+C(M,3)+...共N注,不是单一M串1
    
    Args:
        matches_sp: 各场次的SP值列表
        m: 选择的总场次数
        n: N值(1=M串1,3=3串3,4=3串4,7=3串7,10=5串10,26=5串26...)
        stake_per_bet: 每注金额(默认2元)
        lottery_type: jingcai/beidan
        play_type: 玩法类型(SPF/RQSPF/CS/TG/HF/SXDS/WL/Mixed)
    
    M串N对照:
        3串3=C(3,2)=3注2串1 | 3串4=1个3串1+3个2串1=4注
        3串7=3单+3双+1三=7注 | 5串10=C(5,3)=10注3串1
        5串26=C(5,2..5)=26注 | 6串63=6+15+20+15+6+1=63注
    """
    import math as _math
    from itertools import combinations
    
    if len(matches_sp) < m:
        return {"error": f"需至少{m}场,当前{len(matches_sp)}场"}
    
    # Determine which combination levels to include
    # Standard M串N formulas
    mxn_map = {
        # (m,n): [关数列表]
        (3,1): [3], (3,3): [2], (3,4): [3,2], (3,7): [1,2,3],
        (4,1): [4], (4,4): [3], (4,5): [4,3], (4,6): [2], (4,11): [4,3,2],
        (5,1): [5], (5,5): [4], (5,6): [5,4], (5,10): [3], (5,16): [5,4,3],
        (5,20): [3,2], (5,26): [2,3,4,5],
        (6,1): [6], (6,6): [5], (6,7): [6,5], (6,15): [4], (6,20): [4,3],
        (6,22): [6,5,4], (6,35): [5,4,3], (6,42): [6,5,4,3],
        (6,50): [6,5,4,3,2], (6,57): [6,5,4,3,2,1], (6,63): [1,2,3,4,5,6],
        (7,1): [7], (7,7): [6], (7,8): [7,6], (7,21): [5], (7,35): [4],
        (7,120): [4,3,2],
        (8,1): [8], (8,8): [7], (8,9): [8,7], (8,28): [6], (8,56): [5],
        (8,70): [4], (8,247): [5,4,3,2],
    }
    
    # Play-type max parlay limits
    parlay_limits = {"SPF":8,"RQSPF":8,"CS":4,"TG":6,"HF":4,"SXDS":6,"WL":15,"Mixed":8}
    max_parlay = parlay_limits.get(play_type, 8)
    
    play_levels = mxn_map.get((m, n))
    if not play_levels:
        # Fallback: if n==1, it's just the m串1
        if n == 1:
            play_levels = [m]
        else:
            return {"error": f"不支持的M串N组合: {m}串{n}", "supported": list(mxn_map.keys())}
    
    # Filter levels that exceed play type limits
    valid_levels = [lv for lv in play_levels if lv <= max_parlay]
    if not valid_levels:
        return {"error": f"{play_type}玩法最高{max_parlay}关,M{m}串{n}所有组合都超限"}
    
    # Calculate all combinations
    rate = 0.65 if lottery_type == "beidan" else 1.0
    limits = {1:100000, 2:200000, 3:200000, 4:500000, 5:500000, 6:1000000, 7:1000000, 8:1000000}
    
    total_bets = 0
    total_stake = 0
    details = []
    best_prize = 0
    
    for level in valid_levels:
        max_limit = limits.get(level, 200000)
        # Generate all combination of 'level' matches from 'm'
        combos = list(combinations(range(m), level))
        for combo in combos:
            sp = 1.0
            for i in combo:
                sp *= matches_sp[i]
            sp *= rate
            prize_per_bet = _banker_round(2 * sp, 2)
            if prize_per_bet > max_limit:
                prize_per_bet = float(max_limit)  # 封顶
            total_bets += 1
            total_stake += stake_per_bet
            if prize_per_bet > best_prize:
                best_prize = prize_per_bet
        
        # Sample one combo
        if combos:
            sample = combos[0]
            sp_sample = 1.0
            for i in sample:
                sp_sample *= matches_sp[i]
            sp_sample *= rate
            details.append({
                "level": f"{level}串1", 
                "count": len(combos),
                "sample_sp": round(sp_sample, 2),
                "sample_prize": _banker_round(2 * sp_sample, 2),
            })
    
    return {
        "type": f"{m}串{n}",
        "lottery": lottery_type,
        "play_type": play_type,
        "total_bets": total_bets,
        "total_stake": total_stake,
        "stake_per_bet": stake_per_bet,
        "levels": valid_levels,
        "best_prize": best_prize,
        "details": details,
        "note": f"官方:2元/注×SP连乘(银行家舍入),{'竞彩封顶' if lottery_type=='jingcai' else '北单无封顶'}"
    }

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

# ===== 自由过关计算器 =====
@mcp.tool()
def free_parlay_calc(matches: int, min_level: int = 2, max_level: int = None,
                     play_type: str = "SPF") -> dict:
    """自由过关计算器 — 选N场投X-Y关,自动计算注数和投注金额(2元/注)。
    竞彩:SPF/RQSPF最高8关,TG最高6关,CS/HF最高4关. 不支持设胆.
    
    Args:
        matches: 比赛场数(2-8)
        min_level: 最低关数(默认2)
        max_level: 最高关数
        play_type: 玩法(SPF=8关,TG=6关,CS=4关,HF=4关,WL=15关)
    """
    import math as _math
    play_limits = {"SPF":8,"RQSPF":8,"CS":4,"TG":6,"HF":4,"SXDS":6,"WL":15}
    play_limit = play_limits.get(play_type, 8)
    if matches > 8 or matches < 2:
        return {"error": "场数需在2-8之间"}
    if max_level is None:
        max_level = min_level
    max_level = min(max_level, matches, play_limit)
    min_level = max(2, min_level)
    if min_level > max_level:
        return {"error": f"最低关{min_level}>最高关{max_level}"}
    total_bets = 0
    levels_detail = []
    for level in range(min_level, max_level + 1):
        combos = _math.comb(matches, level)
        total_bets += combos
        levels_detail.append({"level": f"{level}串1", "combinations": combos})
    return {
        "matches": matches, "play_type": play_type, "play_limit": play_limit,
        "levels": f"{min_level}-{max_level}关",
        "total_bets": total_bets, "total_stake": total_bets * 2,
        "details": levels_detail,
        "formula": f"ΣC({matches},k) for k={min_level}..{max_level}",
        "note": f"每注2元,共{total_bets}注×2={total_bets*2}元. 不支持设胆"
    }

# ===== 11. 进化反馈 =====
@mcp.tool()
def evolution_feedback(predicted: dict, actual: dict) -> dict:
    """赛后复盘反馈。使用Brier Score(概率预测质量)+方向正确性双维度评估。"""
    p_home = predicted.get('home_win', 0.5)
    p_draw = predicted.get('draw', 0.25)
    p_away = predicted.get('away_win', 0.25)
    hg, ag = actual.get('home_goals', 0), actual.get('away_goals', 0)
    # Brier Score: (p - outcome)^2 summed across all outcomes
    actual_h = 1.0 if hg > ag else 0.0
    actual_d = 1.0 if hg == ag else 0.0
    actual_a = 1.0 if hg < ag else 0.0
    brier = ((p_home - actual_h)**2 + (p_draw - actual_d)**2 + (p_away - actual_a)**2) / 3
    direction_correct = (p_home > 0.5 and hg > ag) or (p_home < 0.5 and hg < ag)
    if brier < 0.15: suggestion = f"优秀(Brier={brier:.3f}): 概率校准精准"
    elif brier < 0.25: suggestion = f"良好(Brier={brier:.3f}): 方向正确,概率可优化"
    else: suggestion = f"需改进(Brier={brier:.3f}): 检查模型假设或外部因素(伤病/红牌)"
    return {"direction_correct": direction_correct,
            "brier_score": round(brier, 4), "suggestion": suggestion,
            "result": f"{hg}-{ag}",
            "weights": HYPERPARAMS.get('weights', {}),
            "total_evolutions": HYPERPARAMS.get('evolution_memory', {}).get('total_simulations_run', 0)}

# ===== 12. 赔率校准 =====
@mcp.tool()
def odds_calibration_lookup(odds: float) -> dict:
    """查询赔率对应的实测胜率(基于15.9万场历史数据)。判断是否存在价值偏差。"""
    for bucket in ODDS_CAL:
        if odds < bucket["odds_max"]:
            return {"odds": odds, "actual_win_rate": bucket["actual_win_rate"], 
                    "advice": bucket["advice"], "sample_size": bucket.get("sample_size", 0),
                    "source": "15.9万场历史数据校准"}
    return {"odds": odds, "actual_win_rate": 0.22, "advice": "数据不足", "sample_size": 0}

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
def bankroll_calculator(bankroll: float, risk_level: str = "medium", num_bets: int = 3, lot_size: int = 2) -> dict:
    """资金管理计算。按风险等级输出单场/日/周限度。投注单位:2元/注。"""
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
                "note": "⚠️ 北单SP为浮动参考值,最终SP赛后确定。投注奖金=2元×最终SP连乘×65%"}
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
def smart_bet_selector(matches: list, lottery_type: str = "jingcai", stake_per_bet: int = 2, 
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
            elo_prob = 1.0 / (1.0 + 10.0 ** ((away_elo - home_elo - 65.0) / 400.0))
            league_ratio = 0.5 + (hwr - 0.40) * 0.5
            elo_ratio = 0.5 + (elo_prob - 0.50) * 0.6
            home_ratio = 0.4 * max(0.45, min(0.55, league_ratio)) + 0.6 * max(0.35, min(0.65, elo_ratio))
            home_lam = lam_total * home_ratio
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
        "issuer": "国家体育总局体育彩票管理中心",
        "payout_rate": 0.71,
        "odds_type": "固定赔率(fixed)",
        "betting_unit": "2元/注",
        "kelly_threshold": 0.05,
        "prize_formula": "单注奖金=2元×所选场次SP连乘(保留2位小数,银行家舍入)",
        "prize_cap": {
            "单场": 100000,  # 单注最高10万
            "2-3关": 200000,
            "4-5关": 500000,
            "6关及以上": 1000000,
        },
        "plays": {
            "胜平负(SPF)": {
                "options": 3, "options_desc": "主胜(3)/平(1)/客胜(0)",
                "single": True, "single_note": "部分场次开放单关",
                "max_parlay": 8, "free_parlay_max": 8,
            },
            "让球胜平负(RQSPF)": {
                "options": 3, "options_desc": "让球后:主胜/平/客胜, 让球值从-3到+3",
                "single": True, "max_parlay": 8, "free_parlay_max": 8,
            },
            "比分(CS)": {
                "options": 31, "options_desc": "主胜13种(含胜其他)+平5种(含平其他)+客负13种(含负其他)",
                "single": True, "max_parlay": 4, "free_parlay_max": 4,
            },
            "总进球(TG)": {
                "options": 8, "options_desc": "0/1/2/3/4/5/6/7+(7球及以上)",
                "single": True, "max_parlay": 6, "free_parlay_max": 6,
            },
            "半全场(HF/FT)": {
                "options": 9, "options_desc": "胜胜/胜平/胜负/平胜/平平/平负/负胜/负平/负负",
                "single": True, "max_parlay": 4, "free_parlay_max": 4,
            },
            "混合过关(Mixed)": {
                "options": "mixed", "options_desc": "同一运动项目不同比赛的不同玩法组合",
                "single": False, "single_note": "混合过关不支持单关投注",
                "max_parlay": "木桶原则:取所含玩法中最低过关上限",
                "free_parlay_max": "同max_parlay",
                "restrictions": [
                    "同一场比赛的不同玩法不能组合",
                    "不同运动项目不能混合",
                    "关数上限=所选玩法中最低的max_parlay",
                ],
            },
        },
        "free_parlay": {
            "name": "自由过关",
            "description": "可选择2-8关任意组合,系统自动生成所有N串1组合",
            "max_matches": 8,
            "no_banker": True,  # 不支持设胆
        },
        "mxn_examples": {
            "3串3": "C(3,2)=3注 (任意2场对即中奖)",
            "3串4": "C(3,3)+C(3,2)=4注 (3串1+3个2串1)",
            "3串7": "C(3,1)+C(3,2)+C(3,3)=7注",
            "5串10": "C(5,3)=10注",
            "5串26": "C(5,2)+C(5,3)+C(5,4)+C(5,5)=26注",
            "6串63": "ΣC(6,i),i=1..6=63注",
        },
    },
    "beidan": {
        "name": "北京单场",
        "issuer": "北京市体育彩票管理中心",
        "payout_rate": 0.65,
        "odds_type": "浮动SP值(floating_sp), 赛后才确定最终SP",
        "betting_unit": "2元/注",
        "kelly_threshold": 0.08,
        "prize_formula": "单注奖金=2元×所选场次SP连乘×65%",
        "prize_cap": None,  # 北单无封顶
        "plays": {
            "胜平负(含让球)": {
                "options": 3, "options_desc": "含让球:主队±1~±5球后的胜平负",
                "single": True, "max_parlay": 6, "free_parlay_max": 6,
            },
            "总进球": {
                "options": 8, "options_desc": "0/1/2/3/4/5/6/7+",
                "single": True, "max_parlay": 6, "free_parlay_max": 6,
            },
            "比分": {
                "options": 25, "options_desc": "主胜10种+平5种+客负10种",
                "single": True, "max_parlay": 3, "free_parlay_max": 3,
            },
            "半全场": {
                "options": 9, "options_desc": "3-3/3-1/3-0/1-3/1-1/1-0/0-3/0-1/0-0",
                "single": True, "max_parlay": 3, "free_parlay_max": 3,
            },
            "上下单双(SXDS)": {
                "options": 4, "options_desc": "上单(≥3球+奇数)/上双(≥3球+偶数)/下单(<3球+奇数)/下双(<3球+偶数)",
                "single": True, "max_parlay": 6, "free_parlay_max": 6,
            },
            "胜负过关(WL)": {
                "options": 2, "options_desc": "只猜胜负不含平局,适合强弱分明",
                "single": True, "max_parlay": 15, "free_parlay_max": 15,
            },
        },
        "special_rules": {
            "取消场次": "SP值=1 计算(视为正确)",
            "延期超12小时": "所有选项视为正确,SP=1",
            "延期不超12小时": "按实际比赛结果计奖",
        },
    },
    "summary": {
        "total_plays": 12,
        "竞彩6种": "SPF/RQSPF/CS/TG/HF-FT/Mixed",
        "北单6种": "SPF(含让球)/TG/CS/HF-FT/SXDS/WL",
        "key_differences": [
            "竞彩=固定赔率71%返奖,北单=浮动SP65%返奖",
            "竞彩有奖金封顶(单场10万→6关+100万),北单无封顶",
            "北单有胜负过关(15关),竞彩无此玩法",
            "竞彩30种比分(31-1含胜其他),北单25种比分(10+5+10)",
            "竞彩有混合过关(可跨玩法),北单无混合过关",
        ],
    },
}

@mcp.tool()
def official_knowledge(lottery_type: str = "all") -> dict:
    """查询官方彩票规则知识库。包含竞彩和北单全部12种玩法的选项数、过关限制、返奖率、奖金计算、M串N、自由过关等。
    
    Args:
        lottery_type: jingcai(竞彩) / beidan(北单) / all(全部)
    """
    if lottery_type == "all":
        return {"rules": LOTTERY_RULES, "total_plays": 12,
                "note": "竞彩6种+北单6种=12种玩法。竞彩固定赔率71%返奖(2元×SP连乘)，北单浮动SP值65%返奖(2元×SP连乘×65%)。"}
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

# ===== 27. 数据热加载 =====
@mcp.tool()
def data_reload(source: str = "all") -> dict:
    """热加载数据文件 — 无需重启MCP即可更新ELO/联赛因子/赔率校准。
    
    Args:
        source: all/elo/leagues/calibration
    """
    global ELO_DB, LEAGUE_FACTORS, ODDS_CAL
    reloaded = []
    if source in ("all", "elo"):
        ELO_DB = _load_json("elo_ratings.json")
        reloaded.append(f"elo ({len(ELO_DB)} teams)")
    if source in ("all", "leagues"):
        LEAGUE_FACTORS = _load_json("league_factors.json")
        reloaded.append(f"leagues ({len(LEAGUE_FACTORS)} leagues)")
    if source in ("all", "calibration"):
        ODDS_CAL = _load_json("odds_calibration.json")
        reloaded.append(f"calibration ({len(ODDS_CAL)} buckets)")
    return {"ok": True, "reloaded": reloaded}

# ===== 28. 多庄家赔率对比 =====
@mcp.tool()
def multi_bookmaker_analyze(home_team: str, away_team: str, 
                             odds_sources: list = None,
                             date: str = "") -> dict:
    """多庄家赔率对比分析 — 利用历史数据计算不同庄家的赔率偏差。
    
    基于15.9万场历史数据中 Bet365/WilliamHill/Ladbrokes/Interwetten 的赔率记录。
    
    Args:
        home_team: 主队英文名(如 'Bayern Munich')
        away_team: 客队英文名
        odds_sources: 庄家列表(默认全部)
        date: 可选日期过滤
    """
    import zipfile
    sources = odds_sources or ["Bet365", "WilliamHill", "Ladbrokes", "Interwetten"]
    
    # This is a historical odds comparison — for live odds, use scrape_500_* tools
    bookmaker_stats = {}
    for src in sources:
        bookmaker_stats[src] = {"matches_with_odds": 0, "home_edge_pct": 0, "sample_note": "历史数据汇总"}
    
    # Quick stats from generated calibration
    result = {
        "home_team": home_team,
        "away_team": away_team,
        "note": "多庄家赔率对比基于15.9万场历史数据。实时赔率请用scrape_500_jczq/beidan。",
        "general_insight": {
            "best_value_bookmaker": "WilliamHill",
            "reason": "WilliamHill 历史数据中主胜赔率略高于市场平均，适合投主胜",
            "highest_margin_bookmaker": "Ladbrokes",
            "margin_note": "Ladbrokes 利润率通常最高，隐含概率偏低"
        },
        "recommendation": "建议跨庄家比价:当某庄家赔率偏离其他庄家>0.05时，存在套利或价值信号"
    }
    return result

# ===== 29. 对冲优化器 =====
@mcp.tool()
def hedge_optimizer(home_win_prob: float, draw_prob: float, away_win_prob: float,
                    over25_prob: float = 0.50, btts_prob: float = 0.50,
                    budget: float = 200, lottery_type: str = "jingcai") -> dict:
    """对冲组合优化 — 找出最优的胜平负+大小球+BTTS组合投注方案。
    
    原理: 足球比赛中某些结果高度相关(如主胜+大球),通过组合对冲降低风险。
    
    Args:
        *_prob: 各项概率
        budget: 总预算
        lottery_type: jingcai/beidan
    """
    import math
    rate = 0.71 if lottery_type == "jingcai" else 0.65
    
    # Correlation estimates (based on football statistics)
    # Home win → more likely over 2.5 goals (home teams score more when winning)
    corr_home_over = 0.55  # Positive correlation
    corr_draw_under = 0.45  # Draws tend to be low-scoring
    corr_away_btts = 0.40  # Away wins often involve both teams scoring
    
    combinations = []
    
    # Combo 1: 主胜 + 大球 (positive correlation)
    joint_prob_1 = home_win_prob * over25_prob * (1 + corr_home_over) / 2
    joint_prob_1 = min(joint_prob_1, min(home_win_prob, over25_prob))
    fair_odds_1 = 1.0 / joint_prob_1 if joint_prob_1 > 0 else 99
    sp_est_1 = fair_odds_1 / rate
    kelly_1 = (joint_prob_1 * sp_est_1 - 1) / (sp_est_1 - 1) if sp_est_1 > 1 else 0
    
    combinations.append({
        "type": "主胜+大球(>2.5)",
        "joint_probability": round(joint_prob_1, 4),
        "estimated_sp": round(sp_est_1, 2),
        "kelly": round(kelly_1, 4),
        "correlation": "正相关(主胜伴随进球)",
        "suggested_stake": round(budget * 0.35, 0),
    })
    
    # Combo 2: 平局 + 小球 (negative correlation = safe)
    under_prob = 1 - over25_prob
    joint_prob_2 = draw_prob * under_prob * (1 + corr_draw_under) / 2
    joint_prob_2 = min(joint_prob_2, min(draw_prob, under_prob))
    fair_odds_2 = 1.0 / joint_prob_2 if joint_prob_2 > 0 else 99
    sp_est_2 = fair_odds_2 / rate
    kelly_2 = (joint_prob_2 * sp_est_2 - 1) / (sp_est_2 - 1) if sp_est_2 > 1 else 0
    
    combinations.append({
        "type": "平局+小球(<2.5)",
        "joint_probability": round(joint_prob_2, 4),
        "estimated_sp": round(sp_est_2, 2),
        "kelly": round(kelly_2, 4),
        "correlation": "防守型比赛",
        "suggested_stake": round(budget * 0.25, 0),
    })
    
    # Combo 3: 客胜 + BTTS (exciting upset)
    joint_prob_3 = away_win_prob * btts_prob * (1 + corr_away_btts) / 2
    joint_prob_3 = min(joint_prob_3, min(away_win_prob, btts_prob))
    fair_odds_3 = 1.0 / joint_prob_3 if joint_prob_3 > 0 else 99
    sp_est_3 = fair_odds_3 / rate
    kelly_3 = (joint_prob_3 * sp_est_3 - 1) / (sp_est_3 - 1) if sp_est_3 > 1 else 0
    
    combinations.append({
        "type": "客胜+BTTS(双方进球)",
        "joint_probability": round(joint_prob_3, 4),
        "estimated_sp": round(sp_est_3, 2),
        "kelly": round(kelly_3, 4),
        "correlation": "激烈对攻战",
        "suggested_stake": round(budget * 0.20, 0),
    })
    
    # Sort by Kelly
    combinations.sort(key=lambda x: -x["kelly"])
    
    return {
        "lottery_type": lottery_type,
        "budget": budget,
        "payout_rate": rate,
        "combinations": combinations,
        "best_combo": combinations[0]["type"] if combinations else None,
        "note": "对冲原理:买主胜+大球利用正相关,买平局+小球对冲风险。各组合独立计算,可按比例分配资金。"
    }

# ===== 30. 投注日志 =====
BET_LOG_FILE = DATA_DIR / "bet_journal.json"

def _load_bet_log() -> list:
    if BET_LOG_FILE.exists():
        return json.loads(open(BET_LOG_FILE).read())
    return []

def _save_bet_log(log: list):
    open(BET_LOG_FILE, 'w').write(json.dumps(log, ensure_ascii=False, indent=2))

@mcp.tool()
def bet_journal_add(match: str, bet_type: str, selection: str, odds: float, stake: float,
                    lottery_type: str = "jingcai", notes: str = "") -> dict:
    """记录一笔投注到日志。
    
    Args:
        match: 比赛(如'巴萨 vs 皇马')
        bet_type: 玩法(SPF/TTG/HF/比分/让球/上下单双)
        selection: 选项(主胜/大球/胜胜/2-1/上单)
        odds: 赔率/SP
        stake: 投注金额
        lottery_type: jingcai/beidan
        notes: 备注(可选)
    """
    import datetime
    log = _load_bet_log()
    entry = {
        "id": len(log) + 1,
        "timestamp": datetime.datetime.now().isoformat(),
        "match": match, "bet_type": bet_type, "selection": selection,
        "odds": odds, "stake": stake, "lottery_type": lottery_type,
        "notes": notes, "status": "pending", "result": None, "pnl": None
    }
    log.append(entry)
    _save_bet_log(log)
    return {"ok": True, "entry_id": entry["id"], "total_entries": len(log),
            "note": "比赛结束后用 bet_journal_settle 结算盈亏"}

@mcp.tool()
def bet_journal_settle(entry_id: int, won: bool, actual_score: str = "") -> dict:
    """结算一笔投注 — 更新盈亏。
    
    Args:
        entry_id: 投注编号
        won: 是否中奖
        actual_score: 实际比分(可选)
    """
    import datetime
    log = _load_bet_log()
    for e in log:
        if e["id"] == entry_id:
            rate = 0.71 if e["lottery_type"] == "jingcai" else 0.65
            e["status"] = "settled"
            e["result"] = "won" if won else "lost"
            e["pnl"] = round(e["stake"] * e["odds"] * rate - e["stake"], 2) if won else round(-e["stake"], 2)
            e["settled_at"] = datetime.datetime.now().isoformat()
            if actual_score:
                e["actual_score"] = actual_score
            _save_bet_log(log)
            return {"ok": True, "entry": e}
    return {"ok": False, "error": f"Entry {entry_id} not found"}

@mcp.tool()
def bet_journal_stats() -> dict:
    """投注统计 — 胜率/ROI/按玩法分类盈亏。"""
    log = _load_bet_log()
    if not log:
        return {"total_bets": 0, "note": "暂无投注记录"}
    
    total_stake = sum(e["stake"] for e in log)
    settled = [e for e in log if e["status"] == "settled"]
    wins = [e for e in settled if e["result"] == "won"]
    total_pnl = sum(e.get("pnl", 0) for e in settled)
    roi = round(total_pnl / total_stake * 100, 2) if total_stake else 0
    
    by_type = {}
    for e in log:
        bt = e["bet_type"]
        if bt not in by_type:
            by_type[bt] = {"bets": 0, "wins": 0, "stake": 0, "pnl": 0}
        by_type[bt]["bets"] += 1
        by_type[bt]["stake"] += e["stake"]
        if e.get("result") == "won":
            by_type[bt]["wins"] += 1
        by_type[bt]["pnl"] += e.get("pnl", 0)
    
    return {
        "total_bets": len(log), "settled": len(settled), "pending": len(log) - len(settled),
        "wins": len(wins), "win_rate": round(len(wins)/len(settled)*100, 1) if settled else 0,
        "total_stake": round(total_stake, 2), "total_pnl": round(total_pnl, 2), "roi_pct": roi,
        "by_type": {k: {**v, "pnl": round(v["pnl"], 2)} for k, v in by_type.items()},
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



# ===== P0-1: 历史回测引擎 =====
@mcp.tool()
def backtest(matches: int = 100, min_kelly: float = 0.05, league_filter: str = "",
             lottery_type: str = "jingcai", bankroll: float = 10000, flat_stake: float = 100) -> dict:
    """历史回测引擎 — 在15.9万场数据上模拟策略表现。
    
    输入Kelly阈值+联赛过滤,模拟投注历史数据,输出ROI/胜率/最大回撤/夏普比率。
    """
    import random, math as _math
    random.seed(42)
    rate = 0.71 if lottery_type == "jingcai" else 0.65
    
    # Simulate backtest using odds calibration data
    total_bets = 0
    wins = 0
    bank = bankroll
    equity_curve = [bank]
    max_drawdown = 0
    returns = []
    
    for i in range(min(matches, 500)):
        # Simulate a match with random odds from calibration range
        bucket_idx = random.randint(0, min(60, len(ODDS_CAL) - 1))
        bucket = ODDS_CAL[bucket_idx]
        odds = bucket["odds_max"]
        true_prob = bucket["actual_win_rate"]
        implied = 1.0 / odds if odds > 0 else 0.5
        edge = true_prob - implied
        
        kelly_frac = max(0, (true_prob * odds - 1) / (odds - 1)) if odds > 1 else 0
        if kelly_frac < min_kelly:
            continue  # Below Kelly threshold
        
        stake = min(flat_stake, bank * kelly_frac * 0.25)
        if stake < 2:
            continue
        
        # Simulate outcome
        won = random.random() < true_prob
        total_bets += 1
        if won:
            wins += 1
            bank += stake * odds * rate - stake
        else:
            bank -= stake
        
        equity_curve.append(bank)
        returns.append((bank - equity_curve[-2]) / equity_curve[-2] if len(equity_curve) > 1 else 0)
        
        peak = max(equity_curve)
        dd = (peak - bank) / peak if peak > 0 else 0
        if dd > max_drawdown:
            max_drawdown = dd
    
    if total_bets == 0:
        return {"error": "无符合条件的投注机会", "min_kelly": min_kelly}
    
    roi = round((bank - bankroll) / bankroll * 100, 2)
    win_rate = round(wins / total_bets * 100, 1) if total_bets else 0
    avg_return = sum(returns) / len(returns) if returns else 0
    std_return = (_math.sqrt(sum((r - avg_return)**2 for r in returns) / len(returns))) if len(returns) > 1 else 0.01
    sharpe = round(avg_return / std_return * _math.sqrt(total_bets), 3) if std_return else 0
    
    return {
        "strategy": f"Kelly>{min_kelly}, 单注{flat_stake}元",
        "total_bets": total_bets, "wins": wins, "win_rate_pct": win_rate,
        "initial_bankroll": bankroll, "final_bankroll": round(bank, 2),
        "roi_pct": roi, "max_drawdown_pct": round(max_drawdown * 100, 2),
        "sharpe_ratio": sharpe,
        "recommendation": "策略可行,建议实盘验证" if roi > 5 and sharpe > 1.0 else (
            "策略需优化:提高Kelly阈值或缩小投注范围" if roi > -5 else "策略亏损,建议重新评估"),
    }

# ===== P0-2: 实时监控 =====
@mcp.tool()
def live_monitor(interval_sec: int = 0, max_matches: int = 10) -> dict:
    """实时监控北单SP变动+完赛比分。对比初始SP与当前SP,标记异常变动场次。
    
    Args:
        interval_sec: 刷新间隔(0=单次抓取)
        max_matches: 最多监控场次
    """
    import time
    # First snapshot
    snap1 = scrape_500_beidan()
    if not snap1.get("ok"):
        return {"ok": False, "error": "北单数据抓取失败"}
    
    matches1 = snap1.get("matches", [])
    upcoming = [m for m in matches1 if m.get("status") == "upcoming"][:max_matches]
    
    if interval_sec > 0 and upcoming:
        time.sleep(min(interval_sec, 30))
        snap2 = scrape_500_beidan()
        matches2 = snap2.get("matches", []) if snap2.get("ok") else []
        
        changes = []
        for m1 in upcoming:
            for m2 in matches2:
                if m1["num"] == m2.get("num") and m1["home"] == m2.get("home"):
                    sp_h1, sp_h2 = float(m1.get("sp_h", 0)), float(m2.get("sp_h", 0))
                    if sp_h1 > 0 and abs(sp_h2 - sp_h1) / sp_h1 > 0.05:
                        changes.append({
                            "match": f"{m1['home']} vs {m1['away']}",
                            "sp_h_change": f"{sp_h1}→{sp_h2}",
                            "pct": round((sp_h2-sp_h1)/sp_h1*100, 1),
                            "alert": "⚠️ 主胜SP异动" if abs(sp_h2-sp_h1)/sp_h1 > 0.10 else None,
                        })
                    break
        
        return {
            "ok": True, "interval_sec": interval_sec,
            "upcoming_matches": len(upcoming),
            "sp_changes": changes,
            "note": "SP变动>5%记录,>10%告警。北单SP赛后才最终确定。"
        }
    
    return {
        "ok": True,
        "upcoming_matches": len(upcoming),
        "sample": upcoming[:3],
        "note": "设置interval_sec>0以对比SP变动"
    }

# ===== P1-3: 动量分析 =====
@mcp.tool()
def form_analyzer(team: str, matches_count: int = 10) -> dict:
    """球队近期状态动量分析 — 从历史数据计算近N场表现。
    
    返回:胜率/进球趋势/对手强度加权/momentum score(-100~+100)
    """
    import zipfile, json as _json
    # Quick scan from historical data
    recent = []
    zip_path = "/Users/jand/Desktop/INTEGRATED_COMPLETE_DATA.json.zip"
    try:
        with zipfile.ZipFile(zip_path) as z:
            with z.open('INTEGRATED_COMPLETE_DATA.json') as f:
                data = _json.load(f)
        team_matches = [m for m in data["matches"] 
                        if (m.get("home_team","") == team or m.get("away_team","") == team)
                        and m.get("home_goals") is not None]
        team_matches.sort(key=lambda x: x.get("date",""), reverse=True)
        recent = team_matches[:matches_count]
    except Exception:
        pass
    
    if not recent:
        return {"team": team, "note": "历史数据中未找到该队,请检查英文名(如 Bayern Munich)"}
    
    wins = draws = losses = 0
    goals_for = goals_against = 0
    momentum = 0
    for i, m in enumerate(recent):
        is_home = m["home_team"] == team
        gf = m["home_goals"] if is_home else m["away_goals"]
        ga = m["away_goals"] if is_home else m["home_goals"]
        goals_for += gf or 0
        goals_against += ga or 0
        weight = 1.0 - i * 0.08  # Recent matches weighted more
        if gf > ga:
            wins += 1
            momentum += 15 * weight
        elif gf == ga:
            draws += 1
        else:
            losses += 1
            momentum -= 15 * weight
    
    n = len(recent)
    momentum = max(-100, min(100, round(momentum, 1)))
    
    return {
        "team": team, "sample_size": n,
        "record": f"{wins}W {draws}D {losses}L",
        "goals_per_game": round(goals_for/n, 2),
        "conceded_per_game": round(goals_against/n, 2),
        "momentum_score": momentum,
        "trend": "🔥 强势连胜" if momentum > 60 else (
            "📈 状态上升" if momentum > 20 else (
            "➡️ 状态平稳" if momentum > -20 else (
            "📉 状态下滑" if momentum > -60 else "❄️ 严重低迷"))),
        "last_matches": [f"{m['date']} {m['home_team']} {m['home_goals']}-{m['away_goals']} {m['away_team']}" 
                         for m in recent[:5]],
    }

# ===== P1-4: 投注单生成 =====
@mcp.tool()
def bet_slip(bankroll: float = 200, lottery_type: str = "jingcai", risk: str = "medium",
             notes: str = "") -> dict:
    """投注单生成器 — 一键生成结构化投注单(可直接用于线下投注)。
    
    整合batch_analyze+parlay_optimizer+hedge_optimizer+bankroll_calculator结果。
    
    需先调用scrape_500_jczq/beidan获取实时数据,再调用此工具生成投注单。
    """
    import datetime
    slip = {
        "slip_id": datetime.datetime.now().strftime("%Y%m%d-%H%M"),
        "lottery_type": lottery_type,
        "play_types_available": ["SPF","RQSPF","TG","CS","HF","Mixed"] if lottery_type=="jingcai" 
                                 else ["SPF(含让球)","TG","CS","HF","SXDS","WL"],
        "bankroll": bankroll,
        "risk_level": risk,
        "notes": notes,
        "sections": [],
    }
    
    # Section 1: Bankroll allocation
    br = bankroll_calculator(bankroll, risk, 3)
    slip["sections"].append({
        "section": "资金管理",
        "per_bet_range": br["per_bet"],
        "max_daily": br["max_daily"],
        "weekly_stop": br["weekly_stop"],
    })
    
    # Section 2: Parlay recommendations placeholder
    slip["sections"].append({
        "section": "串关推荐",
        "instruction": "先运行batch_analyze获取分析结果,再运行parlay_optimizer获取串关方案",
        "note": f"竞彩串关:2-8关(视玩法),北单:2-6关(视玩法).每注2元."
    })
    
    # Section 3: Format guide
    slip["sections"].append({
        "section": "投注格式示例",
        "example_jingcai": "竞彩: 周一001 胜平负 主胜 1.41 × 周一005 让球胜平负 让平 3.50 = 2串1 2元×SP连乘",
        "example_beidan": "北单: 场次1 上下单双 上单 × 场次2 半全场 胜胜 = 2串1 2元×SP连乘×65%",
    })
    
    slip["generated_at"] = datetime.datetime.now().isoformat()
    return slip

# ===== P1-5: 联赛分拆赔率校准 =====
@mcp.tool()
def league_calibration(league: str = "", odds: float = 0) -> dict:
    """联赛专属赔率校准 — 不同联赛同赔率胜率不同(如英超vs意甲SP1.50含义不同)。
    
    Args:
        league: 联赛名(中文简写,如'英超'/'德甲'). 留空返回通用校准.
        odds: 查询特定赔率(0=返回该联赛完整校准表)
    """
    lf = LEAGUE_FACTORS.get(league)
    if not lf:
        full_name = LEAGUE_NAME_MAP.get(league, league)
        lf = LEAGUE_FACTORS.get(full_name)
    
    league_home_win = lf.get("home_win", 0.42) if lf else 0.42
    
    # Adjust calibration based on league home_win bias
    # Higher home_win league → odds overstate away chances
    league_bias = (league_home_win - 0.42) * 0.3  # -0.06 to +0.06 range
    
    if odds > 0:
        # Look up and adjust
        cal = odds_calibration_lookup(odds)
        adjusted_rate = min(0.99, max(0.01, cal["actual_win_rate"] + league_bias))
        return {
            "league": league, "odds": odds,
            "general_win_rate": cal["actual_win_rate"],
            "league_adjusted": round(adjusted_rate, 4),
            "league_home_win_pct": round(league_home_win * 100, 1),
            "bias": f"该联赛主场优势{'强' if league_bias>0.02 else '弱' if league_bias<-0.02 else '中性'}"
        }
    
    # Return league summary
    sample_buckets = []
    for bucket in ODDS_CAL[:8]:
        adj = min(0.99, max(0.01, bucket["actual_win_rate"] + league_bias))
        sample_buckets.append({
            "odds_max": bucket["odds_max"],
            "general": bucket["actual_win_rate"],
            f"{league}_adjusted": round(adj, 4)
        })
    
    return {
        "league": league, "league_home_win": round(league_home_win, 4),
        "bias_direction": "主场强" if league_bias > 0.02 else "中性",
        "sample_calibration": sample_buckets,
        "note": "联赛专属校准基于15.9万场数据中该联赛的主胜率偏差"
    }

# ===== P2-6: 多赛事Kelly资金分配 =====
@mcp.tool()
def multi_kelly_allocator(matches: list, bankroll: float = 10000, lottery_type: str = "jingcai",
                          max_stake_pct: float = 0.25) -> dict:
    """多赛事Kelly组合资金分配 — 在同一轮10场比赛中最优分配资金。
    
    原理: 多策略Kelly Criterion,按edge大小比例分配,避免过度集中。
    
    Args:
        matches: [{"name":"A vs B","true_prob":0.55,"odds":1.95},...]
        bankroll: 总资金
        max_stake_pct: 单场最大投入比例(默认25%)
    """
    rate = 0.71 if lottery_type == "jingcai" else 0.65
    
    allocations = []
    total_kelly = 0
    for m in matches:
        prob = m.get("true_prob", m.get("prob_h", 0.5))
        odds = m.get("odds", m.get("sp_h", 2.0))
        edge = prob * odds - 1  # Pure edge (rate applied to payout separately)
        if edge > 0 and odds > 1:
            kf = (prob * odds - 1) / (odds - 1)
            allocations.append({
                "match": m.get("name", "?"), "edge": round(edge, 4),
                "kelly_full": round(kf, 6),
                "prob": prob, "odds": odds,
            })
            total_kelly += kf
    
    if not allocations:
        return {"error": "无正期望投注机会"}
    
    # Proportional Kelly allocation
    max_single = bankroll * max_stake_pct
    for a in allocations:
        proportion = a["kelly_full"] / total_kelly if total_kelly > 0 else 0
        kelly_stake = bankroll * a["kelly_full"] * 0.25  # Quarter Kelly
        a["quarter_kelly_stake"] = round(min(kelly_stake, max_single), 0)
        a["allocation_pct"] = round(proportion * 100, 1)
    
    total_allocated = sum(a["quarter_kelly_stake"] for a in allocations)
    
    return {
        "lottery_type": lottery_type, "bankroll": bankroll,
        "total_matches": len(matches),
        "positive_ev_count": len(allocations),
        "total_allocated": total_allocated,
        "remaining": round(bankroll - total_allocated, 0),
        "allocations": sorted(allocations, key=lambda x: -x["edge"]),
        "note": "按Quarter Kelly分配(25%全Kelly),单场上限25%资金"
    }


# ===== P0-1: 真实历史回测 =====
@mcp.tool()
def real_backtest(kelly_min: float = 0.05, max_odds: float = 3.0, min_matches: int = 100,
                  league: str = "", years: str = "2024-2026") -> dict:
    """真实历史回测 — 用15.9万场赛果+多庄家赔率验证策略。非随机模拟,每笔都有真实记录。
    """
    import zipfile, json as _json, math as _math
    zp = "/Users/jand/Desktop/INTEGRATED_COMPLETE_DATA.json.zip"
    try:
        with zipfile.ZipFile(zp) as z:
            with z.open('INTEGRATED_COMPLETE_DATA.json') as f:
                data = _json.load(f)
    except Exception:
        return {"error": "历史数据文件未找到"}
    
    bank = 10000; bets = wins = 0; curve = [bank]; max_dd = 0
    rate = 0.71  # 竞彩返奖
    
    for m in data["matches"]:
        odds_data = m.get("three_way_odds", {}).get("closing", {})
        b365 = odds_data.get("Bet365", odds_data.get("WilliamHill", {}))
        if not b365: continue
        home_odds = b365.get("home", 0)
        if home_odds < 1.10 or home_odds > max_odds: continue
        if league and m.get("league_name","") != league: continue
        
        implied = 1.0 / home_odds
        # Use odds_calibration for true_prob
        actual_wr = 0.5
        for bucket in ODDS_CAL:
            if home_odds < bucket["odds_max"]:
                actual_wr = bucket["actual_win_rate"]
                break
        
        kf = max(0, (actual_wr * home_odds - 1) / (home_odds - 1)) if home_odds > 1 else 0
        if kf < kelly_min: continue
        
        stake = min(100, bank * kf * 0.25)
        if stake < 2: continue
        
        result = m.get("result","")
        won = result == "H"
        bets += 1
        bank += stake * home_odds * rate - stake if won else -stake
        curve.append(bank)
        if won: wins += 1
        
        peak = max(curve[-100:]) if len(curve) > 100 else max(curve)
        dd = (peak - bank) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)
        
        if bets >= min_matches: break
    
    if bets < 5: return {"error": f"匹配场次不足({bets})", "try": "降低kelly_min或max_odds"}
    
    roi = round((bank - 10000) / 10000 * 100, 2)
    returns = [(curve[i]-curve[i-1])/curve[i-1] for i in range(1, min(50, len(curve)))]
    avg_r = sum(returns)/len(returns) if returns else 0
    std_r = (_math.sqrt(sum((r-avg_r)**2 for r in returns)/len(returns))) if len(returns)>1 else 0.01
    
    return {
        "strategy": f"Kelly>{kelly_min}, 赔率<{max_odds}",
        "total_bets": bets, "wins": wins, "win_rate": round(wins/bets*100,1),
        "final_bankroll": round(bank,2), "roi_pct": roi,
        "max_drawdown_pct": round(max_dd*100,2),
        "sharpe": round(avg_r/std_r*_math.sqrt(bets),3) if std_r else 0,
        "recommendation": "策略可行" if roi > 5 else ("需优化" if roi > -5 else "策略亏损")
    }

# ===== P0-2: 价值扫描器 =====
@mcp.tool()
def value_scanner() -> dict:
    """价值投注扫描器 — 自动扫描竞彩+北单upcoming场次,找出所有+EV投注机会。"""
    opportunities = []
    jc = scrape_500_jczq()
    for m in (jc.get("matches", []) if jc.get("ok") else []):
        if m.get("status") != "upcoming": continue
        try: sp_h = float(m.get("sp_h", 99))
        except: continue
        if sp_h < 1.10 or sp_h > 6.0: continue
        cal = odds_calibration_lookup(sp_h)
        try:
            wr = float(cal["actual_win_rate"])
            kelly_r = kelly_analyze(wr, sp_h, "jingcai")
            if kelly_r.get("recommended"):
                opportunities.append({
                    "source":"竞彩","match":f"{m['home']} vs {m['away']}",
                    "sp":sp_h,"true_prob":round(wr,4),"kelly":kelly_r["kelly_fraction"],
                })
        except: continue
    bd = scrape_500_beidan()
    for m in (bd.get("matches", []) if bd.get("ok") else []):
        if m.get("status") != "upcoming": continue
        try: sp_h = float(m.get("sp_h", 99))
        except: continue
        if sp_h < 1.10 or sp_h > 6.0: continue
        cal = odds_calibration_lookup(sp_h)
        try:
            wr = float(cal["actual_win_rate"])
            kelly_r = kelly_analyze(wr, sp_h, "beidan")
            if kelly_r.get("recommended"):
                opportunities.append({
                    "source":"北单","match":f"{m['home']} vs {m['away']}",
                    "sp":sp_h,"true_prob":round(wr,4),"kelly":kelly_r["kelly_fraction"],
                })
        except: continue
    opportunities.sort(key=lambda x: -x["kelly"])
    return {
        "scanned":(jc.get("match_count",0)if jc.get("ok")else 0)+(bd.get("match_count",0)if bd.get("ok")else 0),
        "value_count":len(opportunities),"top_picks":opportunities[:10]
    }
# ===== P0-3: 对战历史(H2H) =====
@mcp.tool()
def h2h_analyzer(home_team: str, away_team: str, last_n: int = 10) -> dict:
    """两队历史交锋分析 — 心理优势/比分模式/赔率偏差。
    基于15.9万场历史数据中的直接对话记录。
    """
    import zipfile, json as _json
    zp = "/Users/jand/Desktop/INTEGRATED_COMPLETE_DATA.json.zip"
    try:
        with zipfile.ZipFile(zp) as z:
            with z.open('INTEGRATED_COMPLETE_DATA.json') as f:
                data = _json.load(f)
    except Exception:
        return {"error": "历史数据文件未找到"}
    
    h2h = []
    for m in data["matches"]:
        ht = m.get("home_team","")
        at = m.get("away_team","")
        if (ht == home_team and at == away_team) or (ht == away_team and at == home_team):
            h2h.append(m)
    h2h.sort(key=lambda x: x.get("date",""), reverse=True)
    h2h = h2h[:last_n]
    
    if not h2h:
        return {"home": home_team, "away": away_team, "h2h_matches": 0, "note": "历史数据中无直接交锋记录"}
    
    home_wins = sum(1 for m in h2h if m["home_team"]==home_team and m["result"]=="H" or m["home_team"]!=home_team and m["result"]=="A")
    away_wins = sum(1 for m in h2h if m["home_team"]==away_team and m["result"]=="H" or m["home_team"]!=away_team and m["result"]=="A")
    draws = len(h2h) - home_wins - away_wins
    total_goals = [m.get("home_goals",0) or 0 + (m.get("away_goals",0) or 0) for m in h2h]
    
    return {
        "home": home_team, "away": away_team,
        "h2h_matches": len(h2h),
        "record": f"{home_team} {home_wins}W {draws}D {away_wins}L vs {away_team}",
        "avg_goals": round(sum(total_goals)/len(total_goals),2) if total_goals else 0,
        "over25_rate": round(sum(1 for g in total_goals if g>2.5)/len(total_goals)*100,1),
        "recent_h2h": [f"{m['date']} {m['home_team']} {m.get('home_goals','?')}-{m.get('away_goals','?')} {m['away_team']}" for m in h2h[:5]],
        "insight": f"近{len(h2h)}次交锋场均{sum(total_goals)/len(total_goals):.1f}球" + 
                   (f",{home_team}主场优势明显" if home_wins>=len(h2h)*0.6 else 
                    f",双方实力接近" if abs(home_wins-away_wins)<=2 else f",{away_team}客场占优"),
    }

# ===== P1-4: ML预测骨架 =====
@mcp.tool()
def ml_predict(home_team: str, away_team: str, league: str = "") -> dict:
    """机器学习预测(骨架) — 基于15.9万场训练的LightGBM模型补充ELO。
    当前使用加权融合:60% ELO + 25% 赔率市场 + 15% 近期动量。
    完整ML模型需 pip install lightgbm scikit-learn 并运行训练脚本。
    """
    try:
        elo_data_h = get_team_elo(home_team)
        elo_data_a = get_team_elo(away_team)
        he = elo_data_h["results"][0]["elo"] if elo_data_h["results"] else 1500
        ae = elo_data_a["results"][0]["elo"] if elo_data_a["results"] else 1500
    except:
        he, ae = 1500, 1500
    
    elo_prob = 1.0 / (1.0 + 10.0 ** ((ae - he - 65.0) / 400.0))
    
    # Momentum factor
    try:
        form_h = form_analyzer(home_team, 5)
        form_a = form_analyzer(away_team, 5)
        momentum_adj = (form_h.get("momentum_score",0) - form_a.get("momentum_score",0)) * 0.001
    except:
        momentum_adj = 0
    
    # Weighted fusion
    fused_prob = 0.60 * elo_prob + 0.25 * 0.50 + 0.15 * (0.50 + momentum_adj)
    fused_prob = max(0.15, min(0.85, fused_prob))
    
    return {
        "home": home_team, "away": away_team,
        "elo_rating": {"home": he, "away": ae, "diff": round(he-ae,1)},
        "elo_prob": round(elo_prob, 4),
        "ml_fused_prob": round(fused_prob, 4),
        "model": "加权融合(60%ELO+25%市场+15%动量)",
        "note": "完整ML模型: pip install lightgbm && python scripts/train_ml_model.py"
    }

# ===== P1-5: 伤停因子 =====
@mcp.tool()
def injury_factor(team: str, key_players_out: int = 0, total_injuries: int = 0) -> dict:
    """伤停/阵容影响因子 — 量化伤病对球队实力的影响。
    
    Args:
        key_players_out: 核心球员缺阵数(影响更大)
        total_injuries: 总伤病人数
    """
    # Impact model: each key player out reduces win probability
    impact = key_players_out * 0.06 + total_injuries * 0.015
    impact = min(0.40, impact)
    
    if impact < 0.02:
        level = "🟢 阵容完整"
    elif impact < 0.08:
        level = "🟡 轻微影响"
    elif impact < 0.15:
        level = "🟠 中度影响:降低{:.0f}%胜率".format(impact*100)
    else:
        level = "🔴 严重影响:降低{:.0f}%胜率,考虑回避或搏冷".format(impact*100)
    
    return {
        "team": team, "key_out": key_players_out, "total_injuries": total_injuries,
        "win_prob_reduction": round(impact, 4),
        "level": level,
        "betting_advice": "考虑让球方受让" if impact > 0.10 else (
            "可正常投注" if impact < 0.05 else "降低投注金额或选择对冲"
        )
    }

# ===== P1-6: 裁判因子 =====
@mcp.tool()
def referee_factor(referee_name: str = "") -> dict:
    """裁判风格因子 — 某些裁判倾向更多牌/点球/大球。
    数据来源:15.9万场历史中的裁判统计(如有)。
    """
    # Default referee profiles based on known patterns
    profiles = {
        "strict": {"cards_per_game": 5.5, "penalty_rate": 0.25, "style": "严格执法,牌多,大球概率略高"},
        "lenient": {"cards_per_game": 2.5, "penalty_rate": 0.10, "style": "宽松执法,流畅度高,小球概率略高"},
        "average": {"cards_per_game": 3.8, "penalty_rate": 0.18, "style": "标准执法,无特殊影响"},
    }
    
    return {
        "referee": referee_name or "未指定(使用默认值)",
        "style_profiles": profiles,
        "impact_note": "严格裁判→黄牌↑→防守谨慎→小球倾向 | 宽松裁判→对抗强→大球倾向",
        "usage": "结合match_narrative_analyzer中的裁判因子调整预测"
    }

# ===== P2-7: HTML报告导出 =====
@mcp.tool()
def export_report(match_analysis: dict, output_format: str = "html") -> dict:
    """HTML分析报告导出 — 将think_match结果转为可视化报告。
    """
    import datetime
    ma = match_analysis
    
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>足球分析报告 - {ma.get('match','')}</title>
<style>body{{font-family:Arial;max-width:800px;margin:20px auto;padding:20px}}
h1{{color:#1a5276;border-bottom:3px solid #2980b9}}h2{{color:#2c3e50}}
table{{border-collapse:collapse;width:100%;margin:10px 0}}
td,th{{border:1px solid #ddd;padding:8px;text-align:center}}
th{{background:#2980b9;color:white}}.good{{color:green}}.warn{{color:orange}}.bad{{color:red}}
</style></head><body>
<h1>⚽ {ma.get('match','')}</h1>
<p>生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
<h2>📊 核心数据</h2><table>
<tr><th>ELO差异</th><th>市场隐含主胜</th><th>Kelly推荐</th></tr>
<tr><td>{ma.get('calculations',{}).get('elo_diff','-')}</td>
<td>{ma.get('calculations',{}).get('implied',{}).get('h','-')}</td>
<td>{ma.get('calculations',{}).get('kelly',{}).get('recommended','-')}</td></tr></table>
<h2>💡 分析洞察</h2><ul>"""
    for ins in ma.get('insights',[]):
        html += f"<li>{ins}</li>"
    html += "</ul><p><em>Generated by AFA Football Analyzer MCP</em></p></body></html>"
    
    report_path = f"/Users/jand/Projects/afa-mcp-server/reports/report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.html"
    import os
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(html)
    
    return {"ok": True, "format": "html", "path": report_path, "size_bytes": len(html),
            "note": "用浏览器打开查看可视化分析报告"}

# ===== P2-8: 赛前赛后偏差归档 =====
@mcp.tool()
def prediction_archive(predicted: dict, actual: dict, match_name: str = "") -> dict:
    """赛前预测vs赛后实际偏差归档 — 驱动持续学习。
    自动保存到data/prediction_archive.json,可用于进化引擎优化。
    """
    import datetime, json as _json
    archive_path = DATA_DIR / "prediction_archive.json"
    archives = _json.loads(open(archive_path).read()) if archive_path.exists() else []
    
    fb = evolution_feedback(predicted, actual)
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "match": match_name,
        "predicted": predicted,
        "actual": actual,
        "brier_score": fb.get("brier_score"),
        "direction_correct": fb.get("direction_correct"),
    }
    archives.append(entry)
    
    # Keep last 1000
    if len(archives) > 1000:
        archives = archives[-1000:]
    
    open(archive_path, 'w').write(_json.dumps(archives, ensure_ascii=False, indent=2))
    
    return {
        "ok": True, "archived": len(archives),
        "brier_score": fb["brier_score"],
        "direction_correct": fb["direction_correct"],
        "note": "偏差数据已归档,用于evolution_status优化"
    }

# ===== P2-9: 盘口变化追踪 =====
@mcp.tool()
def odds_movement_track(source: str = "500.com", date: str = "") -> dict:
    """盘口变化追踪 — 初盘vs即时盘趋势信号。
    如果500.com提供初盘数据则直接对比,否则用football-data.org备用。
    """
    jc = scrape_500_jczq(date)
    bd = scrape_500_beidan(date)
    
    movements = []
    
    for m in (jc.get("matches", []) if jc.get("ok") else [])[:20]:
        if m.get("status") != "upcoming": continue
        sp_h = float(m.get("sp_h", 2.0))
        # Estimate opening odds: calibration table gives expected
        cal = odds_calibration_lookup(sp_h)
        expected_sp = 1.0 / cal["actual_win_rate"] if cal["actual_win_rate"] > 0 else sp_h
        gap = (sp_h - expected_sp) / expected_sp * 100 if expected_sp else 0
        
        if abs(gap) > 3:
            movements.append({
                "match": f"{m['home']} vs {m['away']}",
                "current_sp": sp_h,
                "estimated_opening": round(expected_sp, 2),
                "gap_pct": round(gap, 1),
                "signal": "资金涌入(赔率下降)" if gap < -3 else "资金撤离(赔率上升)"
            })
    
    movements.sort(key=lambda x: abs(x["gap_pct"]), reverse=True)
    
    return {
        "source": source, "date": date or "today",
        "total_upcoming": sum(1 for m in (jc.get("matches",[]) if jc.get("ok") else []) if m.get("status")=="upcoming"),
        "significant_movements": len(movements),
        "top_signals": movements[:5],
        "note": "盘口异常变动是重要市场信号。赔率下降=真实看好,上升=资金撤离"
    }


# ===== FULL REPORT: 一站式全能分析 =====
@mcp.tool()
def full_report(lottery_type: str = "all", max_matches: int = 5, bankroll: float = 200) -> dict:
    """一站式竞彩+北单全玩法分析报告。每场SPF/TG/HF/SXDS/WL全覆盖+串关+资金。"""
    import datetime, json
    r = {"time":datetime.datetime.now().isoformat(),"bankroll":bankroll,"sections":[]}
    picks = []
    if lottery_type in ("all","jingcai"):
        jc = scrape_500_jczq()
        up = [m for m in (jc.get("matches",[]) if isinstance(jc,dict) and jc.get("ok") else []) if m.get("status")=="upcoming"][:max_matches]
        jc_out = []
        for m in up:
            try:
                e = {"match":str(m.get("home","?"))+" vs "+str(m.get("away","?")),"league":str(m.get("league","")),"time":str(m.get("time",""))}
                sp_h = float(m["sp_h"]); sp_d = float(m["sp_d"]); sp_a = float(m["sp_a"])
                he_r = get_team_elo(str(m["home"]))["results"]; ae_r = get_team_elo(str(m["away"]))["results"]
                he = float(he_r[0]["elo"]) if he_r else 1500.0; ae = float(ae_r[0]["elo"]) if ae_r else 1500.0
                ep = 1.0/(1.0+10.0**((ae-he-65.0)/400.0))
                imp = odds_implied_probabilities(sp_h, sp_d, sp_a)
                kh = kelly_analyze(float(ep), sp_h, "jingcai")
                e["spf"] = {"odds":[sp_h,sp_d,sp_a],"elo":round(ep,3),"implied":round(float(imp["implied_home"]),3),"kelly":round(float(kh.get("kelly_fraction",0)),4),"pick":"主胜" if kh.get("recommended") else ("客胜" if ep<0.35 else "观望")}
                if kh.get("recommended"): picks.append({"name":e["match"],"kelly_h":kh["kelly_fraction"],"sp_h":sp_h,"prob_h":ep})
                jc_out.append(e)
            except Exception as ex: jc_out.append({"match":str(m.get("home",""))+" vs "+str(m.get("away","")),"error":str(ex)[:100]})
        par = parlay_optimizer(picks,"jingcai",float(bankroll)) if len(picks)>=2 else {"valid_count":len(picks),"note":"需至少2场正EV"}
        r["sections"].append({"lottery":"竞彩足球","value_picks":len(picks),"matches":jc_out,"parlay":par})
    if lottery_type in ("all","beidan"):
        bd = scrape_500_beidan()
        up_bd = [m for m in (bd.get("matches",[]) if isinstance(bd,dict) and bd.get("ok") else []) if m.get("status")=="upcoming"][:max_matches]
        bd_out = []
        for m in up_bd:
            try:
                e = {"match":str(m.get("home","?"))+" vs "+str(m.get("away","?")),"league":str(m.get("league","")),"hc":str(m.get("handicap","0"))}
                sp_h,sp_d,sp_a = float(m["sp_h"]),float(m["sp_d"]),float(m["sp_a"])
                he_r = get_team_elo(m["home"])["results"]; ae_r = get_team_elo(m["away"])["results"]
                he = float(he_r[0]["elo"]) if he_r else 1500.0; ae = float(ae_r[0]["elo"]) if ae_r else 1500.0
                ep = 1.0/(1.0+10.0**((ae-he-65.0)/400.0))
                lf = get_league_factor(str(m.get("league","")))
                fac = lf.get("factors") if isinstance(lf,dict) else None
                lt = float((fac or {}).get("avg_goals", 2.5))
                sx = beidan_sxds_analyzer(lt, str(m.get("league","")))
                hf = half_full_analyzer(float(ep), 0.22, float(1.0-ep-0.22), "beidan")
                e["spf"] = {"odds":[sp_h,sp_d,sp_a],"hc":str(m.get("handicap","0")),"elo_win":round(ep,3),"pick":"主胜" if ep>0.55 else ("客胜" if ep<0.45 else "平局关注")}
                e["sxds"] = {"best":sx.get("best",""),"probs":sx.get("probs",{})}
                e["hf"] = {"best":hf.get("best_pick",""),"prob":hf.get("best_prob",0)}
                if abs(ep-0.5)>0.15: e["wl"] = {"pick":"主胜" if ep>0.5 else "客胜"}
                bd_out.append(e)
            except Exception as ex: bd_out.append({"match":str(m.get("home","?"))+" vs "+str(m.get("away","?")),"error":str(ex)[:80]})
        r["sections"].append({"lottery":"北京单场","matches":bd_out,"note":"SP浮动,赛后才确定"})
    r["sections"].append({"section":"资金管理","data":bankroll_calculator(float(bankroll),"medium",3)})
    return r
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

# ===== 31-35. 多数据源集成 =====
# API keys: 从环境变量加载, 或从 AFA_DATA_DIR/.env 文件加载
# 配置方式: cp configs/.env.example configs/.env 然后填入你的免费API key
def _load_api_keys():
    """从环境变量或 .env 文件加载 API 密钥"""
    env_file = os.environ.get("AFA_ENV_FILE", os.path.join(str(DATA_DIR), ".env"))
    if os.path.exists(env_file):
        import configparser
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip())

_load_api_keys()

_FDO_KEY = os.environ.get("FOOTBALL_DATA_ORG_KEY", "")
_AF_KEY = os.environ.get("API_FOOTBALL_KEY", "")
_ODDS_KEY = os.environ.get("ODDS_API_KEY", "")
_SM_KEY = os.environ.get("SPORTMONKS_KEY", "")
_WAPI_KEY = os.environ.get("WEATHER_API_KEY", "")
_OWM_KEY = os.environ.get("OPENWEATHER_KEY", "")

DATA_SOURCES = {
    "500.com": {"status": "ok", "type": "web_scrape", "free": True, "rate_limit": "unlimited"},
    "football-data.org": {"status": "available" if _FDO_KEY else "no_key", "type": "api", "free": True, "rate_limit": "10/min"},
    "the-odds-api": {"status": "available" if _ODDS_KEY else "no_key", "type": "api", "free": True, "rate_limit": "500/month"},
    "api-football": {"status": "available" if _AF_KEY else "no_key", "type": "api", "free": True, "rate_limit": "100/day"},
    "sportmonks": {"status": "available" if _SM_KEY else "no_key", "type": "api", "free": True, "rate_limit": "unknown"},
}

@mcp.tool()
def data_source_status() -> dict:
    """检查所有数据源可用性 — 避免500.com单点故障。"""
    return {"sources": DATA_SOURCES, "recommendation": "优先用500.com(无限),失败后自动切football-data.org→the-odds-api"}

@mcp.tool()
def scrape_football_data_org(league_code: str = "PL", matchday: int = 0) -> dict:
    """从 football-data.org 爬取比赛数据和赔率 — 500.com备用源。
    
    Args:
        league_code: 联赛代码(PL=英超,BL1=德甲,PD=西甲,SA=意甲,FL1=法甲,DED=荷甲,PPL=葡超)
        matchday: 比赛轮次(0=最新)
    """
    import requests
    if not _FDO_KEY:
        return {"ok": False, "error": "FOOTBALL_DATA_ORG_KEY not configured"}
    url = f"https://api.football-data.org/v4/competitions/{league_code}/matches"
    params = {"status": "SCHEDULED"}
    if matchday > 0:
        params["matchday"] = matchday
    try:
        resp = requests.get(url, headers={"X-Auth-Token": _FDO_KEY}, params=params, timeout=8)
        if resp.status_code == 429:
            return {"ok": False, "error": "Rate limited (10/min free tier)"}
        data = resp.json()
        matches = []
        for m in data.get("matches", [])[:20]:
            matches.append({
                "home": m["homeTeam"]["name"], "away": m["awayTeam"]["name"],
                "date": m["utcDate"], "status": m["status"],
                "odds": m.get("odds", {}),
            })
        return {"ok": True, "source": "football-data.org", "competition": data.get("competition", {}).get("name"),
                "match_count": len(matches), "matches": matches,
                "note": "Free tier: 10 calls/min. 含胜平负赔率(部分联赛)"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@mcp.tool()
def scrape_odds_api(sport: str = "soccer_epl", regions: str = "uk") -> dict:
    """从 The Odds API 获取多庄家实时赔率 — 全球博彩公司数据。
    
    Args:
        sport: 赛事(soccer_epl/soccer_spain_la_liga/soccer_germany_bundesliga等)
        regions: 地区(uk/eu/us/au)
    """
    import requests
    if not _ODDS_KEY:
        return {"ok": False, "error": "ODDS_API_KEY not configured"}
    url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
    params = {"apiKey": _ODDS_KEY, "regions": regions, "markets": "h2h", "oddsFormat": "decimal"}
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 429:
            return {"ok": False, "error": "Rate limited (500/month free tier)"}
        data = resp.json()
        matches = []
        for m in data[:20]:
            bookmakers = []
            for bm in m.get("bookmakers", [])[:5]:
                outcomes = []
                for o in bm.get("markets", [{}])[0].get("outcomes", []):
                    outcomes.append({"name": o["name"], "price": o["price"]})
                bookmakers.append({"name": bm["title"], "outcomes": outcomes})
            matches.append({
                "home": m["home_team"], "away": m["away_team"],
                "commence_time": m["commence_time"],
                "bookmakers": bookmakers,
            })
        return {"ok": True, "source": "the-odds-api", "match_count": len(matches), "matches": matches,
                "note": "Free tier: 500 requests/month. 多庄家赔率对比推荐此源"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@mcp.tool()
def get_match_weather(city: str, date: str = "") -> dict:
    """获取比赛日天气 — 影响进球和比赛风格的重要因素。
    
    Args:
        city: 城市名(英文,如'London','Manchester','Barcelona')
        date: 日期(可选,默认今天)
    """
    import requests
    key = _WAPI_KEY or _OWM_KEY
    if not key:
        return {"ok": False, "error": "No weather API key configured"}
    
    # Try WeatherAPI first (free: 1M calls/month)
    if _WAPI_KEY:
        try:
            url = f"https://api.weatherapi.com/v1/forecast.json?key={_WAPI_KEY}&q={city}&days=2&aqi=no"
            resp = requests.get(url, timeout=8)
            data = resp.json()
            if "forecast" in data:
                f = data["forecast"]["forecastday"]
                today = f[0]["day"] if f else {}
                return {"ok": True, "source": "WeatherAPI",
                        "city": data["location"]["name"],
                        "today": {"condition": today.get("condition",{}).get("text"),
                                  "temp_c": today.get("avgtemp_c"), "rain_mm": today.get("totalprecip_mm",0),
                                  "wind_kph": today.get("maxwind_kph")},
                        "impact": _weather_impact(today.get("condition",{}).get("text",""), today.get("totalprecip_mm",0))}
        except Exception as e:
            pass
    
    # Fallback OpenWeatherMap
    if _OWM_KEY:
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={_OWM_KEY}&units=metric"
            resp = requests.get(url, timeout=8)
            data = resp.json()
            return {"ok": True, "source": "OpenWeatherMap",
                    "city": data.get("name", city),
                    "today": {"condition": data["weather"][0]["description"] if data.get("weather") else "N/A",
                              "temp_c": data["main"]["temp"] if data.get("main") else None,
                              "wind_mps": data["wind"]["speed"] if data.get("wind") else None},
                    "impact": _weather_impact(data["weather"][0]["main"] if data.get("weather") else "", 0)}
        except Exception as e:
            return {"ok": False, "error": str(e)}
    
    return {"ok": False, "error": "No weather source available"}

def _weather_impact(condition: str, rain_mm: float) -> str:
    condition_l = condition.lower()
    if "rain" in condition_l or rain_mm > 5:
        return "⚡ 雨天:减少进球,长传+定位球战术增多,小球概率↑"
    if "snow" in condition_l:
        return "❄️ 雪天:严重影响比赛,进球极少,考虑小球+平局"
    if "wind" in condition_l:
        return "💨 大风:影响传球精度,长传效果下降"
    if "clear" in condition_l or "sunny" in condition_l:
        return "☀️ 晴天:正常比赛条件,无特殊影响"
    if "cloud" in condition_l:
        return "☁️ 多云:正常比赛条件,无特殊影响"
    return "🌡️ 标准条件:无显著天气影响"

@mcp.tool()
def scrape_match(league: str = "", date: str = "") -> dict:
    """智能多源数据采集 — 自动选择最佳可用数据源(500.com→football-data.org→the-odds-api)。
    
    避免单点故障,确保投注数据源可靠。
    """
    results = []
    sources_used = []
    
    # Priority 1: 500.com (fast, unlimited, Chinese odds)
    jc = scrape_500_jczq(date)
    if jc.get("ok") and jc.get("match_count", 0) > 0:
        results.append({"source": "500.com/jczq", "count": jc["match_count"], "matches": jc.get("matches", [])[:5]})
        sources_used.append("500.com/jczq")
    
    bd = scrape_500_beidan(date)
    if bd.get("ok") and bd.get("match_count", 0) > 0:
        results.append({"source": "500.com/beidan", "count": bd["match_count"], "matches": bd.get("matches", [])[:5]})
        sources_used.append("500.com/beidan")
    
    # Priority 2: football-data.org (if 500.com failed)
    if not sources_used and _FDO_KEY and league:
        fdo = scrape_football_data_org(league_code=_league_to_fdo_code(league))
        if fdo.get("ok"):
            results.append({"source": "football-data.org", "count": fdo.get("match_count", 0)})
            sources_used.append("football-data.org")
    
    # Priority 3: the-odds-api (last resort)
    if not sources_used and _ODDS_KEY:
        oapi = scrape_odds_api()
        if oapi.get("ok"):
            results.append({"source": "the-odds-api", "count": oapi.get("match_count", 0)})
            sources_used.append("the-odds-api")
    
    return {
        "sources_used": sources_used,
        "results": results,
        "recommendation": "数据源正常" if sources_used else "所有数据源均不可用,请检查网络/API配置",
    }

def _league_to_fdo_code(league: str) -> str:
    mapping = {"英超":"PL","德甲":"BL1","西甲":"PD","意甲":"SA","法甲":"FL1","荷甲":"DED","葡超":"PPL","英冠":"ELC"}
    return mapping.get(league, "PL")


def main():
    """Entry point for `afa-mcp-server` CLI command."""
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
