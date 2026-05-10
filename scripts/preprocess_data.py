#!/usr/bin/env python3
"""AFA MCP 数据预处理 — 从15.9万场历史数据生成校准表/联赛因子/ELO"""
import json, zipfile, math, sys
from collections import defaultdict
from pathlib import Path

ZIP_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/Users/jand/Desktop/INTEGRATED_COMPLETE_DATA.json.zip")
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/Users/jand/Projects/afa-mcp-server/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"📦 Loading {ZIP_PATH}...")
with zipfile.ZipFile(ZIP_PATH) as z:
    with z.open('INTEGRATED_COMPLETE_DATA.json') as f:
        data = json.load(f)

matches = data['matches']
print(f"✅ Loaded {len(matches)} matches")

# ===== 1. 赔率校准表 (Odds Calibration) =====
print("📊 Generating odds calibration...")
odds_buckets = defaultdict(lambda: {"total": 0, "wins": 0})
for m in matches:
    odds_data = m.get('three_way_odds', {})
    closing = odds_data.get('closing', {})
    if not closing:
        continue
    # Use Bet365 closing odds as primary
    b365 = closing.get('Bet365', closing.get('WilliamHill', {}))
    if not b365:
        continue
    home_odds = b365.get('home', 0)
    result = m.get('result', '')
    if not home_odds or home_odds <= 1.0:
        continue
    # Bucket by odds range
    bucket = round(home_odds * 20) / 20  # 0.05 granularity
    bucket = round(bucket, 2)
    odds_buckets[bucket]["total"] += 1
    if result == 'H':
        odds_buckets[bucket]["wins"] += 1

calibration = []
for odds in sorted(odds_buckets.keys()):
    b = odds_buckets[odds]
    if b["total"] >= 30:  # Minimum sample size
        calibration.append({
            "odds_max": odds + 0.025,
            "actual_win_rate": round(b["wins"] / b["total"], 4),
            "sample_size": b["total"],
            "advice": "强队稳胆" if odds < 1.35 else (
                "正期望" if odds < 1.85 else (
                "谨慎" if odds < 2.40 else (
                "避单选" if odds < 2.90 else "冷门区")))
        })

# Add terminal bucket
calibration.append({"odds_max": 999, "actual_win_rate": 0.22, "sample_size": 0, "advice": "数据不足"})

with open(OUT_DIR / "odds_calibration.json", "w") as fp:
    json.dump(calibration, fp, ensure_ascii=False, indent=2)
print(f"   → odds_calibration.json: {len(calibration)} buckets")

# ===== 2. 联赛因子 (League Factors) =====
print("📊 Generating league factors...")
league_stats = defaultdict(lambda: {"matches": 0, "goals": 0, "home_wins": 0, "draws": 0, 
                                      "over25": 0, "over35": 0, "btts": 0,
                                      "home_goals": 0, "away_goals": 0})
for m in matches:
    ln = m.get('league_name', '')
    if not ln:
        continue
    hg = m.get('home_goals', 0) or 0
    ag = m.get('away_goals', 0) or 0
    if hg is None or ag is None:
        continue
    ls = league_stats[ln]
    ls["matches"] += 1
    ls["goals"] += hg + ag
    ls["home_goals"] += hg
    ls["away_goals"] += ag
    if hg > ag: ls["home_wins"] += 1
    elif hg == ag: ls["draws"] += 1
    if hg + ag > 2.5: ls["over25"] += 1
    if hg + ag > 3.5: ls["over35"] += 1
    if hg > 0 and ag > 0: ls["btts"] += 1

league_factors = {}
for ln, ls in sorted(league_stats.items()):
    n = ls["matches"]
    if n < 50: continue
    league_factors[ln] = {
        "avg_goals": round(ls["goals"] / n, 2),
        "home_win": round(ls["home_wins"] / n, 4),
        "draw": round(ls["draws"] / n, 4),
        "over25": round(ls["over25"] / n, 4),
        "over35": round(ls["over35"] / n, 4),
        "btts": round(ls["btts"] / n, 4),
        "home_goals_avg": round(ls["home_goals"] / n, 2),
        "away_goals_avg": round(ls["away_goals"] / n, 2),
        "sample_size": n,
    }

with open(OUT_DIR / "league_factors.json", "w") as fp:
    json.dump(league_factors, fp, ensure_ascii=False, indent=2)
print(f"   → league_factors.json: {len(league_factors)} leagues")

# ===== 3. ELO 评分 (基于赛果迭代) =====
print("📊 Computing ELO ratings (iterating through 158k matches)...")
elo = defaultdict(lambda: 1500.0)
K = 32
for m in sorted(matches, key=lambda x: (x.get('date','2000'), x.get('time','00:00') or '00:00')):
    ht = m.get('home_team', '')
    at = m.get('away_team', '')
    if not ht or not at:
        continue
    he, ae = elo[ht], elo[at]
    exp_home = 1.0 / (1.0 + 10.0 ** ((ae - he - 65.0) / 400.0))
    hg = m.get('home_goals', 0) or 0
    ag = m.get('away_goals', 0) or 0
    actual = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
    # Scale K by goal difference
    margin = abs(hg - ag)
    k_factor = K * (1 + min(margin, 5) * 0.3)
    elo[ht] = round(he + k_factor * (actual - exp_home), 1)
    elo[at] = round(ae + k_factor * ((1 - actual) - (1 - exp_home)), 1)

with open(OUT_DIR / "elo_ratings.json", "w") as fp:
    json.dump(dict(sorted(elo.items())), fp, ensure_ascii=False, indent=2)
print(f"   → elo_ratings.json: {len(elo)} teams")
print(f"   Top 5: {sorted(elo.items(), key=lambda x:-x[1])[:5]}")
print(f"\n✅ All data files generated in {OUT_DIR}")
