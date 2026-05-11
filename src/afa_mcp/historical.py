"""统一历史数据访问层 — 延迟加载,全局缓存,消除5处重复zip读取"""
import zipfile, json
from pathlib import Path

HIST_ZIP = Path("/Users/jand/Desktop/INTEGRATED_COMPLETE_DATA.json.zip")

class HistoricalData:
    __slots__ = ('_matches',)
    _matches = None
    
    def load(self):
        if self._matches is None and HIST_ZIP.exists():
            with zipfile.ZipFile(HIST_ZIP) as z:
                with z.open('INTEGRATED_COMPLETE_DATA.json') as f:
                    self._matches = json.loads(f.read())["matches"]
        return self._matches or []
    
    def team_recent(self, team, n=10):
        ms = [m for m in self.load() 
              if m.get("home_team")==team or m.get("away_team")==team
              if m.get("home_goals") is not None]
        ms.sort(key=lambda x: x.get("date",""), reverse=True)
        return ms[:n]
    
    def h2h(self, home, away, n=10):
        ms = [m for m in self.load()
              if (m.get("home_team")==home and m.get("away_team")==away)
              or (m.get("home_team")==away and m.get("away_team")==home)]
        ms.sort(key=lambda x: x.get("date",""), reverse=True)
        return ms[:n]
    
    def with_odds(self, min_year=2016):
        return [m for m in self.load()
                if m.get("year",0) >= min_year
                and m.get("three_way_odds",{}).get("closing")]
    
    def count(self):
        return len(self.load())

HIST = HistoricalData()
