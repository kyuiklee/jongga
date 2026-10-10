"""백테스트 결과 검증용 (앱의 계산을 그대로 재현 + 민감도·집중도 점검). 결과: site/data/check.json"""
import os, json, bisect
from datetime import date
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "site", "data")
TR = json.load(open(os.path.join(DATA, "trades.json"), encoding="utf-8"))
BENCH = json.load(open(os.path.join(DATA, "bench.json"), encoding="utf-8"))
ROWS = [dict(zip(TR["fields"], r)) for r in TR["rows"]]
CAL = sorted(x[0] for x in BENCH["kospi"])


def months_back(d, m):
    y, mo, dd = map(int, d.split("-"))
    mo -= m
    while mo <= 0:
        mo += 12; y -= 1
    import calendar
    dd = min(dd, calendar.monthrange(y, mo)[1])
    return f"{y:04d}-{mo:02d}-{dd:02d}"


def run(typ="P", mkt="Q", min_score=65, top=5, N=5, stop=False, fee=0.25, per=24, start=None, end=None, drop=None):
    last = ROWS[-1]["date"]
    sd = start or months_back(last, per)
    pool = [r for r in ROWS if r["date"] > sd and (end is None or r["date"] <= end)
            and (typ == "A" or r["type"] == typ) and (mkt == "A" or r["mkt"] == mkt) and r["score"] >= min_score]
    grp = {}
    for r in pool:
        grp.setdefault(r["date"] + r["type"], []).append(r)
    rs = []
    for g in grp.values():
        for r in sorted(g, key=lambda x: -x["score"])[:top]:
            c = r.get("c") or []
            if stop and r.get("s") and r["s"] <= N:
                v = r["sr"]
            else:
                v = c[N - 1] if len(c) >= N else None
            if v is not None:
                rs.append({**r, "ret": v - fee})
    if drop:
        rs = sorted(rs, key=lambda x: -x["ret"])[drop:]
    if not rs:
        return None
    by = {}
    for r in rs:
        by.setdefault(r["date"], []).append(r["ret"])
    exits = {}
    for d in sorted(by):
        m = sum(by[d]) / len(by[d]) / 100
        k = bisect.bisect_left(CAL, d) + N
        ed = CAL[k] if k < len(CAL) else d
        exits.setdefault(ed, []).append(m)
    eq, peak, mdd = 1.0, 1.0, 0.0
    for d in sorted(exits):
        for m in exits[d]:
            eq *= 1 + m / N
        peak = max(peak, eq); mdd = min(mdd, eq / peak - 1)
    rets = [r["ret"] for r in rs]
    return {"cum": round((eq - 1) * 100, 1), "n": len(rs), "days": len(by), "win": round(np.mean([x > 0 for x in rets]) * 100),
            "avg": round(float(np.mean(rets)), 2), "med": round(float(np.median(rets)), 2), "mdd": round(mdd * 100, 1)}


def idx_ret(key, d0, d1):
    s = [x for x in BENCH[key] if d0 <= x[0] <= d1]
    return round((s[-1][1] / s[0][1] - 1) * 100, 1) if len(s) > 1 else None


base = dict(typ="P", mkt="Q", min_score=65, top=5, N=5, stop=False)
out = {"last_date": ROWS[-1]["date"], "base": run(**base)}
sd = months_back(ROWS[-1]["date"], 24)
out["kosdaq_same_period"] = idx_ret("kosdaq", sd, ROWS[-1]["date"])
out["kospi_same_period"] = idx_ret("kospi", sd, ROWS[-1]["date"])

# 기간을 나눠서
mid = months_back(ROWS[-1]["date"], 12)
out["halves"] = {"앞 1년": run(**base, start=sd, end=mid), "뒤 1년": run(**base, start=mid)}
out["halves_kosdaq"] = {"앞 1년": idx_ret("kosdaq", sd, mid), "뒤 1년": idx_ret("kosdaq", mid, ROWS[-1]["date"])}
out["by_6m"] = {}
for k in range(4):
    a, b = months_back(ROWS[-1]["date"], 24 - 6 * k), months_back(ROWS[-1]["date"], 18 - 6 * k)
    out["by_6m"][f"{a}~{b}"] = {"strategy": (run(**base, start=a, end=b) or {}).get("cum"), "kosdaq": idx_ret("kosdaq", a, b)}

# 상위 수익 거래를 빼면
out["drop_best"] = {f"상위 {k}건 제외": (run(**base, drop=k) or {}).get("cum") for k in (5, 10, 20, 50)}

# 같은 조건에서 하나씩 바꿔 보기
out["vary_score"] = {s: (run(**{**base, "min_score": s}) or {}).get("cum") for s in (0, 50, 55, 60, 65, 70, 75)}
out["vary_hold"] = {n: (run(**{**base, "N": n}) or {}).get("cum") for n in (1, 2, 3, 4, 5, 6, 7, 10, 15, 20)}
out["vary_top"] = {t: (run(**{**base, "top": t}) or {}).get("cum") for t in (1, 2, 3, 5, 7, 10, 20)}
out["vary_market"] = {m: (run(**{**base, "mkt": m}) or {}).get("cum") for m in ("Q", "K", "A")}
out["vary_type"] = {t: (run(**{**base, "typ": t}) or {}).get("cum") for t in ("P", "B")}
out["with_stop"] = (run(**{**base, "stop": True}) or {}).get("cum")
out["fee_0_5"] = (run(**base, fee=0.5) or {}).get("cum")

# 그리드: 점수 × 보유기간 (코스닥 눌림 상위 5)
out["grid_score_x_hold"] = {s: {n: (run(**{**base, "min_score": s, "N": n}) or {}).get("cum") for n in (3, 5, 7, 10)}
                            for s in (55, 60, 65, 70)}

# 개별 거래 이상치 (데이터 오류 점검)
pool = [r for r in ROWS if r["date"] > sd and r["type"] == "P" and r["mkt"] == "Q" and r["score"] >= 65 and len(r.get("c") or []) >= 5]
ext = sorted(pool, key=lambda r: -r["c"][4])
out["best_trades"] = [[r["date"], r["name"], r["score"], r["c"][4]] for r in ext[:8]]
out["worst_trades"] = [[r["date"], r["name"], r["score"], r["c"][4]] for r in ext[-5:]]

json.dump(out, open(os.path.join(DATA, "check.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out["base"], ensure_ascii=False))
