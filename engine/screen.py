"""검색 조건과 점수 계산.

Colab 노트북 검색기와 같은 공식이며, 과거 날짜에도 그대로 적용해 백테스트에 씁니다.
"""
import numpy as np
import pandas as pd


def features(code, h, shares=np.nan):
    """종목 하나의 일봉으로 날짜별 지표를 한 번에 계산"""
    h = h[["Open", "High", "Low", "Close", "Volume"]].astype(float).copy()
    h.index = pd.to_datetime(h.index).normalize()
    h = h[~h.index.duplicated(keep="last")].sort_index()
    O, H, L, C, V = (h[c] for c in ("Open", "High", "Low", "Close", "Volume"))
    f = pd.DataFrame(index=h.index)
    f["code"] = code
    f["open"], f["high"], f["low"], f["close"], f["volume"] = O, H, L, C, V
    f["chg"] = (C / C.shift(1) - 1) * 100
    rng = (H - L).replace(0, np.nan)
    f["pos"] = ((C - L) / rng).fillna(1.0)
    f["gap_hi"] = (H / C - 1) * 100
    f["body"] = (C / O - 1) * 100
    f["amt_eok"] = C * V / 1e8                      # 거래대금 근사 (종가×거래량)
    f["marcap"] = shares * C if pd.notna(shares) and shares > 0 else np.nan
    # 시총을 모를 때 눌림목 후보 정렬용 대체값 (20일 평균 거래대금)
    f["size"] = f["marcap"].fillna(f["amt_eok"].rolling(20).mean() * 1e8 * 50)
    f["nbar"] = np.arange(1, len(h) + 1)
    ma5, ma20, ma60 = C.rolling(5).mean(), C.rolling(20).mean(), C.rolling(60).mean()
    f["ma5"], f["ma20"], f["ma60"] = ma5, ma20, ma60
    f["prev_hi20"] = H.shift(1).rolling(20).max()
    f["prev_hi60"] = H.shift(1).rolling(60).max()
    f["vol_ratio"] = V / V.shift(1).rolling(20).mean().clip(lower=1)
    f["run5"] = (C / C.shift(5) - 1) * 100
    low20 = L.shift(5).rolling(20).min()
    peak = H.rolling(20).max()
    f["surge"] = (peak / low20 - 1) * 100
    f["dd"] = (C / peak - 1) * 100
    f["vol_dry"] = V.rolling(3).mean() / V.shift(3).rolling(17).max().clip(lower=1)
    f["d5"] = (C / ma5 - 1).abs() * 100
    f["d20"] = (C / ma20 - 1).abs() * 100
    f["n_date"] = pd.Series(h.index, index=h.index).shift(-1)
    for a, b in (("n_open", O), ("n_high", H), ("n_low", L), ("n_close", C)):
        f[a] = b.shift(-1)
    return f


def pick_day(day, P):
    """하루치 전 종목 지표에서 추천 종목 선정 (돌파형 + 눌림목형)"""
    out = []
    day = day[(day["close"] >= P["min_price"]) & (day["volume"] > 0) & (day["high"] > 0)]

    # ---------- 돌파형 ----------
    b = day[day["chg"].between(P["min_chg"], P["max_chg"]) & (day["amt_eok"] >= P["min_amount_eok"]) &
            (day["pos"] >= P["min_pos"]) & (day["gap_hi"] <= P["max_gap_hi"]) & (day["body"] > 0)]
    b = b.sort_values("amt_eok", ascending=False).head(P["max_candidates"])
    b = b[b["nbar"] >= 61]
    brk20, brk60 = b["close"] > b["prev_hi20"], b["close"] > b["prev_hi60"]
    keep = (b["run5"] <= P["max_run5"]) & (b["vol_ratio"] >= P["min_vol_ratio"])
    if P["require_breakout"]:
        keep &= brk20
    b, brk20, brk60 = b[keep], brk20[keep], brk60[keep]
    if len(b):
        aligned = (b["ma5"] > b["ma20"]) & (b["ma20"] > b["ma60"])
        cs = P.get("chg_sweet", 10)          # 등락률 점수 만점 지점 (주식 +10%, ETF는 더 낮게)
        s = ((b["amt_eok"] / 1000).clip(upper=1) * 25 + b["pos"] * 20 +
             np.where(brk60, 15, np.where(brk20, 8, 0)) + np.where(aligned, 10, 0) +
             (b["vol_ratio"] / 5).clip(upper=1) * 15 + (1 - (b["chg"] - cs).abs() / cs).clip(lower=0) * 15)
        if P["stop_mode"] == "당일저가":
            stop = np.minimum(b["low"], b["ma5"]).round()
        else:
            stop = (b["close"] * (1 - P["stop_pct"] / 100)).round()
        tags = []
        for b60, b20, al, vr in zip(brk60, brk20, aligned, b["vol_ratio"]):
            t = ["60일신고가"] if b60 else (["20일신고가"] if b20 else [])
            if al: t.append("정배열")
            if vr >= 3: t.append(f"거래량x{vr:.0f}")
            tags.append(" ".join(t))
        b = b.assign(score=s.round(1), stop=stop, type="돌파", tags=tags)
        b = b.sort_values("score", ascending=False).head(P["top_n"])
        out.append(b.assign(rank=np.arange(1, len(b) + 1)))

    # ---------- 눌림목형 ----------
    if P["use_pullback"]:
        p = day[day["chg"].between(-3, 3) & (day["amt_eok"] >= P["pb_min_amount_eok"])]
        p = p.sort_values("size", ascending=False).head(P["max_candidates"] * 2)
        p = p[p["nbar"] >= 61]
        near = np.minimum(p["d5"], p["d20"])
        keep = ((p["surge"] >= P["pb_min_surge"]) & p["dd"].between(-P["pb_max_dd"], -P.get("pb_min_dd", 3)) &
                (near <= 3) & (p["close"] >= p["ma60"]) & (p["vol_dry"] <= 0.5))
        p, near = p[keep], near[keep]
        if len(p):
            s = ((p["surge"] / 60).clip(upper=1) * 25 + (1 - p["vol_dry"] / 0.5) * 25 + (1 - near / 3) * 20 +
                 p["pos"] * 15 + (p["marcap"] / 1e12).clip(upper=1).fillna(0) * 15)
            use20 = p["d20"] < p["d5"]
            stop = (np.minimum(np.where(use20, p["ma20"], p["ma5"]), p["low"]) * 0.98).round()
            tags = np.where(use20, "20일선지지 거래량감소", "5일선지지 거래량감소")
            p = p.assign(score=s.round(1), stop=stop, type="눌림", tags=tags)
            p = p.sort_values("score", ascending=False).head(P["top_n"])
            out.append(p.assign(rank=np.arange(1, len(p) + 1)))
    return pd.concat(out) if out else pd.DataFrame()
