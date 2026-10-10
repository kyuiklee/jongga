"""매일 자동 실행: 시세 수집 → 오늘 추천 + 과거 추천(백테스트) 계산 → 앱용 JSON 저장"""
import os, sys, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from screen import features, pick_day
import data as D

ROOT = D.ROOT
OUT = os.path.join(ROOT, "site", "data")


def kst_now():
    return pd.Timestamp.now(tz="Asia/Seoul")


def r2(x, n=2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def main():
    t0 = time.time()
    P = json.load(open(os.path.join(ROOT, "engine", "config.json"), encoding="utf-8"))
    now = kst_now()
    print(f"실행 {now:%Y-%m-%d %H:%M} KST")

    univ = D.load_universe()
    meta = univ.set_index("code")
    hist = D.load_histories(list(univ["code"]))
    if len(hist) < 1000:
        raise SystemExit(f"일봉 수집 종목이 너무 적습니다 ({len(hist)}) — 데이터 소스 문제")

    keep_days = int(P.get("backtest_days", 500)) + 5
    feats = []
    for c, h in hist.items():
        if len(h) < 30:
            continue
        f = features(c, h, meta.at[c, "shares"] if c in meta.index else np.nan)
        feats.append(f.iloc[-keep_days:])
    F = pd.concat(feats)
    F = F[F.index >= F.index.max() - pd.Timedelta(days=int(keep_days * 1.5))]
    last = F.index.max()
    # 거래일 = 전체 종목의 절반 이상이 일봉을 가진 날 (일부 종목만 있는 날짜 제외)
    counts = F.groupby(level=0).size()
    days = counts[counts >= len(hist) * 0.5].index
    print(f"지표 계산 완료: {len(days)}거래일, 최종일 {last.date()}")

    # 과거 기록은 하루 상위 20개까지 저장 → 앱에서 시장(코스피/코스닥)별로 걸러도 상위 종목을 고를 수 있게
    P_store = {**P, "top_n": max(P["top_n"], 20)}
    picks = []
    for d, day in F[F.index.isin(days)].groupby(level=0):
        pk = pick_day(day, P_store)
        if len(pk):
            picks.append(pk.assign(date=d))
    T = pd.concat(picks)
    T["name"] = T["code"].map(meta["name"])
    T["market"] = T["code"].map(meta["market"])
    fee = P["fee_pct"]

    # ----- 오늘(최종 거래일) 추천 -----
    trade_day = days.max()
    today_rows = T[(T["date"] == trade_day) & (T["rank"] <= P["top_n"])].sort_values(["type", "rank"])
    is_today = trade_day.date() == now.date()
    hm = now.hour * 100 + now.minute
    status = "final" if (not is_today or hm >= 1530) else "preview"

    idx = D.index_daily()
    market = {}
    for k, s in idx.items():
        s = s[s.index <= trade_day]
        if len(s) >= 2:
            market[k] = r2((s.iloc[-1] / s.iloc[-2] - 1) * 100)

    def card(r):
        d = {"code": r.code, "name": r.name, "market": r.market, "rank": int(r.rank), "score": r2(r.score, 1),
             "close": int(r.close), "chg": r2(r.chg), "amt": int(round(r.amt_eok)), "gap_hi": r2(r.gap_hi, 1),
             "stop": int(r.stop), "tags": r.tags}
        if r.type == "돌파":
            d.update(vol_ratio=r2(r.vol_ratio, 1), run5=r2(r.run5, 1))
        else:
            d.update(surge=r2(r.surge, 0), dd=r2(r.dd, 1))
        return d

    today = {
        "trade_date": f"{trade_day:%Y-%m-%d}",
        "generated_at": f"{now:%Y-%m-%d %H:%M}",
        "status": status,
        "market": market,
        "friday": trade_day.weekday() == 4,
        "breakout": [card(r) for r in today_rows[today_rows["type"] == "돌파"].itertuples()],
        "pullback": [card(r) for r in today_rows[today_rows["type"] == "눌림"].itertuples()],
        "config": {k: P[k] for k in ("top_n", "min_amount_eok", "min_chg", "max_chg", "fee_pct")},
        # 내 전략용: 유형별 상위 20개 후보 (시장 구분 포함)
        "cands": [{**card(r), "type": "B" if r.type == "돌파" else "P", "mkt": "Q" if r.market == "KOSDAQ" else "K"}
                  for r in T[T["date"] == trade_day].sort_values(["type", "rank"]).itertuples()],
    }

    # ----- 과거 추천 + 다음날 결과 (검증 기록·백테스트 공용) -----
    done = T[T["n_open"].notna() & T["date"].isin(days)].copy()
    stop_px = np.where(done["n_open"] <= done["stop"], done["n_open"],
               np.where(done["n_low"] <= done["stop"], done["stop"], done["n_close"]))
    done["r_open"] = (done["n_open"] / done["close"] - 1) * 100
    done["r_close"] = (stop_px / done["close"] - 1) * 100
    done["r_high"] = (done["n_high"] / done["close"] - 1) * 100
    done["stop_hit"] = (done["n_low"] <= done["stop"]).astype(int)
    done = done.sort_values(["date", "type", "rank"])

    # ----- 보유 기간별 결과 (최대 HOLD_MAX 거래일) -----
    # c: 1~N일째 종가 수익률 목록, s: 손절가에 처음 걸린 날(없으면 0), sr: 그때 매도 수익률
    HOLD_MAX = 20
    hn = {}
    for c in done["code"].unique():
        h = hist[c][["Open", "Low", "Close"]].astype(float).copy()
        h.index = pd.to_datetime(h.index).normalize()
        hn[c] = h[~h.index.duplicated(keep="last")].sort_index()
    paths, sdays, srets = [], [], []
    for r in done.itertuples():
        h = hn[r.code]
        pos = h.index.get_indexer([r.date])[0]
        fut = h.iloc[pos + 1: pos + 1 + HOLD_MAX] if pos >= 0 else h.iloc[0:0]
        paths.append([r2((x / r.close - 1) * 100) for x in fut["Close"]])
        s, sr = 0, None
        for j, (o, lo) in enumerate(zip(fut["Open"], fut["Low"]), 1):
            if o <= r.stop:                      # 시가부터 손절가 아래 → 시가에 매도
                s, sr = j, r2((o / r.close - 1) * 100); break
            if lo <= r.stop:                     # 장중 손절가 터치 → 손절가에 매도
                s, sr = j, r2((r.stop / r.close - 1) * 100); break
        sdays.append(s); srets.append(sr)

    trades = {
        "fields": ["date", "type", "rank", "code", "name", "score", "close", "chg", "r_open", "r_close", "r_high", "stop_hit", "mkt",
                   "c", "s", "sr", "stop"],
        "note": "수익률은 비용 차감 전. 앱에서 비용을 뺍니다. c=보유 1~20일째 종가 수익률, s=손절 걸린 날, sr=손절 수익률",
        "rows": [[f"{r.date:%Y-%m-%d}", "B" if r.type == "돌파" else "P", int(r.rank), r.code, r.name, r2(r.score, 1),
                  int(r.close), r2(r.chg), r2(r.r_open), r2(r.r_close), r2(r.r_high), int(r.stop_hit),
                  "Q" if r.market == "KOSDAQ" else "K", p, s, sr, int(r.stop)]
                 for r, p, s, sr in zip(done.itertuples(), paths, sdays, srets)],
    }
    bench = {k.lower(): [[f"{d:%Y-%m-%d}", r2(v)] for d, v in s.items() if d >= days.min()] for k, s in idx.items()}

    # ----- 시장 상황별 성과: 추천일에 해당 시장 지수가 얼마나 움직였는지 구간별로 집계 -----
    ichg = {k: (s.pct_change() * 100) for k, s in idx.items()}
    reg = done[["date", "type", "market", "r_open", "r_close"]].copy()
    reg["d5"] = [p[4] if len(p) >= 5 else np.nan for p in paths]
    reg["idx_chg"] = [ichg.get(m, pd.Series(dtype=float)).get(d, np.nan) for d, m in zip(reg["date"], reg["market"])]
    bins = [-99, -2, -1, 0, 1, 99]
    labels = ["-2% 이하", "-2~-1%", "-1~0%", "0~+1%", "+1% 이상"]
    reg["bucket"] = pd.cut(reg["idx_chg"], bins=bins, labels=labels)
    fee = P["fee_pct"]
    regime = {"note": "추천일 해당 시장(코스피/코스닥) 지수 등락률 구간별. 비용 차감 후. d5=5일째 종가(손절 없음)", "fee": fee, "rows": []}
    for t in ("돌파", "눌림", "전체"):
        g0 = reg if t == "전체" else reg[reg["type"] == t]
        for lb in labels:
            g = g0[g0["bucket"] == lb]
            if not len(g):
                continue
            row = {"type": t, "bucket": lb, "n": int(len(g)), "days": int(g["date"].nunique())}
            for col in ("r_open", "r_close", "d5"):
                v = g[col].dropna() - fee
                row[col + "_win"] = r2((v > 0).mean() * 100, 0) if len(v) else None
                row[col + "_avg"] = r2(v.mean()) if len(v) else None
            regime["rows"].append(row)

    os.makedirs(OUT, exist_ok=True)
    for name, obj in (("today.json", today), ("trades.json", trades), ("bench.json", bench), ("regime.json", regime)):
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    print(f"저장: 오늘 추천 돌파 {len(today['breakout'])} / 눌림 {len(today['pullback'])}, "
          f"과거 거래 {len(trades['rows'])}건, 상태 {status} ({time.time() - t0:.0f}초)")


if __name__ == "__main__":
    main()
