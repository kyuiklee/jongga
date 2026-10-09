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
    trades = {
        "fields": ["date", "type", "rank", "code", "name", "score", "close", "chg", "r_open", "r_close", "r_high", "stop_hit", "mkt"],
        "note": "수익률은 비용 차감 전. 앱에서 비용을 뺍니다.",
        "rows": [[f"{r.date:%Y-%m-%d}", "B" if r.type == "돌파" else "P", int(r.rank), r.code, r.name, r2(r.score, 1),
                  int(r.close), r2(r.chg), r2(r.r_open), r2(r.r_close), r2(r.r_high), int(r.stop_hit),
                  "Q" if r.market == "KOSDAQ" else "K"]
                 for r in done.itertuples()],
    }
    bench = {k.lower(): [[f"{d:%Y-%m-%d}", r2(v)] for d, v in s.items() if d >= days.min()] for k, s in idx.items()}

    os.makedirs(OUT, exist_ok=True)
    for name, obj in (("today.json", today), ("trades.json", trades), ("bench.json", bench)):
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    print(f"저장: 오늘 추천 돌파 {len(today['breakout'])} / 눌림 {len(today['pullback'])}, "
          f"과거 거래 {len(trades['rows'])}건, 상태 {status} ({time.time() - t0:.0f}초)")


if __name__ == "__main__":
    main()
