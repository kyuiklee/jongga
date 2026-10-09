# 종가매매 검색기

평일 장 마감 무렵 자동으로 전 종목을 검색해서 폰 앱 화면에 보여 줍니다.

- **앱 주소:** https://kyuiklee.github.io/jongga/
- 폰에서 열고 **홈 화면에 추가**하면 앱처럼 쓸 수 있습니다.

## 자동 실행 시간 (평일)
| 시각 (KST) | 내용 |
|---|---|
| 15:12 | 예비 결과 (장중 시세 기준) |
| 15:33 | 확정 결과 |
| 16:05 | 확정 결과 재확인 |

GitHub 예약 실행은 수 분~수십 분 늦게 시작될 수 있습니다. 급할 때는 GitHub 앱 → `jongga` → Actions → **종가매매 검색** → Run workflow 로 바로 실행할 수 있습니다.

## 구성
- `engine/config.json` — 검색 조건 (점수 기준, 거래대금, 손절 방식 등)
- `engine/screen.py` — 조건·점수 계산
- `engine/data.py` — 시세 수집 (네이버 차트 → FinanceDataReader → pykrx)
- `engine/build.py` — 오늘 추천 + 과거 추천 결과(백테스트) 계산
- `site/` — 앱 화면

검색 결과는 조건 필터링이며 매수 권유가 아닙니다.
