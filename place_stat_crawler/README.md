# 네이버 플레이스 PV 통계 크롤러

네이버 스마트플레이스의 플레이스 페이지 조회수(PV) 통계를 수집하는 크롤링 시스템입니다.

## 📁 프로젝트 구조

```
place_stat_crawler/
├── scripts/
│   ├── crawler/
│   │   ├── naver_place_pv_monthly_crawler.py    # 월별 통계 수집 메인 스크립트
│   │   ├── naver_place_pv_stat_crawler.py       # 일별 통계 수집 스크립트
│   │   ├── naver_place_pv_crawler_base.py       # 크롤러 기본 클래스
│   │   ├── naver_place_pv_auth_manager.py       # 인증 관리
│   │   └── __init__.py
│   ├── util/
│   │   ├── config.py                            # 설정 관리
│   │   └── __init__.py
│   └── main.py                                  # 기타 크롤링 스크립트
├── test/
│   ├── test_naver_place_pv_auth_flow.py         # 인증 플로우 테스트
│   └── account_management.md                    # 계정 관리 가이드
├── data/
│   ├── raw/                                     # 원본 데이터 (JSON)
│   └── processed/                               # 가공 데이터 (CSV, Excel)
├── .env                                         # 환경 변수 설정
├── requirements.txt                             # Python 의존성
└── README.md                                    # 이 파일
```

## 🚀 사용 방법

### 1. 환경 설정

`.env` 파일에 클라이언트 정보를 설정합니다:

```env
# 클라이언트 1
CURSOR_TRACE_NAVER_URL=https://nid.naver.com/nidlogin.login
CURSOR_TRACE_ID=your_id_1
CURSOR_TRACE_PW=your_password_1
CURSOR_TRACE_COOKIE=your_cookie_1
CURSOR_TRACE_AUTH_TOKEN=your_auth_token_1

# 클라이언트 2
GOODMORNINGHANIGURO_NAVER_URL=https://nid.naver.com/nidlogin.login
GOODMORNINGHANIGURO_ID=your_id_2
GOODMORNINGHANIGURO_PW=your_password_2
GOODMORNINGHANIGURO_COOKIE=your_cookie_2
GOODMORNINGHANIGURO_AUTH_TOKEN=your_auth_token_2
```

### 2. 월별 통계 수집

```bash
python scripts/crawler/naver_place_pv_monthly_crawler.py
```

실행 후:
1. 클라이언트 선택 (1-2번)
2. 연월 입력 (예: 2025-07)

### 3. 생성되는 파일

**원본 데이터 (data/raw/):**
- `{CLIENT_NAME}_channel_data_{YYYY}_{MM}.json` - 채널별 데이터
- `{CLIENT_NAME}_keyword_data_{YYYY}_{MM}.json` - 키워드별 데이터
- `{CLIENT_NAME}_total_pv_{YYYY}_{MM}.json` - 일별 총 PV

**가공 데이터 (data/processed/):**
- `{CLIENT_NAME}_{YYYY}_{MM}_integrated_statistics.csv` - 통합 상세 데이터
- `{CLIENT_NAME}_{YYYY}_{MM}_daily_summary.xlsx` - 일자별 요약 엑셀

## 📊 데이터 구조

### 통합 통계 CSV
- `date`: 날짜 (YYYY-MM-DD)
- `data_type`: 데이터 유형 ('channel' 또는 'keyword')
- `name`: 채널명 또는 키워드명
- `pv`: 개별 PV 수
- `total_pv`: 해당 날짜의 총 PV
- 기타 메타데이터 컬럼들

### 일자별 요약 엑셀
- **일자별 요약 시트**: 날짜별 행, 채널/키워드별 열
- **통계 요약 시트**: 기본 통계 및 상위 10일
- **채널별 통계 시트**: 채널별 총 PV, 평균 등
- **키워드별 통계 시트**: 키워드별 총 PV, 평균 등

## 🔧 테스트

인증 플로우 테스트:
```bash
python test/test_naver_place_pv_auth_flow.py
```

## 📝 주요 기능

1. **다중 클라이언트 지원**: 여러 업체의 데이터를 개별 관리
2. **자동 인증 관리**: 토큰/쿠키 자동 갱신 및 저장
3. **월별 데이터 수집**: 지정된 연월의 전체 데이터 수집
4. **통합 데이터 생성**: 채널별, 키워드별 데이터를 하나의 테이블로 통합
5. **정렬된 엑셀 출력**: PV 기준으로 정렬된 채널/키워드 컬럼
6. **다양한 출력 형식**: CSV, Excel, JSON 지원

## ⚠️ 주의사항

- 네이버 스마트플레이스 계정이 필요합니다
- 2FA(2단계 인증)가 설정된 경우 수동으로 토큰/쿠키를 입력해야 할 수 있습니다
- API 호출 제한이 있을 수 있으므로 과도한 요청을 피해주세요

## 🔄 향후 계획

- 예약 통계 수집 기능 추가 예정
- 실시간 모니터링 대시보드
- 자동 리포트 생성
