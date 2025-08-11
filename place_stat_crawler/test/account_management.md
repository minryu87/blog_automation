# 계정 정보 관리 및 네이버 인증 과정

## 🔐 계정 정보 관리 방법

### 1. 환경 변수 방식 (권장)
```bash
export NAVER_ID="natenclinic@naver.com"
export NAVER_PASSWORD="naten2021!"
```

### 2. .env 파일 방식
```bash
# .env 파일 생성
NAVER_ID=natenclinic@naver.com
NAVER_PASSWORD=naten2021!
```

### 3. 코드 내 직접 설정 (테스트용)
```python
NAVER_ID = "natenclinic@naver.com"
NAVER_PASSWORD = "naten2021!"
```

## 🔄 네이버 인증 과정

### 1. Selenium을 통한 자동 로그인

#### 과정:
1. **Chrome 브라우저 시작**
   - 헤드리스 모드로 실행
   - 사용자 에이전트 설정
   - 창 크기 설정 (1920x1080)

2. **네이버 로그인 페이지 접속**
   - URL: `https://nid.naver.com/nidlogin.login`
   - 로그인 폼 요소 대기

3. **계정 정보 입력**
   - ID 필드에 `natenclinic@naver.com` 입력
   - 비밀번호 필드에 `naten2021!` 입력
   - 로그인 버튼 클릭

4. **로그인 성공 확인**
   - 리다이렉트 URL 확인
   - 로그인 상태 검증

### 2. 인증 토큰 및 쿠키 추출

#### 추출되는 정보:
1. **Authorization 헤더**
   - Bearer 토큰 형태
   - API 요청에 사용

2. **세션 쿠키**
   - NID_AUT, NID_SES 등
   - 브라우저 세션 유지용

3. **기타 인증 정보**
   - CSRF 토큰
   - 세션 ID

### 3. 인증 정보 저장

#### 저장 위치:
- 파일: `naver_auth.json`
- 형식: JSON
- 내용:
  ```json
  {
    "auth_token": "Bearer xxx...",
    "cookies": {
      "NID_AUT": "xxx...",
      "NID_SES": "xxx...",
      ...
    },
    "last_refresh_time": "2024-01-15T10:30:00"
  }
  ```

## 🛠️ 인증 매니저 동작 방식

### 1. 초기화
```python
auth_manager = NaverAuthManager(
    naver_id="natenclinic@naver.com",
    naver_password="naten2021!"
)
```

### 2. 인증 정보 로드
```python
# 파일에서 기존 인증 정보 확인
if auth_manager.load_auth_from_file():
    # 기존 토큰이 유효한 경우 사용
    pass
else:
    # 새로 로그인 수행
    auth_manager.login_with_selenium()
```

### 3. 토큰 만료 확인
```python
# 토큰 만료 시간: 12시간
expiry_time = last_refresh + timedelta(hours=12)

if datetime.now() > expiry_time:
    # 토큰 갱신 필요
    auth_manager.refresh_auth_if_needed()
```

### 4. API 요청 헤더 생성
```python
headers = auth_manager.get_auth_headers()
# 결과:
{
    'Authorization': 'Bearer xxx...',
    'Cookie': 'NID_AUT=xxx...; NID_SES=xxx...',
    'User-Agent': 'Mozilla/5.0...',
    'Referer': 'https://new.smartplace.naver.com/'
}
```

## 📊 스마트플레이스 API 호출 과정

### 1. API 엔드포인트
```
https://new.smartplace.naver.com/proxy/bizadvisor/api/v3/sites/sp_311b9ba993e974/report
```

### 2. 요청 헤더
```python
headers = {
    'Authorization': 'Bearer xxx...',
    'Cookie': 'NID_AUT=xxx...; NID_SES=xxx...',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Referer': 'https://new.smartplace.naver.com/',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
    'Cache-Control': 'no-cache',
    'Pragma': 'no-cache'
}
```

### 3. 응답 데이터 구조
```json
{
  "success": true,
  "data": {
    "siteId": "sp_311b9ba993e974",
    "siteName": "나텐클리닉",
    "reports": [
      {
        "date": "2024-01-15",
        "visitors": 1234,
        "pageViews": 5678,
        "reviews": 5,
        "rating": 4.8
      }
    ]
  }
}
```

## 🔍 테스트 실행 방법

### 1. 테스트 스크립트 실행
```bash
cd blog_automation/place_stat_crawler/test
python test_stat_crawler.py
```

### 2. 예상 출력
```
🚀 StatCrawler API 테스트 시작
==================================================

=== 인증 정보 확인 ===
인증 헤더: {'Authorization': 'Bearer xxx...', ...}
쿠키 개수: 8
  NID_AUT: xxx...
  NID_SES: xxx...

=== API 엔드포인트 테스트 ===
API URL: https://new.smartplace.naver.com/proxy/bizadvisor/api/v3/sites/sp_311b9ba993e974/report
응답 상태: 200
응답 본문 길이: 1234
응답 본문 (처음 500자): {"success":true,"data":{...}}

=== StatCrawler 메서드 테스트 ===
사용 가능한 메서드: ['get_smartplace_data', ...]
get_smartplace_data 메서드 존재 확인
get_smartplace_data 결과: {...}

==================================================
📊 테스트 결과 요약
==================================================
auth_info           : ✅ 통과
api_endpoints       : ✅ 통과
stat_crawler_methods: ✅ 통과

전체 결과: 3/3 테스트 통과
🎉 모든 테스트가 성공했습니다!
```

## ⚠️ 주의사항

### 1. 보안
- 계정 정보는 환경 변수나 .env 파일로 관리
- 인증 파일은 .gitignore에 추가
- 정기적으로 비밀번호 변경

### 2. 요청 제한
- 과도한 API 호출 방지
- 적절한 대기 시간 설정
- 429 에러 시 자동 재시도

### 3. 토큰 관리
- 토큰 만료 시간 확인
- 자동 토큰 갱신
- 인증 실패 시 재로그인

## 🔧 문제 해결

### 1. 인증 실패
- 네이버 계정 상태 확인
- 2단계 인증 설정 확인
- 브라우저 캐시 삭제

### 2. API 호출 실패
- 네트워크 연결 확인
- 인증 토큰 유효성 확인
- API 엔드포인트 변경 여부 확인

### 3. 데이터 누락
- 사이트 ID 확인
- 날짜 범위 확인
- 권한 설정 확인
