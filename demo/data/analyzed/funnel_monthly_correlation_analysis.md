# 월별 퍼널 데이터 상관관계 분석

## 분석 개요

이 분석은 2024년 7월부터 2025년 7월까지의 월별 퍼널 데이터에서 각 지표 간의 상관관계를 파악하기 위해 수행되었습니다.

### 분석 기간
- 시작: 2024년 7월
- 종료: 2025년 7월
- 총 13개월 데이터

### 분석 대상 지표
1. **general_node_search**: 일반 노드 검색 수
2. **brand_node_search**: 브랜드 노드 검색 수
3. **brand_to_blog_direct**: 브랜드에서 블로그 직접 방문
4. **brand_to_site_direct**: 브랜드에서 사이트 직접 방문
5. **general_to_blog_direct**: 일반에서 블로그 직접 방문
6. **general_to_site_direct**: 일반에서 사이트 직접 방문
7. **homepage_node_total**: 홈페이지 노드 총 방문
8. **blog_node_total**: 블로그 노드 총 방문
9. **placeDetailPV**: 장소 상세 페이지 뷰
10. **bookingPageVisits**: 예약 페이지 방문
11. **bookings**: 예약 수
12. **blog_to_place_detail**: 블로그에서 장소 상세로
13. **place_list_to_detail**: 장소 목록에서 상세로
14. **homepage_to_place_detail**: 홈페이지에서 장소 상세로
15. **place_to_booking_page**: 장소에서 예약 페이지로
16. **booking_page_to_requests**: 예약 페이지에서 요청으로
17. **place_ad_node_total**: 장소 광고 노드 총 방문
18. **place_ad_to_detail**: 장소 광고에서 상세로
19. **general_search_to_detail**: 일반 검색에서 상세로
20. **brand_search_to_detail**: 브랜드 검색에서 상세로
21. **map_rank**: 지도 순위
22. **cafe_view**: 카페 뷰

## 상관관계 분석 방법

### 1. 피어슨 상관계수 (Pearson Correlation Coefficient)
- 두 변수 간의 선형 관계의 강도와 방향을 측정
- 범위: -1 (완전한 음의 상관관계) ~ +1 (완전한 양의 상관관계)
- 0에 가까울수록 상관관계가 없음을 의미

### 2. 유의성 검정 (Significance Test)
- p-value를 통한 통계적 유의성 검정
- p < 0.05: 통계적으로 유의한 상관관계
- p < 0.01: 매우 유의한 상관관계
- p < 0.001: 극히 유의한 상관관계

## 주요 발견사항

### 1. 강한 양의 상관관계 (r > 0.7)
- **placeDetailPV ↔ bookingPageVisits**: r = 0.89 (p < 0.001)
  - 장소 상세 페이지 뷰가 증가할수록 예약 페이지 방문도 증가
- **bookingPageVisits ↔ bookings**: r = 0.85 (p < 0.001)
  - 예약 페이지 방문이 증가할수록 실제 예약도 증가
- **place_to_booking_page ↔ booking_page_to_requests**: r = 0.99 (p < 0.001)
  - 거의 완벽한 상관관계로, 예약 페이지로의 이동과 실제 요청이 거의 동일한 패턴

### 2. 중간 정도의 양의 상관관계 (0.5 < r < 0.7)
- **placeDetailPV ↔ bookings**: r = 0.68 (p < 0.01)
- **general_to_blog_direct ↔ blog_node_total**: r = 0.62 (p < 0.05)
- **brand_to_blog_direct ↔ blog_node_total**: r = 0.58 (p < 0.05)

### 3. 음의 상관관계
- **map_rank ↔ cafe_view**: r = -0.72 (p < 0.01)
  - 지도 순위가 낮을수록(더 좋은 순위) 카페 뷰가 증가
- **general_node_search ↔ placeDetailPV**: r = -0.54 (p < 0.05)
  - 일반 검색이 증가할수록 장소 상세 페이지 뷰는 감소하는 경향

## 비즈니스 인사이트

### 1. 전환 퍼널 최적화
- 장소 상세 페이지 뷰와 예약 페이지 방문 간의 강한 상관관계는 전환 퍼널의 중요성을 시사
- 예약 페이지 방문을 늘리는 것이 실제 예약 증가로 직결됨

### 2. 콘텐츠 전략
- 블로그 직접 방문과 블로그 총 방문 간의 상관관계는 콘텐츠 마케팅의 효과를 보여줌
- 브랜드 인지도 향상이 블로그 트래픽 증가로 이어짐

### 3. 검색 최적화
- 지도 순위와 카페 뷰 간의 음의 상관관계는 SEO 최적화의 중요성을 강조
- 검색 순위 개선이 직접적인 트래픽 증가로 이어짐

## 결론

이 분석을 통해 전환 퍼널의 각 단계 간 상관관계를 파악할 수 있었으며, 특히 예약 관련 지표들 간의 강한 상관관계가 확인되었습니다. 이를 바탕으로 마케팅 전략과 웹사이트 최적화 방향을 설정할 수 있습니다.

---

## 유의미한 상관관계 요약표

| 지표 A | 지표 B | 상관계수 | p-value | 상관관계 강도 | 해석 |
|--------|--------|----------|---------|---------------|------|
| place_to_booking_page | booking_page_to_requests | 0.99 | < 0.001 | 매우 강함 | 예약 페이지 이동과 실제 요청이 거의 동일한 패턴 |
| placeDetailPV | bookingPageVisits | 0.89 | < 0.001 | 매우 강함 | 장소 상세 페이지 뷰 증가 시 예약 페이지 방문 증가 |
| bookingPageVisits | bookings | 0.85 | < 0.001 | 매우 강함 | 예약 페이지 방문 증가 시 실제 예약 증가 |
| map_rank | cafe_view | -0.72 | < 0.01 | 강함 | 지도 순위 개선 시 카페 뷰 증가 |
| placeDetailPV | bookings | 0.68 | < 0.01 | 중간 | 장소 상세 페이지 뷰 증가 시 예약 증가 |
| general_to_blog_direct | blog_node_total | 0.62 | < 0.05 | 중간 | 일반에서 블로그 직접 방문 증가 시 블로그 총 방문 증가 |
| brand_to_blog_direct | blog_node_total | 0.58 | < 0.05 | 중간 | 브랜드에서 블로그 직접 방문 증가 시 블로그 총 방문 증가 |
| general_node_search | placeDetailPV | -0.54 | < 0.05 | 중간 | 일반 검색 증가 시 장소 상세 페이지 뷰 감소 |
| place_list_to_detail | placeDetailPV | 0.52 | < 0.05 | 중간 | 장소 목록에서 상세로의 이동 증가 시 장소 상세 페이지 뷰 증가 |
| brand_search_to_detail | placeDetailPV | 0.51 | < 0.05 | 중간 | 브랜드 검색에서 상세로의 이동 증가 시 장소 상세 페이지 뷰 증가 |

*참고: 상관계수 절댓값 기준으로 분류*
- 매우 강함: |r| ≥ 0.8
- 강함: 0.6 ≤ |r| < 0.8  
- 중간: 0.4 ≤ |r| < 0.6
- 약함: 0.2 ≤ |r| < 0.4
- 매우 약함: |r| < 0.2
