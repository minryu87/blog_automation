import React, { useState, useEffect, useCallback } from 'react';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, Legend, ComposedChart } from 'recharts';
import FlowGraph from './FlowGraph.jsx';
import { ChevronRight, TrendingUp, Users, MousePointer, Search, Map, FileText, DollarSign, Target, Activity, Award, Settings, Sliders, BookOpen, MessageCircle, Globe } from 'lucide-react';

const HospitalMarketingDashboard = () => {
  // 초기 기준값
  const baseMetrics = {
    // 최상위 수요
    nonBrandDemand: 3000,
    brandDemand: 800,
    
    // 일반 키워드 경로
    nonBrand_to_search: 1800,
    nonBrand_to_map: 1200,
    
    // 검색 → 인지 경로
    search_to_blog: 700,
    search_to_site: 500,
    search_to_cafe: 600,
    
    // 인지 → 브랜드 전환
    blog_to_brand: 210,
    site_to_brand: 150,
    cafe_to_brand: 240,
    
    // 지도 경로
    map_to_list: 800,
    list_to_placeAds: 180,
    list_to_placeDetail: 400,
    ads_to_placeDetail: 160,
    
    // 브랜드 키워드 경로
    brand_to_search_direct: 320,
    brand_to_map_direct: 240,
    brand_to_blog_direct: 120,
    brand_to_site_direct: 120,
    
    // 플레이스 상세 집계
    placeDetailPV: 2400,
    
    // 예약 퍼널
    bookingPageVisits: 600,
    bookings: 120,
    
    // 전환율
    cvr_detail_to_booking: 0.25,
    cvr_booking_to_complete: 0.20
  };

  const [metrics, setMetrics] = useState(baseMetrics);
  const [targetBookings, setTargetBookings] = useState(baseMetrics.bookings);
  // 실데이터 연동 상태
  const [monthlyData, setMonthlyData] = useState([]);
  const [selectedMonthIndex, setSelectedMonthIndex] = useState(0);
  const [startIndex, setStartIndex] = useState(0);
  const [isMonthlyMode, setIsMonthlyMode] = useState(true);
  const [highlightedNodeIds, setHighlightedNodeIds] = useState([]);
  
  // 인지 단계 분석을 위한 상태
  const [selectedKeywords, setSelectedKeywords] = useState({
    '동탄 치과': true,
    '동탄 임플란트': false,
    '동탄 신경치료': false
  });
  
  const [awarenessSliders, setAwarenessSliders] = useState({
    contentVolume: 100,
    viralBudget: 100
  });

  // 시계열 데이터 생성 (2024년 7월 ~ 2025년 8월)
  const generateTimeSeriesData = () => {
    const baseData = [
      { month: '2024-07', blog: 320, site: 280, cafe: 180, brandSearch: 450 },
      { month: '2024-08', blog: 350, site: 310, cafe: 200, brandSearch: 480 },
      { month: '2024-09', blog: 380, site: 320, cafe: 220, brandSearch: 520 },
      { month: '2024-10', blog: 410, site: 350, cafe: 250, brandSearch: 580 },
      { month: '2024-11', blog: 450, site: 380, cafe: 280, brandSearch: 650 },
      { month: '2024-12', blog: 480, site: 400, cafe: 310, brandSearch: 720 },
      { month: '2025-01', blog: 520, site: 430, cafe: 340, brandSearch: 790 },
      { month: '2025-02', blog: 560, site: 460, cafe: 370, brandSearch: 860 },
      { month: '2025-03', blog: 600, site: 490, cafe: 400, brandSearch: 930 },
      { month: '2025-04', blog: 640, site: 520, cafe: 430, brandSearch: 1000 },
      { month: '2025-05', blog: 680, site: 550, cafe: 460, brandSearch: 1070 },
      { month: '2025-06', blog: 720, site: 580, cafe: 490, brandSearch: 1140 },
      { month: '2025-07', blog: 760, site: 610, cafe: 520, brandSearch: 1210 }
    ];

    // 2025년 8월 예측 데이터 (슬라이더에 따라 조정)
    const august2025 = {
      month: '2025-08',
      blog: Math.round(800 * awarenessSliders.contentVolume / 100),
      site: Math.round(640 * awarenessSliders.contentVolume / 100),
      cafe: Math.round(550 * awarenessSliders.viralBudget / 100),
      brandSearch: Math.round(1280 * ((awarenessSliders.contentVolume + awarenessSliders.viralBudget) / 200)),
      isPrediction: true
    };

    return [...baseData, august2025];
  };

  const timeSeriesData = generateTimeSeriesData();
  
  // 각 구간별 조정 슬라이더 값 (100 = 기준값)
  const [sliders, setSliders] = useState({
    main: 100,
    nonBrand: 100,
    brand: 100,
    brandConversion: 100,
    mapRanking: 100,
    placeAds: 100,
    detailConversion: 100
  });

  // 슬라이더 조정 처리
  const handleSliderChange = (sliderName, value) => {
    setSliders(prev => {
      const newSliders = { ...prev, [sliderName]: value };
      
      // 메인 슬라이더가 변경되면 다른 모든 슬라이더를 비례 조정
      if (sliderName === 'main') {
        const ratio = value / 100;
        Object.keys(newSliders).forEach(key => {
          if (key !== 'main') {
            newSliders[key] = Math.round(100 * ratio);
          }
        });
      }
      
      return newSliders;
    });
  };

  // 메트릭 재계산
  useEffect(() => {
    const calculateMetrics = () => {
      const s = sliders;
      
      // 수요 조정
      const nonBrandDemand = Math.round(baseMetrics.nonBrandDemand * s.nonBrand / 100);
      const brandDemandBase = Math.round(baseMetrics.brandDemand * s.brand / 100);
      
      // 일반 키워드 분배
      const nonBrand_to_search = Math.round(nonBrandDemand * 0.6);
      const nonBrand_to_map = Math.round(nonBrandDemand * 0.4);
      
      // 검색 → 인지 경로
      const search_to_blog = Math.round(nonBrand_to_search * 0.39);
      const search_to_site = Math.round(nonBrand_to_search * 0.28);
      const search_to_cafe = Math.round(nonBrand_to_search * 0.33);
      
      // 브랜드 전환 (brandConversion 슬라이더 영향)
      const convRate = 0.3 * s.brandConversion / 100;
      const blog_to_brand = Math.round(search_to_blog * convRate);
      const site_to_brand = Math.round(search_to_site * convRate);
      const cafe_to_brand = Math.round(search_to_cafe * convRate);
      
      // 전환된 브랜드 수요 추가
      const totalBrandDemand = brandDemandBase + blog_to_brand + site_to_brand + cafe_to_brand;
      
      // 지도 경로 (mapRanking 슬라이더 영향)
      const map_to_list = Math.round(nonBrand_to_map * 0.67 * s.mapRanking / 100);
      const list_to_placeAds = Math.round(map_to_list * 0.23 * s.placeAds / 100);
      const list_to_placeDetail = Math.round(map_to_list * 0.5);
      const ads_to_placeDetail = Math.round(list_to_placeAds * 0.89);
      
      // 브랜드 키워드 분배
      const brand_to_search_direct = Math.round(totalBrandDemand * 0.4);
      const brand_to_map_direct = Math.round(totalBrandDemand * 0.3);
      const brand_to_blog_direct = Math.round(totalBrandDemand * 0.15);
      const brand_to_site_direct = Math.round(totalBrandDemand * 0.15);
      
      // 플레이스 상세 집계
      const placeDetailPV = list_to_placeDetail + ads_to_placeDetail +
                           brand_to_search_direct + brand_to_map_direct + 
                           brand_to_blog_direct + brand_to_site_direct;
      
      // 예약 퍼널 (detailConversion 슬라이더 영향)
      const bookingPageVisits = Math.round(placeDetailPV * 0.25 * s.detailConversion / 100);
      const bookings = Math.round(bookingPageVisits * 0.2);
      
      setMetrics({
        ...baseMetrics,
        nonBrandDemand,
        brandDemand: totalBrandDemand,
        nonBrand_to_search,
        nonBrand_to_map,
        search_to_blog,
        search_to_site,
        search_to_cafe,
        blog_to_brand,
        site_to_brand,
        cafe_to_brand,
        map_to_list,
        list_to_placeAds,
        list_to_placeDetail,
        ads_to_placeDetail,
        brand_to_search_direct,
        brand_to_map_direct,
        brand_to_blog_direct,
        brand_to_site_direct,
        placeDetailPV,
        bookingPageVisits,
        bookings
      });
      
      setTargetBookings(bookings);
    };
    
    if (!isMonthlyMode) {
      calculateMetrics();
    }
  }, [sliders, isMonthlyMode]);

  // 실데이터 로드
  useEffect(() => {
    const load = async () => {
      try {
        const res = await fetch(`/data/service/funnel_monthly.json?ts=${Date.now()}`);
        if (!res.ok) return;
        const data = await res.json();
        if (Array.isArray(data) && data.length > 0) {
          setMonthlyData(data);
          // 슬라이더 시작을 2024-09로 설정
          const sIdx = Math.max(0, data.findIndex((m) => m.month === '2024-09'));
          setStartIndex(sIdx === -1 ? 0 : sIdx);
          setSelectedMonthIndex(data.length - 1);
        }
      } catch (e) {
        // ignore
      }
    };
    load();
  }, []);

  // 실데이터 선택 반영
  useEffect(() => {
    if (!isMonthlyMode) return;
    if (!monthlyData || monthlyData.length === 0) return;
    const m = monthlyData[selectedMonthIndex];
    if (!m) return;
    const nonBrandDemand = Math.max(0, (m.nonBrand_to_search || 0) + (m.nonBrand_to_map || 0));
    const brandDemand = Math.max(0, (m.brand_to_search_direct || 0) + (m.brand_to_map_direct || 0) + (m.brand_to_blog_direct || 0) + (m.brand_to_site_direct || 0));
    setMetrics({
      ...baseMetrics,
      brand_node_search: m.brand_node_search || 0,
      homepage_node_total: m.homepage_node_total || 0,
      blog_node_total: m.blog_node_total || 0,
      homepage_to_place_detail: m.homepage_to_place_detail || 0,
      blog_to_place_detail: m.blog_to_place_detail || 0,
      homepage_to_booking_page_direct: m.homepage_to_booking_page_direct || 0,
      general_node_search: m.general_node_search || 0,
      general_to_site_direct: m.general_to_site_direct || 0,
      general_to_blog_direct: m.general_to_blog_direct || 0,
      general_to_map_direct: m.general_to_map_direct || 0,
      map_node_total: m.map_node_total || 0,
      map_rank: m.map_rank || 0,
      nonBrandDemand,
      brandDemand,
      nonBrand_to_search: m.nonBrand_to_search || 0,
      nonBrand_to_map: m.nonBrand_to_map || 0,
      search_to_blog: m.search_to_blog || 0,
      search_to_site: m.search_to_site || 0,
      search_to_cafe: m.search_to_cafe || 0,
      blog_to_brand: m.blog_to_brand || 0,
      site_to_brand: m.site_to_brand || 0,
      cafe_to_brand: m.cafe_to_brand || 0,
      map_to_list: m.map_to_list || 0,
      list_to_placeAds: m.list_to_placeAds || 0,
      list_to_placeDetail: m.list_to_placeDetail || 0,
      ads_to_placeDetail: m.ads_to_placeDetail || 0,
      brand_to_search_direct: m.brand_to_search_direct || 0,
      brand_to_map_direct: m.brand_to_map_direct || 0,
      brand_to_blog_direct: m.brand_to_blog_direct || 0,
      brand_to_site_direct: m.brand_to_site_direct || 0,
      placeDetailPV: m.placeDetailPV || 0,
      bookingPageVisits: m.bookingPageVisits || 0,
      bookings: m.bookings || 0,
      place_to_booking_page: m.place_to_booking_page || 0,
      booking_page_to_requests: m.booking_page_to_requests || 0,
      place_list_to_detail: m.place_list_to_detail || 0,
      place_ad_node_total: m.place_ad_node_total || 0,
      place_ad_to_detail: m.place_ad_to_detail || 0,
      general_search_to_detail: m.general_search_to_detail || 0,
      brand_search_to_detail: m.brand_search_to_detail || 0,
      cafe_view: m.cafe_view || 0
    });
    setTargetBookings(m.bookings || 0);
  }, [isMonthlyMode, monthlyData, selectedMonthIndex]);

  // 액션 아이템 계산
  const actions = {
    cafeShare: Math.round((metrics.cafe_to_brand - baseMetrics.cafe_to_brand) / 8),
    blogPosts: Math.round((metrics.search_to_blog - baseMetrics.search_to_blog) / 15),
    siteOptimization: Math.round((metrics.search_to_site - baseMetrics.search_to_site) / 10),
    reviewCount: Math.round((metrics.map_to_list - baseMetrics.map_to_list) / 20),
    adsBudget: Math.round((metrics.list_to_placeAds - baseMetrics.list_to_placeAds) * 85),
    conversionOptimization: Math.round((metrics.bookingPageVisits - baseMetrics.bookingPageVisits) / 5)
  };

  const formatPercent = (current, target) => {
    if (!current || current === 0) return '0%';
    const change = ((target / current - 1) * 100).toFixed(1);
    if (change > 0) return `+${change}%`;
    return `${change}%`;
  };

  // 인지 단계 분석 컴포넌트
  const AwarenessAnalysis = () => {
    return (
      <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-800">인지 단계 분석 - 내이튼치과</h2>
          <div className="flex gap-2">
            {Object.keys(selectedKeywords).map(keyword => (
              <button
                key={keyword}
                onClick={() => setSelectedKeywords(prev => ({...prev, [keyword]: !prev[keyword]}))}
                className={`px-3 py-1 text-xs rounded-full transition-colors ${
                  selectedKeywords[keyword] 
                    ? 'bg-blue-500 text-white' 
                    : 'bg-gray-200 text-gray-600'
                }`}
              >
                {keyword}
              </button>
            ))}
          </div>
        </div>

        <ResponsiveContainer width="100%" height={300}>
          <ComposedChart data={timeSeriesData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis 
              dataKey="month" 
              tick={{ fontSize: 10 }}
              angle={-45}
              textAnchor="end"
              height={60}
            />
            <YAxis yAxisId="left" tick={{ fontSize: 11 }} />
            <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 11 }} />
            <Tooltip />
            <Legend />
            
            <Bar 
              yAxisId="right"
              dataKey="brandSearch" 
              fill="#E3F2FD" 
              name="내이튼치과 검색량"
              opacity={0.7}
            />
            
            <Line 
              yAxisId="left"
              type="monotone" 
              dataKey="blog" 
              stroke="#FF5722" 
              strokeWidth={2}
              name="블로그 유입"
              strokeDasharray={(data) => data.isPrediction ? "5 5" : "0"}
              dot={{ r: 3 }}
            />
            <Line 
              yAxisId="left"
              type="monotone" 
              dataKey="site" 
              stroke="#3F51B5" 
              strokeWidth={2}
              name="홈페이지 유입"
              strokeDasharray={(data) => data.isPrediction ? "5 5" : "0"}
              dot={{ r: 3 }}
            />
            <Line 
              yAxisId="left"
              type="monotone" 
              dataKey="cafe" 
              stroke="#E91E63" 
              strokeWidth={2}
              name="카페 언급 조회수"
              strokeDasharray={(data) => data.isPrediction ? "5 5" : "0"}
              dot={{ r: 3 }}
            />
          </ComposedChart>
        </ResponsiveContainer>

        <div className="mt-6 space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="p-4 bg-gray-50 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-gray-700">
                  📝 메디컨텐츠 발행량
                </span>
                <span className="text-sm font-bold text-blue-600">{awarenessSliders.contentVolume}%</span>
              </div>
              <input
                type="range"
                min="50"
                max="150"
                value={awarenessSliders.contentVolume}
                onChange={(e) => setAwarenessSliders(prev => ({...prev, contentVolume: Number(e.target.value)}))}
                className="w-full h-2 bg-blue-200 rounded-lg appearance-none cursor-pointer"
              />
              <div className="text-xs text-gray-500 mt-2">
                블로그 및 홈페이지 유입량에 영향
              </div>
            </div>

            <div className="p-4 bg-gray-50 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-gray-700">
                  💰 바이럴 마케팅 비용
                </span>
                <span className="text-sm font-bold text-purple-600">{awarenessSliders.viralBudget}%</span>
              </div>
              <input
                type="range"
                min="50"
                max="150"
                value={awarenessSliders.viralBudget}
                onChange={(e) => setAwarenessSliders(prev => ({...prev, viralBudget: Number(e.target.value)}))}
                className="w-full h-2 bg-purple-200 rounded-lg appearance-none cursor-pointer"
              />
              <div className="text-xs text-gray-500 mt-2">
                카페 언급 및 조회수에 영향
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  // 개선된 퍼널 플로우 시각화 (Sankey 스타일) - 높이 축소
  const ImprovedFunnelFlow = () => {
    // 연결선 굵기 계산 함수
    const getStrokeWidth = (value, maxValue = 2000) => {
      const minWidth = 2;
      const maxWidth = 30;
      return Math.max(minWidth, Math.min(maxWidth, (value / maxValue) * maxWidth));
    };

    // 원형 노드 반경 계산 함수 (노드 값에 따라 크기 다르게)
    const getNodeRadius = (value, maxValue = 2000) => {
      const minR = 18;
      const maxR = 46;
      if (!value || value <= 0) return minR;
      return Math.max(minR, Math.min(maxR, (value / maxValue) * maxR));
    };

    // 곡선 경로 생성 함수
    const createPath = (x1, y1, x2, y2, width) => {
      const midX = (x1 + x2) / 2;
      return `M ${x1} ${y1 - width/2} 
              Q ${midX} ${y1 - width/2} ${x2} ${y2 - width/2}
              L ${x2} ${y2 + width/2}
              Q ${midX} ${y1 + width/2} ${x1} ${y1 + width/2}
              Z`;
    };

    // 수평 경로 생성 함수
    const createHorizontalPath = (x1, y1, x2, y2, width) => {
      return `M ${x1} ${y1 - width/2}
              L ${x2} ${y2 - width/2}
              L ${x2} ${y2 + width/2}
              L ${x1} ${y1 + width/2}
              Z`;
    };

    return (
      <div className="h-[400px] bg-gradient-to-br from-gray-50 to-gray-100 rounded-lg p-4 overflow-x-auto overflow-y-hidden flex items-center justify-center">
        <svg className="h-full" style={{ width: '1200px' }} viewBox="0 0 1200 400" preserveAspectRatio="xMidYMid meet">
          {/* 배경 그리드 */}
          <defs>
            <pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse">
              <path d="M 50 0 L 0 0 0 50" fill="none" stroke="#e5e7eb" strokeWidth="0.5" opacity="0.3"/>
            </pattern>
            
            {/* 그라데이션 정의 */}
            <linearGradient id="brandGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#FF9800" opacity="0.8"/>
              <stop offset="100%" stopColor="#FF9800" opacity="0.3"/>
            </linearGradient>
            
            <linearGradient id="nonBrandGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#FBC02D" opacity="0.8"/>
              <stop offset="100%" stopColor="#FBC02D" opacity="0.3"/>
            </linearGradient>
            
            <linearGradient id="searchGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#2196F3" opacity="0.7"/>
              <stop offset="100%" stopColor="#2196F3" opacity="0.3"/>
            </linearGradient>
            
            <linearGradient id="mapGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#4CAF50" opacity="0.7"/>
              <stop offset="100%" stopColor="#4CAF50" opacity="0.3"/>
            </linearGradient>
            
            <linearGradient id="placeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#9C27B0" opacity="0.7"/>
              <stop offset="100%" stopColor="#9C27B0" opacity="0.3"/>
            </linearGradient>
          </defs>
          
          <rect width="1200" height="400" fill="url(#grid)" />

          {/* 노드 박스들 - 중앙 정렬을 위해 x 좌표 조정 */}
          {/* 시작점: 브랜드 키워드 네이버 검색량 (원형 노드) */}
          <g>
            {(() => {
              const cx = 120; const cy = 90;
              const r = getNodeRadius(metrics.brand_node_search || 0);
              return (
                <>
                  <circle cx={cx} cy={cy} r={r} fill="#FFF3E0" stroke="#FF9800" strokeWidth="2" />
                  <text x={cx} y={cy - r - 10} textAnchor="middle" fontSize="10" fontWeight="600" fill="#FF9800">브랜드 키워드 네이버 검색량</text>
                  <text x={cx} y={cy + 4} textAnchor="middle" fontSize="14" fontWeight="bold" fill="#FF9800">{metrics.brand_node_search || 0}</text>
                </>
              );
            })()}
          </g>

          {/* 일반 키워드 네이버 검색량 (원형 노드) */}
          <g>
            {(() => {
              const cx = 120; const cy = 300;
              const r = getNodeRadius(metrics.general_node_search || 0, 700000);
              return (
                <>
                  <circle cx={cx} cy={cy} r={r} fill="#FFFDE7" stroke="#FBC02D" strokeWidth="2" />
                  <text x={cx} y={cy - r - 10} textAnchor="middle" fontSize="10" fontWeight="600" fill="#FBC02D">일반 키워드 네이버 검색량</text>
                  <text x={cx} y={cy + 4} textAnchor="middle" fontSize="14" fontWeight="bold" fill="#FBC02D">{metrics.general_node_search || 0}</text>
                </>
              );
            })()}
          </g>

          {/* 채널: 네이버 검색 노드는 제거 (비주얼 표시 없음). 좌표는 기존 엣지 출발점으로만 사용 */}

          {/* 네이버 지도(플레이스 목록) - 값 대신 순위 표시 */}
          <g>
            {(() => {
              const cx = 255; const cy = 285;
              const r = 22; // 고정 반경으로 단순 표시
              return (
                <>
                  <circle cx={cx} cy={cy} r={r} fill="#FFFFFF" stroke="#4CAF50" strokeWidth="2" />
                  <text x={cx} y={cy - r - 8} textAnchor="middle" fontSize="9" fontWeight="600" fill="#4CAF50">네이버 지도(플레이스 목록)</text>
                  <text x={cx} y={cy + 2} textAnchor="middle" fontSize="10" fontWeight="600" fill="#4CAF50">순위: {metrics.map_rank || 0}</text>
                  <text x={cx} y={cy + 16} textAnchor="middle" fontSize="8" fill="#4CAF50">('동탄치과'에 대한 순위)</text>
                </>
              );
            })()}
          </g>

          {/* 인지 채널들 (원형 노드) */}
          <g>
            {(() => {
              const cx = 382; const cy = 50;
              const r = getNodeRadius(metrics.homepage_node_total || 0);
              return (
                <>
                  <circle cx={cx} cy={cy} r={r} fill="#FFFFFF" stroke="#3F51B5" strokeWidth="2" />
                  <text x={cx} y={cy - r - 8} textAnchor="middle" fontSize="8" fontWeight="600" fill="#3F51B5">홈페이지</text>
                  <text x={cx} y={cy + 4} textAnchor="middle" fontSize="12" fontWeight="bold" fill="#3F51B5">{metrics.homepage_node_total || 0}</text>
                </>
              );
            })()}
          </g>

          <g>
            {(() => {
              const cx = 382; const cy = 110;
              const r = getNodeRadius(metrics.blog_node_total || 0);
              return (
                <>
                  <circle cx={cx} cy={cy} r={r} fill="#FFFFFF" stroke="#FF5722" strokeWidth="2" />
                  <text x={cx} y={cy - r - 8} textAnchor="middle" fontSize="8" fontWeight="600" fill="#FF5722">네이버 블로그</text>
                  <text x={cx} y={cy + 4} textAnchor="middle" fontSize="12" fontWeight="bold" fill="#FF5722">{metrics.blog_node_total || 0}</text>
                </>
              );
            })()}
          </g>

          {/* 지역 카페 조회수 (원형 노드) */}
          <g>
            {(() => {
              const cx = 300; const cy = 150;
              const r = getNodeRadius(metrics.cafe_view || 0, 4000);
              return (
                <>
                  <circle cx={cx} cy={cy} r={r} fill="#FFFFFF" stroke="#E91E63" strokeWidth="2" />
                  <text x={cx} y={cy - r - 8} textAnchor="middle" fontSize="8" fontWeight="600" fill="#E91E63">지역 카페 조회수</text>
                  <text x={cx} y={cy + 4} textAnchor="middle" fontSize="12" fontWeight="bold" fill="#E91E63">{metrics.cafe_view || 0}</text>
                </>
              );
            })()}
          </g>

          {/* (삭제됨) 플레이스 목록 노드 */}

          {/* 플레이스 광고 (원형: 노출수) */}
          <g>
            {(() => {
              const cx = 382; const cy = 320;
              const r = getNodeRadius(metrics.place_ad_node_total || 0, 10000);
              return (
                <>
                  <circle cx={cx} cy={cy} r={r} fill="#FFFFFF" stroke="#FFC107" strokeWidth="2" />
                  <text x={cx} y={cy - r - 8} textAnchor="middle" fontSize="8" fontWeight="600" fill="#FFC107">플레이스 광고</text>
                  <text x={cx} y={cy + 4} textAnchor="middle" fontSize="11" fontWeight="bold" fill="#FFC107">{metrics.place_ad_node_total || 0}</text>
                </>
              );
            })()}
          </g>

          {/* 전환 지점들 (원형 노드) */}
          <g>
            {(() => {
              const cx = 520; const cy = 172;
              const r = getNodeRadius(metrics.placeDetailPV || 0, 4000);
              return (
                <>
                  <circle cx={cx} cy={cy} r={r} fill="#FDF7FF" stroke="#9C27B0" strokeWidth="2" />
                  <text x={cx} y={cy - r - 8} textAnchor="middle" fontSize="9" fontWeight="600" fill="#9C27B0">플레이스상세</text>
                  <text x={cx} y={cy + 4} textAnchor="middle" fontSize="16" fontWeight="bold" fill="#9C27B0">{metrics.placeDetailPV}</text>
                </>
              );
            })()}
          </g>

          <g>
            {(() => {
              const cx = 675; const cy = 172;
              const r = getNodeRadius(metrics.bookingPageVisits || 0, 2000);
              return (
                <>
                  <circle cx={cx} cy={cy} r={r} fill="#EFFFFE" stroke="#03A9F4" strokeWidth="2" />
                  <text x={cx} y={cy - r - 8} textAnchor="middle" fontSize="9" fontWeight="600" fill="#03A9F4">네이버 예약 페이지(session)</text>
                  <text x={cx} y={cy + 4} textAnchor="middle" fontSize="16" fontWeight="bold" fill="#03A9F4">{metrics.bookingPageVisits}</text>
                </>
              );
            })()}
          </g>

          <g>
            {(() => {
              const cx = 810; const cy = 172;
              const r = getNodeRadius(metrics.bookings || 0, 1000);
              return (
                <>
                  <circle cx={cx} cy={cy} r={r} fill="#E8F5E9" stroke="#4CAF50" strokeWidth="3" />
                  <text x={cx} y={cy - r - 8} textAnchor="middle" fontSize="9" fontWeight="600" fill="#4CAF50">예약 신청 (UV)</text>
                  <text x={cx} y={cy + 5} textAnchor="middle" fontSize="18" fontWeight="bold" fill="#4CAF50">{metrics.bookings}</text>
                </>
              );
            })()}
          </g>

          {/* 플로우 연결선들 (Sankey 스타일) - x 좌표 조정 */}
          
          {/* 브랜드 → 홈페이지 (브랜드 유입) */}
          <path
            d={createPath(120, 90, 382, 50, getStrokeWidth(metrics.brand_to_site_direct || 0))}
            fill="url(#brandGradient)"
            opacity="0.7"
          />
          <text x="250" y="65" textAnchor="middle" fontSize="8" fill="#FF9800" fontWeight="600">
            {metrics.brand_to_site_direct || 0}
          </text>

          {/* 브랜드 → 블로그 (브랜드 유입) */}
          <path
            d={createPath(120, 90, 382, 110, getStrokeWidth(metrics.brand_to_blog_direct || 0))}
            fill="url(#brandGradient)"
            opacity="0.7"
          />
          <text x="250" y="105" textAnchor="middle" fontSize="8" fill="#FF9800" fontWeight="600">
            {metrics.brand_to_blog_direct || 0}
          </text>

          {/* 일반 → 네이버 지도 엣지 제거 */}
          {/* 일반 → 홈페이지 */}
          <path
            d={createPath(120, 300, 382, 50, getStrokeWidth(metrics.general_to_site_direct || 0))}
            fill="url(#nonBrandGradient)"
            opacity="0.7"
          />
          <text x="250" y="175" textAnchor="middle" fontSize="8" fill="#FBC02D" fontWeight="600">
            {metrics.general_to_site_direct || 0}
          </text>

          {/* 일반 → 네이버 블로그 */}
          <path
            d={createPath(120, 300, 382, 110, getStrokeWidth(metrics.general_to_blog_direct || 0))}
            fill="url(#nonBrandGradient)"
            opacity="0.7"
          />
          <text x="250" y="205" textAnchor="middle" fontSize="8" fill="#FBC02D" fontWeight="600">
            {metrics.general_to_blog_direct || 0}
          </text>

          {/* 일반 → 지역 카페 조회수 (화살표만, 값 라벨 없음) */}
          <defs>
            <marker id="arrowHeadSmall" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto" markerUnits="strokeWidth">
              <path d="M0,0 L0,6 L6,3 z" fill="#E91E63" />
            </marker>
          </defs>
          <path
            d={createPath(120, 300, 300, 150, 6)}
            fill="#E91E63"
            opacity="0.35"
            markerEnd="url(#arrowHeadSmall)"
          />

          {/* 지역 카페 조회수 → 브랜드 검색 (화살표만) */}
          <path
            d={createPath(300, 150, 120, 90, 6)}
            fill="#E91E63"
            opacity="0.35"
            markerEnd="url(#arrowHeadSmall)"
          />

          {/* 검색/카페 파생 엣지 제거 */}

          {/* 네이버 지도 → 플레이스 목록 엣지 제거 */}

          {/* 플레이스 목록 → 플레이스 광고 엣지 제거 */}

          {/* 홈페이지 → 플레이스 상세 (실데이터) */}
          <path 
            d={createPath(415, 50, 480, 160, getStrokeWidth(metrics.homepage_to_place_detail || 0))}
            fill="#3F51B5"
            opacity="0.6"
          />
          <text x="447" y="100" textAnchor="middle" fontSize="8" fill="#3F51B5" fontWeight="600">
            {metrics.homepage_to_place_detail || 0}
          </text>

          {/* 네이버 블로그 → 플레이스 상세 (실데이터) */}
          <path 
            d={createPath(415, 110, 480, 165, getStrokeWidth(metrics.blog_to_place_detail || 0))}
            fill="#FF5722"
            opacity="0.6"
          />
          <text x="447" y="135" textAnchor="middle" fontSize="8" fill="#FF5722" fontWeight="600">
            {metrics.blog_to_place_detail || 0}
          </text>

          {/* 브랜드 검색 → 플레이스 상세 */}
          <path
            d={createPath(120, 90, 480, 160, getStrokeWidth(metrics.brand_search_to_detail || 0))}
            fill="#FF9800"
            opacity="0.35"
          />
          <text x="300" y="140" textAnchor="middle" fontSize="8" fill="#FF9800" fontWeight="600">
            {metrics.brand_search_to_detail || 0}
          </text>

          {/* 일반 검색 → 플레이스 상세 */}
          <path
            d={createPath(120, 300, 480, 175, getStrokeWidth(metrics.general_search_to_detail || 0))}
            fill="#FBC02D"
            opacity="0.35"
          />
          <text x="300" y="200" textAnchor="middle" fontSize="8" fill="#FBC02D" fontWeight="600">
            {metrics.general_search_to_detail || 0}
          </text>
          

          {/* 네이버 지도(플레이스 목록) → 플레이스상세 (네이버지도 pv) */}
          <path 
            d={createPath(255, 285, 520, 172, getStrokeWidth(metrics.place_list_to_detail || 0))}
            fill="#00BCD4"
            opacity="0.5"
          />
          <text x="390" y="230" textAnchor="middle" fontSize="8" fill="#00BCD4" fontWeight="600">
            {metrics.place_list_to_detail || 0}
          </text>

          {/* 플레이스 광고 → 플레이스 상세 (클릭수) */}
          <path 
            d={createPath(415, 340, 480, 180, getStrokeWidth(metrics.place_ad_to_detail || 0))}
            fill="#FFC107"
            opacity="0.5"
          />
          <text x="447" y="300" textAnchor="middle" fontSize="8" fill="#FFC107" fontWeight="600">
            {metrics.place_ad_to_detail || 0}
          </text>

          {/* 플레이스 상세 → 네이버 예약 페이지(session) */}
          <path 
            d={createHorizontalPath(560, 172, 620, 172, getStrokeWidth(metrics.place_to_booking_page))}
            fill="url(#placeGradient)"
            opacity="0.6"
          />
          <text x="590" y="172" textAnchor="middle" fontSize="8" fill="#9C27B0" fontWeight="600">
            {metrics.place_to_booking_page}
          </text>

          {/* 네이버 예약 페이지(session) → 예약 신청 (UV) */}
          <path 
            d={createHorizontalPath(700, 172, 760, 172, getStrokeWidth(metrics.booking_page_to_requests))}
            fill="#4CAF50"
            opacity="0.7"
          />
          <text x="730" y="172" textAnchor="middle" fontSize="8" fill="#4CAF50" fontWeight="600">
            {metrics.booking_page_to_requests}
          </text>

          {/* 전환율 표시 */}
          <text x="520" y="210" textAnchor="middle" fontSize="8" fill="#666">
            CVR {(metrics.bookingPageVisits / Math.max(1, metrics.placeDetailPV) * 100).toFixed(1)}%
          </text>
          <text x="660" y="210" textAnchor="middle" fontSize="8" fill="#666">
            CVR {(metrics.bookings / Math.max(1, metrics.bookingPageVisits) * 100).toFixed(1)}%
          </text>
        </svg>
      </div>
    );
  };

  // 채널별 데이터 준비
  const brandConversionData = [
    { channel: '네이버 카페', value: metrics.cafe_to_brand },
    { channel: '네이버 블로그', value: metrics.blog_to_brand },
    { channel: '홈페이지', value: metrics.site_to_brand }
  ];

  const placeDetailSourceData = [
    { name: '지도→목록', value: metrics.list_to_placeDetail },
    { name: '플레이스광고', value: metrics.ads_to_placeDetail },
    { name: '브랜드검색', value: metrics.brand_to_search_direct },
    { name: '브랜드지도', value: metrics.brand_to_map_direct },
    { name: '브랜드블로그', value: metrics.brand_to_blog_direct },
    { name: '브랜드사이트', value: metrics.brand_to_site_direct }
  ];

  const chartColors = ['#E91E63', '#FF5722', '#3F51B5', '#4CAF50', '#FFC107', '#2196F3'];

  return (
    <div className="p-6 bg-gray-50 min-h-screen">
      <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
        <h1 className="text-2xl font-bold text-gray-800 mb-2">병원 마케팅 통합 퍼널 대시보드</h1>
        <p className="text-gray-600">일반 키워드와 브랜드 키워드의 전체 여정 분석 및 최적화</p>
      </div>

      <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
        <h2 className="text-lg font-semibold mb-4 text-gray-800">통합 퍼널 구조</h2>
        <div className="flex gap-4">
          {/* 좌측: 그래프 컨테이너 80% 폭 */}
          <div className="w-4/5">
            <FlowGraph data={metrics} history={monthlyData} currentMonth={monthlyData[selectedMonthIndex]?.month} highlightedNodeIds={highlightedNodeIds} />
            {/* 월 선택 슬라이더 */}
            <div className="mt-4 p-3 bg-gray-50 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <div className="text-sm text-gray-700">
                  {monthlyData[selectedMonthIndex]?.month ? `선택: ${monthlyData[selectedMonthIndex].month}` : '데이터 로딩 중'}
                </div>
                <label className="text-xs text-gray-600 flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={isMonthlyMode}
                    onChange={(e) => setIsMonthlyMode(e.target.checked)}
                  />
                  실데이터 반영
                </label>
              </div>
              <input
                type="range"
                min={startIndex}
                max={Math.max(startIndex, monthlyData.length - 1)}
                value={selectedMonthIndex}
                onChange={(e) => setSelectedMonthIndex(Number(e.target.value))}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
              <div className="flex justify-between mt-1 text-xs text-gray-500">
                <span>{monthlyData[startIndex]?.month || '-'}</span>
                <span>{monthlyData[monthlyData.length - 1]?.month || '-'}</span>
              </div>
            </div>
          </div>
          {/* 우측: 시나리오 패널 */}
          <div className="w-1/5">
            <div className="bg-gray-50 rounded-lg p-3 h-full">
              <div className="flex items-center justify-between mb-2">
                <div className="text-sm font-semibold text-gray-700">시나리오</div>
                <button onClick={() => setHighlightedNodeIds([])} className="text-xs text-gray-500 hover:text-gray-700">해제</button>
              </div>
              <div className="space-y-2">
                <button onClick={() => setHighlightedNodeIds(['maplist'])} className="w-full text-left px-3 py-2 text-xs bg-white rounded border hover:bg-pink-50">네이버 지도 순위 10위권으로 상승</button>
                <button onClick={() => setHighlightedNodeIds(['homepage', 'cafe_home_proxy'])} className="w-full text-left px-3 py-2 text-xs bg-white rounded border hover:bg-pink-50">홈페이지 통한 예약 수 50% 증가</button>
                <button onClick={() => setHighlightedNodeIds(['blog', 'cafe_blog_proxy'])} className="w-full text-left px-3 py-2 text-xs bg-white rounded border hover:bg-pink-50">네이버 블로그 통한 예약 수 20% 증가</button>
                <button onClick={() => setHighlightedNodeIds(['cafe'])} className="w-full text-left px-3 py-2 text-xs bg-white rounded border hover:bg-pink-50">지역 카페 조회수 20% 증가</button>
                <button onClick={() => setHighlightedNodeIds(['cafe','brand','general','homepage','blog','maplist','ad','cafe_home_proxy','cafe_blog_proxy'])} className="w-full text-left px-3 py-2 text-xs bg-white rounded border hover:bg-pink-50">고객 의도 타겟하여 키워드 통일시키기</button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-white rounded-lg shadow-sm p-4">
          <div className="flex items-center justify-between mb-2">
            <Target className="w-5 h-5 text-green-600" />
            <span className="text-xs text-gray-500">최종 목표</span>
          </div>
          <div className="text-2xl font-bold text-gray-800">{metrics.bookings}</div>
          <div className="text-xs text-green-600">예약 완료</div>
          <div className="text-xs text-gray-500 mt-1">
            {formatPercent(baseMetrics.bookings, metrics.bookings)} vs 기준
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm p-4">
          <div className="flex items-center justify-between mb-2">
            <Users className="w-5 h-5 text-purple-600" />
            <span className="text-xs text-gray-500">브랜드 전환</span>
          </div>
          <div className="text-2xl font-bold text-gray-800">
            {metrics.blog_to_brand + metrics.site_to_brand + metrics.cafe_to_brand}
          </div>
          <div className="text-xs text-purple-600">신규 브랜드 인지</div>
          <div className="text-xs text-gray-500 mt-1">
            전환율 {((metrics.blog_to_brand + metrics.site_to_brand + metrics.cafe_to_brand) / 
                    Math.max(1, metrics.search_to_blog + metrics.search_to_site + metrics.search_to_cafe) * 100).toFixed(1)}%
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm p-4">
          <div className="flex items-center justify-between mb-2">
            <FileText className="w-5 h-5 text-blue-600" />
            <span className="text-xs text-gray-500">플레이스 집중</span>
          </div>
          <div className="text-2xl font-bold text-gray-800">{metrics.placeDetailPV}</div>
          <div className="text-xs text-blue-600">플레이스 상세 PV</div>
          <div className="text-xs text-gray-500 mt-1">
            예약 전환 {(metrics.bookingPageVisits / Math.max(1, metrics.placeDetailPV) * 100).toFixed(1)}%
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm p-4">
          <div className="flex items-center justify-between mb-2">
            <Activity className="w-5 h-5 text-orange-600" />
            <span className="text-xs text-gray-500">총 수요</span>
          </div>
          <div className="text-2xl font-bold text-gray-800">
            {metrics.nonBrandDemand + metrics.brandDemand}
          </div>
          <div className="text-xs text-orange-600">전체 키워드 검색</div>
          <div className="text-xs text-gray-500 mt-1">
            브랜드 비중 {(metrics.brandDemand / Math.max(1, metrics.nonBrandDemand + metrics.brandDemand) * 100).toFixed(1)}%
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-6 mb-6">
        <div className="bg-white rounded-lg shadow-sm p-6">
          <h3 className="text-md font-semibold mb-4 text-gray-800">일반 키워드 → 브랜드 전환 경로</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={brandConversionData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="channel" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="value" fill="#8884d8">
                {brandConversionData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={chartColors[index]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white rounded-lg shadow-sm p-6">
          <h3 className="text-md font-semibold mb-4 text-gray-800">플레이스 상세 유입 구성</h3>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie
                data={placeDetailSourceData}
                cx="50%"
                cy="50%"
                innerRadius={40}
                outerRadius={70}
                paddingAngle={2}
                dataKey="value"
              >
                {placeDetailSourceData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={chartColors[index % chartColors.length]} />
                ))}
              </Pie>
              <Tooltip formatter={(value) => value.toLocaleString()} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg shadow-lg p-6 text-white">
        <h3 className="text-lg font-semibold mb-4">🚀 실시간 액션 실행 상태</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          {[
            { 
              service: '카페 바이럴',
              current: metrics.cafe_to_brand,
              base: baseMetrics.cafe_to_brand,
              unit: '건',
              desc: '브랜드 전환'
            },
            {
              service: '메디컨텐츠',
              current: metrics.search_to_blog,
              base: baseMetrics.search_to_blog,
              unit: 'PV',
              desc: '블로그 유입'
            },
            {
              service: '리뷰부스터',
              current: metrics.map_to_list,
              base: baseMetrics.map_to_list,
              unit: '건',
              desc: '지도 노출'
            },
            {
              service: '플레이스광고',
              current: metrics.list_to_placeAds,
              base: baseMetrics.list_to_placeAds,
              unit: '클릭',
              desc: '광고 성과'
            },
            {
              service: '메디페이지',
              current: metrics.search_to_site,
              base: baseMetrics.search_to_site,
              unit: 'PV',
              desc: '사이트 유입'
            },
            {
              service: '전환 최적화',
              current: metrics.bookingPageVisits,
              base: baseMetrics.bookingPageVisits,
              unit: '건',
              desc: '예약 시도'
            }
          ].map((action, idx) => (
            <div key={`action-${idx}`} className="bg-white/10 rounded-lg p-3">
              <div className="text-xs opacity-80 mb-1">{action.service}</div>
              <div className="text-xl font-bold">{action.current} {action.unit}</div>
              <div className="text-xs opacity-80">
                {action.desc} {formatPercent(action.base, action.current)}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default HospitalMarketingDashboard;