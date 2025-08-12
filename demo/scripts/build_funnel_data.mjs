import fs from 'fs';
import path from 'path';

const rootDir = '/Users/min/codes/medilawyer_sales/blog_automation/demo';
const rawDir = path.join(rootDir, 'data', 'raw');
const outDir = path.join(rootDir, 'data', 'service');

function ensureDir(dir) {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

function stripBOM(s) {
  if (!s) return s;
  return s.replace(/^\uFEFF/, '');
}

function parseCSV(content) {
  const lines = content.split(/\r?\n/).filter(Boolean);
  if (lines.length <= 1) return [];
  const header = lines[0].split(',').map((h) => stripBOM(h.trim()));
  return lines.slice(1).map((line) => {
    const cols = [];
    let current = '';
    let inQuotes = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (ch === '"') {
        inQuotes = !inQuotes;
      } else if (ch === ',' && !inQuotes) {
        cols.push(current);
        current = '';
      } else {
        current += ch;
      }
    }
    cols.push(current);
    const obj = {};
    header.forEach((h, idx) => (obj[h] = stripBOM((cols[idx] ?? '').trim())));
    return obj;
  });
}

function yyyymm(dateStr) {
  // input like 2024-09-01
  const [y, m] = dateStr.split('-');
  return `${y}-${m}`;
}

function sumInto(obj, key, value) {
  if (!Number.isFinite(value)) return;
  obj[key] = (obj[key] ?? 0) + value;
}

function build() {
  ensureDir(outDir);

  const placeCsvPath = path.join(rawDir, 'NATENCLINIC_2025_07_integrated_statistics.csv');
  const bookingCsvPath = path.join(rawDir, 'NATENCLINIC_2025_07_booking_integrated_statistics.csv');
  const brandSearchAggPath = path.join(rawDir, 'keyword_search_amount_agg.csv');
  const brandHomepageInPath = path.join(rawDir, 'keyword_naver_homepage_in.csv');
  const brandBlogInElzaPath = path.join(rawDir, 'keyword_blog_in_elza79.csv');
  const brandBlogInNatenPath = path.join(rawDir, 'keyword_blog_in_natenclinic.csv');

  const placeRaw = fs.readFileSync(placeCsvPath, 'utf-8');
  const bookingRaw = fs.readFileSync(bookingCsvPath, 'utf-8');
  const brandSearchRaw = fs.readFileSync(brandSearchAggPath, 'utf-8');
  const brandHomepageRaw = fs.readFileSync(brandHomepageInPath, 'utf-8');
  const brandBlogElzaRaw = fs.readFileSync(brandBlogInElzaPath, 'utf-8');
  const brandBlogNatenRaw = fs.readFileSync(brandBlogInNatenPath, 'utf-8');

  const placeRows = parseCSV(placeRaw);
  const bookingRows = parseCSV(bookingRaw);
  const brandSearchRows = parseCSV(brandSearchRaw);
  const brandHomepageRows = parseCSV(brandHomepageRaw);
  const brandBlogElzaRows = parseCSV(brandBlogElzaRaw);
  const brandBlogNatenRows = parseCSV(brandBlogNatenRaw);

  // 월별 집계 컨테이너
  const monthToMetrics = {};

  // 채널/키워드 기반 유입 → 지도/검색/블로그/사이트/플레이스 목록/플레이스 광고 등으로 분류
  for (const r of placeRows) {
    if (!r.date) continue;
    const key = yyyymm(r.date);
    if (!monthToMetrics[key]) {
      monthToMetrics[key] = {
        month: key,
        // 수요/인지 경로
        nonBrand_to_search: 0,
        nonBrand_to_map: 0,
        search_to_blog: 0,
        search_to_site: 0,
        search_to_cafe: 0,
        // 지도 경로
        map_to_list: 0,
        list_to_placeAds: 0,
        list_to_placeDetail: 0,
        ads_to_placeDetail: 0,
        // 브랜드 분배(데이터 없을 경우 가정)
        brand_to_search_direct: 0,
        brand_to_map_direct: 0,
        brand_to_blog_direct: 0,
        brand_to_site_direct: 0,
        // 플레이스 상세/예약
        placeDetailPV: 0,
        bookingPageVisits: 0,
        bookings: 0,
      };
    }
    const pv = Number(r.pv ?? '0');
    const name = r.name ?? '';
    const dataType = r.data_type ?? '';

    if (dataType === 'channel') {
      // 간단 매핑 규칙
      if (name.includes('지도')) sumInto(monthToMetrics[key], 'nonBrand_to_map', pv);
      else if (name.includes('검색')) sumInto(monthToMetrics[key], 'nonBrand_to_search', pv);
      else if (name.includes('블로그')) sumInto(monthToMetrics[key], 'search_to_blog', pv);
      else if (name.includes('웹사이트') || name.includes('홈페이지')) sumInto(monthToMetrics[key], 'search_to_site', pv);
      else if (name.includes('카페')) sumInto(monthToMetrics[key], 'search_to_cafe', pv);
      else if (name.includes('플레이스광고')) sumInto(monthToMetrics[key], 'list_to_placeAds', pv);
      else if (name.includes('플레이스목록')) sumInto(monthToMetrics[key], 'map_to_list', pv);
      else if (name.includes('플레이스상세')) sumInto(monthToMetrics[key], 'placeDetailPV', pv);
    }
  }

  // 지도 → 목록 유도치가 없으면 간단한 비율로 추정(지도의 50%)
  for (const key of Object.keys(monthToMetrics)) {
    const m = monthToMetrics[key];
    if (!m.map_to_list) m.map_to_list = Math.round((m.nonBrand_to_map || 0) * 0.5);
    if (!m.list_to_placeAds) m.list_to_placeAds = Math.round(m.map_to_list * 0.2);
    m.ads_to_placeDetail = Math.round(m.list_to_placeAds * 0.85);
    // 검색→인지 경로가 비어있으면 합리적 가정으로 생성
    const baseSearch = m.nonBrand_to_search || 0;
    if (!m.search_to_blog || !m.search_to_site || !m.search_to_cafe) {
      m.search_to_blog = Math.round(baseSearch * 0.39);
      m.search_to_site = Math.round(baseSearch * 0.28);
      m.search_to_cafe = Math.round(baseSearch * 0.33);
    }
    // 목록→상세가 없으면 기본 전환율로 생성
    if (!m.list_to_placeDetail) m.list_to_placeDetail = Math.round(m.map_to_list * 0.5);
    // 브랜드 수요는 관측치 없으므로 전체의 30%를 가정하고 분배
    const brandTotal = Math.round((m.nonBrand_to_search + m.nonBrand_to_map) * 0.3);
    m.brand_to_search_direct = Math.round(brandTotal * 0.4);
    m.brand_to_map_direct = Math.round(brandTotal * 0.3);
    m.brand_to_blog_direct = Math.round(brandTotal * 0.15);
    m.brand_to_site_direct = Math.round(brandTotal * 0.15);
    // 일반→브랜드 전환 가정
    const convRate = 0.3;
    m.blog_to_brand = Math.round(m.search_to_blog * convRate);
    m.site_to_brand = Math.round(m.search_to_site * convRate);
    m.cafe_to_brand = Math.round(m.search_to_cafe * convRate);
    // 플레이스 상세 PV 합산
    m.placeDetailPV = (m.placeDetailPV || 0)
      + m.brand_to_search_direct + m.brand_to_map_direct
      + m.brand_to_blog_direct + m.brand_to_site_direct
      + m.ads_to_placeDetail + m.list_to_placeDetail;
  }

  // 예약 통계 반영 (bookingRows)
  // channel_code 기준: pll(목록), ple(상세), psa(광고), bnb(블로그), bne(네이버기타), bmp(지도)
  for (const r of bookingRows) {
    if (!r.date) continue;
    const key = yyyymm(r.date);
    if (!monthToMetrics[key]) continue;
    const pageVisits = Number(r.page_visits ?? '0');
    const bookingRequests = Number(r.booking_requests ?? '0');
    const code = (r.channel_code || '').trim();
    const m = monthToMetrics[key];

    if (code === 'ple') sumInto(m, 'placeDetailPV', pageVisits);
    if (code === 'psa') sumInto(m, 'ads_to_placeDetail', pageVisits);
    if (code === 'pll') sumInto(m, 'list_to_placeDetail', pageVisits);
    // 예약 시도/완료 추정: booking_requests는 예약 완료로 사용, 시도는 상세 PV의 25% 기준으로 보수적으로 설정
    m.bookingPageVisits = Math.max(m.bookingPageVisits, Math.round(m.placeDetailPV * 0.25));
    m.bookings = Math.max(m.bookings, bookingRequests);
  }

  // 브랜드 노드 및 엣지 주입 (명세 기반)
  // 1) 브랜드 노드 값: keyword_search_amount_agg.csv 에서 is_brand(Y) cnt
  for (const r of brandSearchRows) {
    const key = (r.date || '').trim();
    const isBrand = (r.is_brand ?? r.is_Brand ?? '').trim();
    if (!key || isBrand !== 'Y') continue;
    if (!monthToMetrics[key]) {
      monthToMetrics[key] = { month: key };
    }
    const cnt = Number(r.cnt || '0');
    monthToMetrics[key].brand_node_search = (monthToMetrics[key].brand_node_search || 0) + cnt;
  }

  // 2) 브랜드 -> 홈페이지 엣지: keyword_naver_homepage_in.csv 에서 is_brand(Y) cnt
  for (const r of brandHomepageRows) {
    const key = (r.date || '').trim();
    const isBrand = (r.is_brand ?? r.is_Brand ?? '').trim();
    if (!key || isBrand !== 'Y') continue;
    if (!monthToMetrics[key]) monthToMetrics[key] = { month: key };
    const cnt = Number(r.cnt || '0');
    monthToMetrics[key].brand_to_site_direct = (monthToMetrics[key].brand_to_site_direct || 0) + cnt;
  }

  // 3) 브랜드 -> 블로그 엣지: 두 파일 합계 (is_brand(Y))
  const blogBrandAdd = (rows) => {
    for (const r of rows) {
      const key = (r.date || '').trim();
      const isBrand = (r.is_brand ?? r.is_Brand ?? '').trim();
      if (!key || isBrand !== 'Y') continue;
      if (!monthToMetrics[key]) monthToMetrics[key] = { month: key };
      const cnt = Number(r.cnt || '0');
      monthToMetrics[key].brand_to_blog_direct = (monthToMetrics[key].brand_to_blog_direct || 0) + cnt;
    }
  };
  blogBrandAdd(brandBlogElzaRows);
  blogBrandAdd(brandBlogNatenRows);

  // 명세에 없는 브랜드 엣지는 0으로 보정
  for (const key of Object.keys(monthToMetrics)) {
    const m = monthToMetrics[key];
    m.brand_to_search_direct = 0;
    m.brand_to_map_direct = m.brand_to_map_direct || 0; // 유지하되 없으면 0
  }

  const months = Object.keys(monthToMetrics).sort();
  const out = months.map((k) => monthToMetrics[k]);

  fs.writeFileSync(path.join(outDir, 'funnel_monthly.json'), JSON.stringify(out, null, 2), 'utf-8');
  console.log(`Built ${out.length} month records -> data/service/funnel_monthly.json`);
}

build();


