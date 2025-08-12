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
  const homepageTotalsPath = path.join(rawDir, 'channel_homepage.csv');
  const blogTotalsPath = path.join(rawDir, 'cnt_blog.csv');
  const channelPlaceDetailPath = path.join(rawDir, 'channel_place_detail.csv');
  const placeToBookingPath = path.join(rawDir, 'placedetail_to_bookingpage.csv');
  const bookingRequestsPath = path.join(rawDir, 'cnt_booking_requests.csv');
  const placeAdPath = path.join(rawDir, 'placead.csv');
  const keywordPlaceDetailPath = path.join(rawDir, 'keyword_placedetail.csv');
  const mapRankPath = path.join(rawDir, 'map_rank.csv');

  const placeRaw = fs.readFileSync(placeCsvPath, 'utf-8');
  const bookingRaw = fs.readFileSync(bookingCsvPath, 'utf-8');
  const brandSearchRaw = fs.readFileSync(brandSearchAggPath, 'utf-8');
  const brandHomepageRaw = fs.readFileSync(brandHomepageInPath, 'utf-8');
  const brandBlogElzaRaw = fs.readFileSync(brandBlogInElzaPath, 'utf-8');
  const brandBlogNatenRaw = fs.readFileSync(brandBlogInNatenPath, 'utf-8');
  const homepageTotalsRaw = fs.readFileSync(homepageTotalsPath, 'utf-8');
  const blogTotalsRaw = fs.readFileSync(blogTotalsPath, 'utf-8');
  const channelPlaceDetailRaw = fs.readFileSync(channelPlaceDetailPath, 'utf-8');
  const placeToBookingRaw = fs.readFileSync(placeToBookingPath, 'utf-8');
  const bookingRequestsRaw = fs.readFileSync(bookingRequestsPath, 'utf-8');
  const placeAdRaw = fs.readFileSync(placeAdPath, 'utf-8');
  const keywordPlaceDetailRaw = fs.readFileSync(keywordPlaceDetailPath, 'utf-8');
  const mapRankRaw = fs.readFileSync(mapRankPath, 'utf-8');

  const placeRows = parseCSV(placeRaw);
  const bookingRows = parseCSV(bookingRaw);
  const brandSearchRows = parseCSV(brandSearchRaw);
  const brandHomepageRows = parseCSV(brandHomepageRaw);
  const brandBlogElzaRows = parseCSV(brandBlogElzaRaw);
  const brandBlogNatenRows = parseCSV(brandBlogNatenRaw);
  const homepageTotalRows = parseCSV(homepageTotalsRaw);
  const blogTotalRows = parseCSV(blogTotalsRaw);
  const channelPlaceDetailRows = parseCSV(channelPlaceDetailRaw);
  const placeToBookingRows = parseCSV(placeToBookingRaw);
  const bookingRequestsRows = parseCSV(bookingRequestsRaw);
  const placeAdRows = parseCSV(placeAdRaw);
  const keywordPlaceDetailRows = parseCSV(keywordPlaceDetailRaw);
  const mapRankRows = parseCSV(mapRankRaw);

  // helper: parse date like 'Sep.24' to '2024-09'
  const monthAbbrevToNum = {
    Jan: '01', Feb: '02', Mar: '03', Apr: '04', May: '05', Jun: '06', Jul: '07', Aug: '08', Sep: '09', Oct: '10', Nov: '11', Dec: '12'
  };
  function parseMonthToken(token) {
    // e.g., 'Sep.24' or 'Jan.25'
    if (!token) return null;
    const m = token.match(/^(\w{3})\.(\d{2})$/);
    if (!m) return null;
    const mon = monthAbbrevToNum[m[1]];
    if (!mon) return null;
    const yy = Number(m[2]);
    const yyyy = yy >= 70 ? 1900 + yy : 2000 + yy;
    return `${yyyy}-${mon}`;
  }

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
      else if (name.includes('플레이스상세')) sumInto(monthToMetrics[key], 'placeDetailBase', pv);
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
    // 일반→브랜드 전환 가정 (브랜드 유입 엣지에는 포함하지 않음)
    const convRate = 0.3;
    m.blog_to_brand = Math.round(m.search_to_blog * convRate);
    m.site_to_brand = Math.round(m.search_to_site * convRate);
    m.cafe_to_brand = Math.round(m.search_to_cafe * convRate);
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

    if (code === 'ple') sumInto(m, 'placeDetailBase', pageVisits);
    if (code === 'psa') sumInto(m, 'ads_to_placeDetail', pageVisits);
    if (code === 'pll') sumInto(m, 'list_to_placeDetail', pageVisits);
    // 예약 완료는 월 합산
    m.bookings = (m.bookings || 0) + bookingRequests;
  }

  // 브랜드/일반 노드 주입: keyword_search_amount_agg.csv
  for (const r of brandSearchRows) {
    const key = (r.date || '').trim();
    const flag = (r.is_brand ?? r.is_Brand ?? '').trim();
    if (!key || !flag) continue;
    if (!monthToMetrics[key]) monthToMetrics[key] = { month: key };
    const cnt = Number(r.cnt || '0');
    if (flag === 'Y') monthToMetrics[key].brand_node_search = (monthToMetrics[key].brand_node_search || 0) + cnt;
    if (flag === 'N') monthToMetrics[key].general_node_search = (monthToMetrics[key].general_node_search || 0) + cnt;
  }

  // 2) 브랜드/일반 -> 홈페이지 엣지
  for (const r of brandHomepageRows) {
    const key = (r.date || '').trim();
    const flag = (r.is_brand ?? r.is_Brand ?? '').trim();
    if (!key || !flag) continue;
    if (!monthToMetrics[key]) monthToMetrics[key] = { month: key };
    const cnt = Number(r.cnt || '0');
    if (flag === 'Y') monthToMetrics[key].brand_to_site_from_csv = (monthToMetrics[key].brand_to_site_from_csv || 0) + cnt;
    if (flag === 'N') monthToMetrics[key].general_to_site_from_csv = (monthToMetrics[key].general_to_site_from_csv || 0) + cnt;
  }

  // 3) 브랜드/일반 -> 블로그 엣지: 두 파일 합계
  const blogBrandAdd = (rows) => {
    for (const r of rows) {
      const key = (r.date || '').trim();
      const flag = (r.is_brand ?? r.is_Brand ?? '').trim();
      if (!key || !flag) continue;
      if (!monthToMetrics[key]) monthToMetrics[key] = { month: key };
      const cnt = Number(r.cnt || '0');
      if (flag === 'Y') monthToMetrics[key].brand_to_blog_from_csv = (monthToMetrics[key].brand_to_blog_from_csv || 0) + cnt;
      if (flag === 'N') monthToMetrics[key].general_to_blog_from_csv = (monthToMetrics[key].general_to_blog_from_csv || 0) + cnt;
    }
  };
  blogBrandAdd(brandBlogElzaRows);
  blogBrandAdd(brandBlogNatenRows);

  // 브랜드 엣지 확정: CSV 값으로 덮어쓰기, 나머지는 0
  for (const key of Object.keys(monthToMetrics)) {
    const m = monthToMetrics[key];
    m.brand_to_blog_direct = m.brand_to_blog_from_csv || 0;
    m.brand_to_site_direct = m.brand_to_site_from_csv || 0;
    m.brand_to_search_direct = 0;
    m.brand_to_map_direct = 0;
    m.general_to_blog_direct = m.general_to_blog_from_csv || 0;
    m.general_to_site_direct = m.general_to_site_from_csv || 0;
    // 지도 임시값
    m.general_to_map_direct = 1000;
    m.map_node_total = m.general_to_map_direct;
  }

  // 홈페이지/블로그 노드 총량 주입
  for (const r of homepageTotalRows) {
    const key = (r.date || '').trim();
    if (!key) continue;
    if (!monthToMetrics[key]) monthToMetrics[key] = { month: key };
    const tot = Number(r.tot || '0');
    monthToMetrics[key].homepage_node_total = (monthToMetrics[key].homepage_node_total || 0) + tot;
  }
  for (const r of blogTotalRows) {
    const key = (r.date || '').trim();
    if (!key) continue;
    if (!monthToMetrics[key]) monthToMetrics[key] = { month: key };
    const tot = Number(r.tot || '0');
    monthToMetrics[key].blog_node_total = (monthToMetrics[key].blog_node_total || 0) + tot;
  }

  // 홈페이지/네이버 블로그 → 플레이스상세 엣지 주입
  for (const r of channelPlaceDetailRows) {
    const key = (r.date || '').trim();
    if (!key) continue;
    if (!monthToMetrics[key]) monthToMetrics[key] = { month: key };
    const ch = (r.channel || '').trim();
    const pv = Number(r.pv || '0');
    if (ch === '웹사이트') sumInto(monthToMetrics[key], 'homepage_to_place_detail', pv);
    if (ch === '네이버 블로그') sumInto(monthToMetrics[key], 'blog_to_place_detail', pv);
    if (ch === '네이버지도') sumInto(monthToMetrics[key], 'place_list_to_detail', pv);
    // 플레이스상세 노드 총량은 월별 pv 합계
    sumInto(monthToMetrics[key], 'place_detail_node_total', pv);
  }

  // 최종 placeDetailPV 및 예약 시도 재계산
  for (const key of Object.keys(monthToMetrics)) {
    const m = monthToMetrics[key];
    // 플레이스상세 노드 값: channel_place_detail.csv 월별 pv 합계로 정의
    m.placeDetailPV = m.place_detail_node_total || 0;
  }

  // 플레이스상세 → 네이버 예약 페이지(session) 엣지 및 노드: placedetail_to_bookingpage.csv cnt
  for (const r of placeToBookingRows) {
    const key = (r.date || '').trim();
    if (!key) continue;
    if (!monthToMetrics[key]) monthToMetrics[key] = { month: key };
    const cnt = Number(r.cnt || '0');
    monthToMetrics[key].bookingPageVisits = cnt; // 노드 값
    monthToMetrics[key].place_to_booking_page = cnt; // 엣지 값 (동일)
  }

  // 네이버 예약 페이지(session) → 예약 신청 (UV) 엣지 및 노드: cnt_booking_requests.csv cnt
  for (const r of bookingRequestsRows) {
    const key = (r.date || '').trim();
    if (!key) continue;
    if (!monthToMetrics[key]) monthToMetrics[key] = { month: key };
    const cnt = Number(r.cnt || '0');
    monthToMetrics[key].bookings = cnt; // 노드 값
    monthToMetrics[key].booking_page_to_requests = cnt; // 엣지 값
  }

  // 플레이스 광고: 노드(노출수), 엣지(클릭수)
  for (const r of placeAdRows) {
    const key = (r.date || '').trim();
    if (!key) continue;
    if (!monthToMetrics[key]) monthToMetrics[key] = { month: key };
    const exposure = Number(r.exposure || '0');
    const click = Number(r.click || '0');
    monthToMetrics[key].place_ad_node_total = exposure;
    monthToMetrics[key].place_ad_to_detail = click;
  }

  // 플레이스 목록 노드: 0으로 고정
  for (const key of Object.keys(monthToMetrics)) {
    monthToMetrics[key].place_list_node_total = 0;
  }

  // 브랜드/일반 검색 → 플레이스 상세 (keyword_placedetail.csv, search_cnt)
  for (const r of keywordPlaceDetailRows) {
    const ym = parseMonthToken((r.date || '').trim());
    if (!ym) continue;
    if (!monthToMetrics[ym]) monthToMetrics[ym] = { month: ym };
    const flag = (r.is_brand || '').trim();
    const searchCnt = Number(r.search_cnt || '0');
    if (flag === 'Y') sumInto(monthToMetrics[ym], 'brand_search_to_detail', searchCnt);
    if (flag === 'N') sumInto(monthToMetrics[ym], 'general_search_to_detail', searchCnt);
  }

  // 네이버 지도(플레이스 목록) 노드 내부 표시용: 월별 지도 순위
  for (const r of mapRankRows) {
    const key = (r.date || '').trim();
    if (!key) continue;
    if (!monthToMetrics[key]) monthToMetrics[key] = { month: key };
    const rank = Number(r.rank || '0');
    monthToMetrics[key].map_rank = rank;
  }

  const months = Object.keys(monthToMetrics).sort();
  const out = months.map((k) => monthToMetrics[k]);

  fs.writeFileSync(path.join(outDir, 'funnel_monthly.json'), JSON.stringify(out, null, 2), 'utf-8');
  console.log(`Built ${out.length} month records -> data/service/funnel_monthly.json`);
}

build();


