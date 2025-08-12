import React, { useMemo, useEffect } from 'react';
import ReactFlow, { Background, Controls, MiniMap, useNodesState, useEdgesState, NodeToolbar, Handle, Position } from 'reactflow';
import 'reactflow/dist/style.css';

// 열 위치 좌표 (고정 배치)
const COL_X = [80, 260, 460, 700, 900, 1100];
const ROW_Y = {
  cafe: 140,
  brand: 80,
  general: 300,
  homepage: 40,
  blog: 110,
  maplist: 240,
  ad: 320,
  detail: 170,
  booking: 170,
  request: 170,
};

// 색상 팔레트
const COLORS = {
  cafe: '#E91E63',
  brand: '#FF9800',
  general: '#FBC02D',
  homepage: '#3F51B5',
  blog: '#FF5722',
  maplist: '#00BCD4',
  ad: '#FFC107',
  detail: '#9C27B0',
  booking: '#03A9F4',
  request: '#4CAF50',
  linkGray: '#BDBDBD',
};

// 값 스케일 → 노드 사이즈/엣지 두께
const scale = (value, min = 28, max = 84, domainMax = 10000) => {
  if (!value || value <= 0) return min;
  const r = Math.min(1, value / domainMax);
  return Math.round(min + (max - min) * r);
};

const fmt = (value) => {
  if (value == null) return '0';
  const v = Number(value) || 0;
  return v.toLocaleString();
};

// 커스텀 노드들
const CircleNode = ({ id, data }) => {
  const size = data.size || 36;
  const color = data.color || '#888';
  return (
    <div title={data.tooltip || ''} style={{ width: size, height: size, borderRadius: '50%', border: `4px solid ${color}`, background: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 6, boxShadow: '0 1px 3px rgba(0,0,0,0.08)', position: 'relative' }}>
      <Handle type="target" position={Position.Left} style={{ width: 8, height: 8, background: color, border: 'none' }} />
      <Handle type="source" position={Position.Right} style={{ width: 8, height: 8, background: color, border: 'none' }} />
      <Handle id="top" type="source" position={Position.Top} style={{ width: 8, height: 8, background: color, border: 'none' }} />
      {/* 노드 상단 타이틀 */}
      <div style={{ position: 'absolute', top: -26, left: '50%', transform: 'translateX(-50%)', fontSize: 10, fontWeight: 700, color: '#111', width: Math.max(140, size*2), textAlign: 'center', lineHeight: 1.2 }} dangerouslySetInnerHTML={{ __html: data.titleHtml || data.title }} />
      <div style={{ textAlign: 'center', lineHeight: 1.15 }}>
        <div style={{ fontSize: 14, fontWeight: 800, color: data.valueColor || color }}>{fmt(data.value)}</div>
        {data.sub && <div style={{ fontSize: 10, color: '#666' }}>{data.sub}</div>}
      </div>
    </div>
  );
};

const nodeTypes = { circle: CircleNode };

// 테이퍼 엣지: 좌측은 굵고, 우측으로 갈수록 1/5 수준으로 얇아지는 그라데이션 스트립
const TaperEdge = ({ id, sourceX, sourceY, targetX, targetY, data }) => {
  const { colorL = '#bbb', colorR = '#888', widthStart = 10, widthEnd = 2, label = '', arrow = false } = data || {};
  const midX = (sourceX + targetX) / 2;
  const path = `M ${sourceX} ${sourceY - widthStart/2}
               Q ${midX} ${sourceY - widthStart/2} ${targetX} ${targetY - widthEnd/2}
               L ${targetX} ${targetY + widthEnd/2}
               Q ${midX} ${sourceY + widthStart/2} ${sourceX} ${sourceY + widthStart/2}
               Z`;
  const gradId = `grad-${id}`;
  const angleDeg = Math.atan2(targetY - sourceY, targetX - sourceX) * 180 / Math.PI;
  return (
    <g>
      <defs>
        <linearGradient id={gradId} x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor={colorL} stopOpacity={0.9} />
          <stop offset="100%" stopColor={colorR} stopOpacity={0.6} />
        </linearGradient>
      </defs>
      <path d={path} fill={`url(#${gradId})`} />
      {label && (
        <>
          <text x={midX} y={(sourceY + targetY) / 2 - (Math.max(widthStart, widthEnd) / 2 + 10)} textAnchor="middle" fontSize={20} fontWeight={900} fill="#ffffff" stroke="#ffffff" strokeWidth={4} dominantBaseline="middle" style={{ pointerEvents: 'none' }}>{label}</text>
          <text x={midX} y={(sourceY + targetY) / 2 - (Math.max(widthStart, widthEnd) / 2 + 10)} textAnchor="middle" fontSize={20} fontWeight={800} fill="#111827" dominantBaseline="middle" style={{ pointerEvents: 'none' }}>{label}</text>
        </>
      )}
      {arrow && (
        <path d={`M ${targetX} ${targetY} l -12 -7 l 0 14 z`} fill={colorR} transform={`rotate(${angleDeg}, ${targetX}, ${targetY})`} />
      )}
    </g>
  );
};

// 단색+고정 굵기 화살표 엣지 (카페 전용)
const ArrowEdge = ({ id, sourceX, sourceY, targetX, targetY, data }) => {
  const { color = '#f9a8d4', width = 1, label = '' } = data || {};
  const midX = (sourceX + targetX) / 2;
  const path = `M ${sourceX} ${sourceY} Q ${midX} ${sourceY} ${targetX} ${targetY}`;
  const markerId = `arrow-${id}`;
  return (
    <g>
      <defs>
        <marker id={markerId} markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto-start-reverse">
          <path d="M0,0 L0,12 L12,6 z" fill={color} />
        </marker>
      </defs>
      <path d={path} fill="none" stroke={color} strokeWidth={width} markerEnd={`url(#${markerId})`} />
      {label && (
        <text x={midX} y={sourceY - (width + 12)} textAnchor="middle" fontSize={20} fontWeight={800} fill="#111827" dominantBaseline="middle" style={{ pointerEvents: 'none' }}>{label}</text>
      )}
    </g>
  );
};

const baseNode = (id, type, x, y, props) => ({ id, type, position: { x, y }, data: { ...props } });

export default function FlowGraph({ data, history = [] }) {
  const initialNodes = useMemo(() => {
    if (!data) return [];
    const n = [];
    // 히스토리 기반 최대값(일반 검색은 1/10)으로 노드 스케일
    const maxOf = (key, adjust = (v)=>v) => Math.max(1, ...history.map(m => adjust(m[key] || 0)));
    const maxCafe = maxOf('cafe_view');
    const maxBrand = maxOf('brand_node_search');
    const maxGeneral = maxOf('general_node_search', v=>v/10);
    const maxHome = maxOf('homepage_node_total');
    const maxBlog = maxOf('blog_node_total');
    const maxMapList = maxOf('place_list_to_detail');
    const maxAd = maxOf('place_ad_node_total');
    const maxDetail = maxOf('placeDetailPV');
    const maxBooking = maxOf('bookingPageVisits');
    const maxRequest = maxOf('bookings');

    const PALETTE = { general: '#f3f4f6', brand: '#e5e7eb', homepage: '#bfdbfe', blog: '#93c5fd', maplist: '#60a5fa', ad: '#3b82f6', detail: '#2563eb', booking: '#1d4ed8', request: '#1e40af', cafe: '#f9a8d4' };
    n.push(baseNode('cafe', 'circle', COL_X[0], ROW_Y.cafe, { title: '지역 카페 조회수', value: data.cafe_view || 0, color: PALETTE.cafe, size: scale(data.cafe_view, 28, 84, maxCafe) }));
    n.push(baseNode('brand', 'circle', COL_X[1], ROW_Y.brand, { title: '브랜드 키워드 네이버 검색량', value: data.brand_node_search || 0, color: PALETTE.brand, size: scale(data.brand_node_search, 28, 84, maxBrand) }));
    n.push(baseNode('general', 'circle', COL_X[1], ROW_Y.general, { title: '일반 키워드 네이버 검색량', value: data.general_node_search || 0, color: PALETTE.general, size: scale((data.general_node_search||0)/10, 28, 84, maxGeneral) }));
    n.push(baseNode('homepage', 'circle', COL_X[2], ROW_Y.homepage, { title: '홈페이지', value: data.homepage_node_total || 0, color: PALETTE.homepage, size: scale(data.homepage_node_total, 28, 84, maxHome) }));
    n.push(baseNode('blog', 'circle', COL_X[2], ROW_Y.blog, { title: '네이버 블로그', value: data.blog_node_total || 0, color: PALETTE.blog, size: scale(data.blog_node_total, 28, 84, maxBlog) }));
    n.push(baseNode('maplist', 'circle', COL_X[2], ROW_Y.maplist, { title: '네이버 지도(플레이스 목록)', value: data.place_list_to_detail || 0, sub: `순위: ${data.map_rank||0}`, color: PALETTE.maplist, size: scale(data.place_list_to_detail, 28, 84, maxMapList) }));
    n.push(baseNode('ad', 'circle', COL_X[2], ROW_Y.ad, { title: '플레이스 광고', value: data.place_ad_node_total || 0, color: PALETTE.ad, size: scale(data.place_ad_node_total, 28, 84, maxAd) }));
    n.push(baseNode('detail', 'circle', COL_X[3], ROW_Y.detail, { title: '플레이스 상세', value: data.placeDetailPV || 0, color: PALETTE.detail, size: scale(data.placeDetailPV, 28, 84, maxDetail) }));
    n.push(baseNode('booking', 'circle', COL_X[4], ROW_Y.booking, { title: '네이버 예약 페이지', value: data.bookingPageVisits || 0, color: PALETTE.booking, size: scale(data.bookingPageVisits, 28, 84, maxBooking) }));
    n.push(baseNode('request', 'circle', COL_X[5], ROW_Y.request, { title: '예약 신청 (UV)', value: data.bookings || 0, color: PALETTE.request, size: scale(data.bookings, 28, 84, maxRequest) }));
    return n;
  }, [data, history]);
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  useEffect(() => setNodes(initialNodes), [initialNodes, setNodes]);

  const initialEdges = useMemo(() => {
    if (!data) return [];
    const e = [];
    // 엣지 두께 스케일을 위해 최대값 계산 (좌우 동일 두께)
    const values = [];
    const pushVal = (v) => { if (v && v > 0) values.push(v); };
    [
      data.brand_to_site_direct, data.brand_to_blog_direct, data.general_to_site_direct, data.general_to_blog_direct,
      data.place_list_to_detail, data.place_ad_to_detail, data.homepage_to_place_detail, data.blog_to_place_detail,
      data.place_to_booking_page, data.booking_page_to_requests
    ].forEach(pushVal);
    const vmax = Math.max(1, ...values);
    const width = (v) => Math.max(8, Math.round(8 + 36 * (v / vmax))); // 최대 ~44px

    const P = { general: '#f3f4f6', brand: '#e5e7eb', homepage: '#bfdbfe', blog: '#93c5fd', maplist: '#60a5fa', ad: '#3b82f6', detail: '#2563eb', booking: '#1d4ed8', request: '#1e40af', cafe: '#f9a8d4' };

    const add = (source, target, value, id, colorL = P.general, colorR = P.detail) => {
      if (!value || value <= 0) return;
      const w = width(value);
      e.push({ id: id || `${source}-${target}`, source, target, type: 'taper', data: { value, colorL, colorR, widthStart: w, widthEnd: w, label: fmt(value) } });
    };
    // 일반 -> 카페 화살표, 카페 -> 브랜드 화살표 (값 라벨 없음)
    e.push({ id: 'general-cafe', source: 'general', target: 'cafe', type: 'arrow', data: { color: P.cafe, width: 1, label: '' } });
    e.push({ id: 'cafe-brand', source: 'cafe', target: 'brand', type: 'arrow', sourceHandle: 'top', data: { color: P.cafe, width: 1, label: '' } });
    // brand/general -> homepage/blog
    add('brand', 'homepage', data.brand_to_site_direct || 0, 'brand-home', P.brand, P.homepage);
    add('brand', 'blog', data.brand_to_blog_direct || 0, 'brand-blog', P.brand, P.blog);
    add('general', 'homepage', data.general_to_site_direct || 0, 'general-home', P.general, P.homepage);
    add('general', 'blog', data.general_to_blog_direct || 0, 'general-blog', P.general, P.blog);
    // maplist/ad/home/blog -> detail
    add('maplist', 'detail', data.place_list_to_detail || 0, 'maplist-detail', P.maplist, P.detail);
    add('ad', 'detail', data.place_ad_to_detail || 0, 'ad-detail', P.ad, P.detail);
    add('homepage', 'detail', data.homepage_to_place_detail || 0, 'home-detail', P.homepage, P.detail);
    add('blog', 'detail', data.blog_to_place_detail || 0, 'blog-detail', P.blog, P.detail);
    // detail -> booking -> request
    add('detail', 'booking', data.place_to_booking_page || 0, 'detail-booking', P.detail, P.booking);
    add('booking', 'request', data.booking_page_to_requests || 0, 'booking-request', P.booking, P.request);
    return e;
  }, [data]);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  useEffect(() => setEdges(initialEdges), [initialEdges, setEdges]);

  return (
    <div style={{ width: '100%', height: '420px' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        edgeTypes={{ taper: TaperEdge, arrow: ArrowEdge }}
        fitView
        nodesDraggable
        panOnDrag
        zoomOnScroll
      >
        <MiniMap zoomable pannable />
        <Controls showInteractive={true} />
        <Background gap={20} color="#eee" />
      </ReactFlow>
    </div>
  );
}


