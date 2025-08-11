#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
단계별 인증/호출 테스트 스크립트
1) 인증값(Authorization/Cookie) 수동 입력
2) 토큰/쿠키 저장 및 확인
3) 추출 값으로 API 호출 성공 여부
"""

import sys
from pathlib import Path
import json
import time

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.util.config import get_config_manager
from scripts.crawler.naver_place_pv_auth_manager import NaverAuthManager
import requests


def stage_separator(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main():
    cm = get_config_manager()

    # 0) 클라이언트 선택 및 설정 확인
    stage_separator("0) 클라이언트 선택 및 설정 확인")
    client = cm.get_selected_client_config()
    if not client:
        print("❌ 클라이언트 설정을 찾을 수 없습니다. .env를 확인하세요.")
        sys.exit(1)

    auth_cfg = cm.get_auth_config()
    auth_file = (auth_cfg.auth_file_path_template or "{client_name}_auth.json").format(client_name=client.name)

    print(f"- 선택된 클라이언트: {client.name}")
    print(f"- NAVER_URL: {client.naver_url}")
    print(f"- 인증 파일: {auth_file}")

    # 1) 인증값 로드 (파일/환경변수 → 실패 시 프롬프트)
    stage_separator("1) 인증값 로드 (파일/환경변수 → 실패 시 프롬프트)")
    am = NaverAuthManager(
        naver_id=client.id,
        naver_password=client.pw,
        auth_file_path=auth_file,
        token_expiry_hours=auth_cfg.token_expiry_hours,
        naver_login_url=client.naver_url or "https://nid.naver.com/nidlogin.login",
        client_name=client.name,
    )

    # refresh_auth_if_needed는 저장파일 → .env AUTH/COOKIE → 프롬프트 순서로 시도
    if not am.refresh_auth_if_needed():
        print("❌ 인증값 로드/입력 실패")
        sys.exit(1)

    # 2) 토큰/쿠키 확인
    stage_separator("2) 토큰/쿠키 확인")
    token = am.auth_token
    cookies = am.cookies or {}

    token_masked = (token[:20] + "...") if token else None
    print(f"- 토큰 존재: {bool(token)} / 샘플: {token_masked}")
    print(f"- 쿠키 수: {len(cookies)} / 샘플 키: {list(cookies.keys())[:5]}")

    if not token or len(cookies) == 0:
        print("❌ 토큰/쿠키 확인 실패")
        sys.exit(1)

    # 3) 추출 값으로 API 호출
    stage_separator("3) 추출 값으로 API 호출")
    headers = {
        'Authorization': token,
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
        'Referer': 'https://new.smartplace.naver.com/'
    }
    params = {
        'dimensions': 'mapped_channel_name',
        'startDate': '2025-07-01',
        'endDate': '2025-07-01',
        'metrics': 'pv',
        'sort': 'pv',
        'useIndex': 'revenue-all-channel-detail'
    }

    try:
        resp = requests.get(
            'https://new.smartplace.naver.com/proxy/bizadvisor/api/v3/sites/sp_18123cf1aec42/report',
            params=params,
            headers=headers,
            cookies=cookies,
            timeout=20
        )
        print(f"- HTTP 상태: {resp.status_code}")
        if resp.status_code == 200:
            try:
                data = resp.json()
                print(f"✅ API 성공, 레코드 수: {len(data) if isinstance(data, list) else 'N/A'}")
            except Exception:
                print("⚠️ 본문 JSON 파싱 실패")
        else:
            print(f"❌ API 실패, 응답 본문 앞 300자:\n{resp.text[:300]}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ API 요청 오류: {e}")
        sys.exit(1)

    print("\n✨ 모든 단계 테스트 성공")


if __name__ == "__main__":
    main()
