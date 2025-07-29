import asyncio
import logging
import random

import aiohttp
import requests
from bs4 import BeautifulSoup

# --- 로깅 설정 ---
log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)


def scrape_free_proxy_list():
    """
    free-proxy-list.net/ko/ 에서 프록시 목록을 스크래핑합니다.
    :return: ["http://IP:PORT", ...] 형식의 프록시 주소 리스트
    """
    url = "https://free-proxy-list.net/ko/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')

        proxies = []
        table = soup.find('table', class_='table-striped')
        for row in table.tbody.find_all('tr'):
            cols = row.find_all('td')
            if len(cols) > 6 and cols[6].text.strip() == 'yes': # Https 지원 프록시만 선택
                ip = cols[0].text.strip()
                port = cols[1].text.strip()
                proxies.append(f"http://{ip}:{port}")
        return proxies
    except Exception as e:
        logging.error(f"프록시 목록 스크래핑 실패: {e}")
        return []


async def test_proxy(session, proxy):
    """
    단일 프록시가 작동하는지 테스트합니다.
    :param session: aiohttp.ClientSession
    :param proxy: 테스트할 프록시 주소
    :return: 작동하면 프록시 주소, 아니면 None
    """
    test_url = "https://www.naver.com"  # 네이버 접속 가능 여부로 테스트 기준 변경
    try:
        # 네이버는 User-Agent 헤더에 민감하므로 추가
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        async with session.get(test_url, proxy=proxy, timeout=8, headers=headers) as response:
            if response.status == 200:
                logging.info(f"프록시 작동 확인 (Naver 접속 가능): {proxy}")
                return proxy
            return None
    except Exception:
        # logging.debug(f"프록시 테스트 실패: {proxy}")
        return None


async def get_working_proxies(num_to_test=20):
    """
    스크래핑한 프록시 중 실제로 작동하는 목록을 비동기 병렬로 찾아 반환합니다.
    :param num_to_test: 테스트할 프록시 샘플 개수
    :return: ["http://IP:PORT", ...] 형식의 작동하는 프록시 주소 리스트
    """
    logging.info("새로운 프록시 목록을 스크래핑합니다...")
    scraped_proxies = scrape_free_proxy_list()
    if not scraped_proxies:
        logging.warning("스크래핑된 프록시가 없습니다.")
        return []

    logging.info(f"총 {len(scraped_proxies)}개의 프록시를 찾았습니다. 그 중 {num_to_test}개를 무작위로 테스트합니다.")
    proxies_to_test = random.sample(scraped_proxies, min(len(scraped_proxies), num_to_test))

    async with aiohttp.ClientSession() as session:
        tasks = [test_proxy(session, proxy) for proxy in proxies_to_test]
        results = await asyncio.gather(*tasks)

    working_proxies_with_prefix = [p for p in results if p is not None]
    logging.info(f"테스트한 {len(proxies_to_test)}개 중 {len(working_proxies_with_prefix)}개의 작동하는 프록시를 확보했습니다.")
    # None을 제외하고, 'http://' 접두사를 제거하여 반환
    # replace 대신 lstrip을 사용하여 더 안정적으로 접두사 제거
    working_proxies = [p.lstrip("http://") for p in working_proxies_with_prefix if p]

    return working_proxies


if __name__ == '__main__':
    # 이 파일을 직접 실행하면 프록시 테스트를 수행합니다.
    working_list = asyncio.run(get_working_proxies())
    print("\n--- 최종 작동 프록시 목록 ---")
    if working_list:
        for p in working_list:
            print(p)
    else:
        print("작동하는 프록시를 찾지 못했습니다.") 