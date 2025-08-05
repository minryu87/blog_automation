import json
import re
import time
import aiohttp
import asyncio

from config import (
    naver_map_url_base, naver_map_update_headers, n_m_get_place_info_params, n_m_get_post_json,
    initial_value_score, initial_value_isinsulting, initial_value_isdefamatory, initial_value_postWordAddReqList,
    crawling_success_return, crawling_fail_return,
    NAVER_CRAWLING_SIZE, NAVER_INITIAL_CRAWLING_TOTAL_SIZE,
    CRAWLING_SLEEP_TIME, WAIT_TIME_FOR_TOO_MANY_REQUESTS, MAX_CRAWLING_RETRY_COUNT,
    random_ua, REVIEW_SEARCH_DEADLINE_PERIOD_MONTHS
)

import reputation_dto as rep_dto
import agents.crawling_utils as crawl_utils
from agents.aws.lambda_proxy_request import ProxyRequest

from logger import setup_logger
logger = setup_logger(__name__)

# 기본 설정값으로 초기화
proxy_request = ProxyRequest(requester="naver_map_place_crawler")

# 또는 커스텀 설정으로 초기화
# proxy_request = ProxyRequest(
#     enable_lambda=True,
#     lambda_ratio=0.9,
#     lambda_endpoints=["https://custom-lambda.amazonaws.com/proxy"]
# )

async def naver_map_crawling(channel_reputation_dto, app_user_id, is_last, complete_json):
    '''
    네이버 맵 크롤링 메인 함수
    '''
    # Unpack five values from get_map_info
    channel_id, store_name, store_detail, channel_place_id, store_channel_recent_crawling_at = crawl_utils.get_map_info(channel_reputation_dto)
    store_channel_recent_crawling_at = crawl_utils.set_initial_crawling_date(store_channel_recent_crawling_at)

    logger.info('[[[[NAVER MAP CRAWLING]]]] channel_id: %s, store_name: %s, store_detail: %s, channel_place_id: %s, store_channel_recent_crawling_at: %s',
                channel_id, store_name, store_detail, channel_place_id, store_channel_recent_crawling_at)

    async with aiohttp.ClientSession() as session:
        if channel_place_id:
            logger.info('[naver_map_place_crawling] Using provided channel_place_id By Kotlin Backend: %s', channel_place_id)
        else:
            # 2. 데이터 가져오기
            place_list = await fetch_naver_place_list(session, store_name)
            
            if place_list is None:
                logger.error('[naver_map_place_crawling] None Place List: %s, store_name: %s, store_detail: %s', place_list, store_name, store_detail)
                await crawl_utils.send_fail_message(app_user_id, channel_id, store_name, store_detail, complete_json, is_last, 'naver_place_list_none')
                return crawling_fail_return('Noneplace')

            # 3. 데이터 유효성 검사 및 선택
            req_list = await validate_naver_place_list(place_list, store_detail)
            logger.debug('req_list: %s', req_list)
            if req_list is None:
                await crawl_utils.send_fail_message(app_user_id, channel_id, store_name, store_detail, complete_json, is_last, 'naver_validate_place_list_none')
                return crawling_fail_return('Noneplace')

            channel_place_id = req_list['id'] # 이거 확인

        logger.info('[naver_map_place_crawling] channel_place_id: %s', channel_place_id)

        # 4. 크롤링 & 데이터 파싱
        new_post_list, new_post_reply_add_req_list = await parse_naver_place_reviews(
            channel_place_id, store_channel_recent_crawling_at, channel_id, store_name, store_detail, app_user_id, complete_json
        )
        logger.debug('new_post_list: %s', new_post_list)
        logger.debug('new_post_reply_add_req_list: %s', new_post_reply_add_req_list)

        # 5. 최종 데이터 전송
        await crawl_utils.send_success_messages(app_user_id, channel_id, store_name, store_detail, new_post_list, new_post_reply_add_req_list, complete_json, is_last)
        logger.info('[naver_map_place_crawling] final send_success_messages - app_user_id: %s, channel_id: %s, store_name: %s, new_post_list: %s, new_post_reply_add_req_list: %s, is_last: %s', 
                    app_user_id, channel_id, store_name, len(new_post_list), len(new_post_reply_add_req_list), is_last)
        
    return crawling_success_return


async def fetch_naver_place_list(session, store_name):
    """
    네이버에서 장소 목록을 가져오는 함수
    """
    path = '/p/api/search/allSearch'
    place_info_params = n_m_get_place_info_params(store_name)

    async with session.get(f'{naver_map_url_base}{path}', params=place_info_params, headers={**naver_map_update_headers, **{'referer': 'https://map.naver.com/p/search/'}}) as req:
        try:
            place_result = await req.json()

            result_data = place_result.get('result', {})
            place = result_data.get('place') if result_data else {}
            place_list = place.get('list') if place else None

            return place_list
        except Exception as e:
            text = await req.text()
            logger.error('[naver_map_place_crawling/fetch_naver_place_list] Naver Map Search Error - text: %s, error: %s', text, e)
            return None


async def validate_naver_place_list(place_list, store_detail):
    """
    장소 목록에서 유효한 장소를 검증하고 선택하는 함수
    """
    for req_list_data in place_list:
        # 지번 주소 포함 여부 확인 - address는 지번, roadAddress는 도로명
        if store_detail in req_list_data['address']:
            return req_list_data
    return None


async def parse_naver_place_reviews(channel_place_id, store_channel_recent_crawling_at, channel_id, store_name, store_detail, app_user_id, complete_json):
    """
    리뷰를 파싱하고 가공하는 함수
    """
    new_post_list = []
    new_post_reply_add_req_list = []
    store_channel_recent_crawling_at = crawl_utils.normalize_epoch_time_by_unit(store_channel_recent_crawling_at, 'day')
    review_search_deadline = crawl_utils.calculate_date(store_channel_recent_crawling_at, months=REVIEW_SEARCH_DEADLINE_PERIOD_MONTHS) # 백엔드 요청시간에서 1달 전 에폭타임 반환

    idx = 0
    break_check = False

    # 크롤링시작 시간 기록
    crawling_start_time = time.time()

    crawling_retry_count = 0

    while True:
        idx += 1
        post_json = n_m_get_post_json(channel_place_id, idx, NAVER_CRAWLING_SIZE)
        
        try:
            status, response_data, response_headers = await proxy_request.request(
                method='POST',
                url='https://pcmap-api.place.naver.com/graphql',
                json_data=post_json,
                headers={**naver_map_update_headers, **random_ua},
                response_json=False
            )

            logger.info('[naver_map_place_crawling/parse_naver_place_reviews] status: %s, response_data: %s, response_headers: %s, store_name: %s, channel_id: %s',
                        status, len(response_data), response_headers.keys(), store_name, channel_id)
            
            # HTTP 상태 코드 체크
            if status == 429:  # Too Many Requests
                logger.error('[naver_map_place_crawling/parse_naver_place_reviews] Too Many Requests detected - backing off, store_name: %s, store_detail: %s, channel_id: %s',
                            store_name, store_detail, channel_id)
                
                if crawling_retry_count >= MAX_CRAWLING_RETRY_COUNT:
                    logger.error('[naver_map_place_crawling/parse_naver_place_reviews] MAX_CRAWLING_RETRY_COUNT Retry count exceeded, crawling_retry_count: %s, store_name: %s, store_detail: %s, channel_id: %s',
                                crawling_retry_count, store_name, store_detail, channel_id)
                    break
                
                idx -= 1
                crawling_retry_count += 1
                logger.info('[naver_map_place_crawling/parse_naver_place_reviews] Waiting for %s seconds, idx: %s, crawling_retry_count: %s',
                            WAIT_TIME_FOR_TOO_MANY_REQUESTS, idx, crawling_retry_count)

                await asyncio.sleep(WAIT_TIME_FOR_TOO_MANY_REQUESTS)
                continue
            
            elif status != 200:  # 기타 HTTP 오류
                logger.error('[naver_map_place_crawling/parse_naver_place_reviews] HTTP error: %s, store_name: %s, channel_id: %s',
                            status, store_name, channel_id)
                break

            if response_data is None:
                logger.error('[naver_map_place_crawling/parse_naver_place_reviews] Response Text is None, store_name: %s, channel_id: %s',
                             store_name, channel_id)
                break
            # try:
            #     req_text = response_data
            # except Exception as e:
            #     logger.error('[naver_map_place_crawling] Error processing response: %s, store_name: %s, channel_id: %s',
            #                 e, store_name, channel_id)
            #     break
                
        except Exception as e:
            logger.error('[naver_map_place_crawling/parse_naver_place_reviews] Network error: %s, store_name: %s, channel_id: %s',
                        e, store_name, channel_id)
            break

        logger.debug('[naver_map_place_crawling/parse_naver_place_reviews] response_data: %s, store_name: %s, channel_id: %s', type(response_data), store_name, channel_id)

        code_str = ''

        try:
            code_str = re.search(r'{\"data\":{"visitorReviews":[\d\D]*?(?=}}})}}}', response_data).group()
        except Exception as e:
            # Too Many Requests 응답 처리
            if (re.search(r'429', response_data) or 
                re.search(r'Too many requests', response_data, re.IGNORECASE)):
                logger.error('[naver_map_place_crawling/parse_naver_place_reviews] Too Many Requests detected - backing off, store_name: %s, store_detail: %s, channel_id: %s, req_text: %s',
                             store_name, store_detail, channel_id, response_data)

                if crawling_retry_count >= MAX_CRAWLING_RETRY_COUNT:
                    logger.error('[naver_map_place_crawling/parse_naver_place_reviews] MAX_CRAWLING_RETRY_COUNT Retry count exceeded, crawling_retry_count: %s, store_name: %s, store_detail: %s, channel_id: %s, response_data: %s',
                                 crawling_retry_count, store_name, store_detail, channel_id, response_data)
                    break

                idx -= 1  # 현재 인덱스를 다시 시도하기 위해 감소
                crawling_retry_count += 1
                logger.info('[naver_map_place_crawling/parse_naver_place_reviews] Waiting for %s seconds, idx: %s, crawling_retry_count: %s', WAIT_TIME_FOR_TOO_MANY_REQUESTS, idx, crawling_retry_count)

                await asyncio.sleep(WAIT_TIME_FOR_TOO_MANY_REQUESTS)
                continue

            logger.error('[naver_map_place_crawling/parse_naver_place_reviews] Response Text Parsing Error: %s, store_name: %s, channel_id: %s',
                          e, store_name, channel_id)
            break

        logger.debug('[naver_map_place_crawling/parse_naver_place_reviews] code_str: %s, store_name: %s, channel_id: %s', type(code_str), store_name, channel_id)

        json_list = json.loads(code_str)['data']['visitorReviews']['items']

        logger.debug('[naver_map_place_crawling/parse_naver_place_reviews] json_list: %s, store_name: %s, channel_id: %s', json_list, store_name, channel_id)

        if not json_list:
            break_check = True

        # 크롤링 데이터 처리 시작
        for json_data in json_list:
            try:
                if json_data is None or json_data == {}:
                    continue

                # 1. 기본 데이터 추출 및 유효성 검사
                review_visited_at = crawl_utils.fix_date(json_data.get('visited', ''))
                review_created_at = crawl_utils.fix_date(json_data.get('created', ''))
                review_status = json_data.get('status', '')
                review_body = json_data.get('body', '')
                review_id = json_data.get('id', '')

                # 2. 조기 종료 조건 검사
                # 2.1 데드라인 체크
                if review_visited_at < review_search_deadline:
                    logger.info('[naver_map_place_crawling/parse_naver_place_reviews] 크롤링 PROCESS 종료 END : 리뷰 아이디 : %s, 리뷰 내용 : %s, 리뷰 방문일 : %s, 리뷰 생성일 : %s, 리뷰 검색 데드라인 : %s, 리뷰 방문일 < 리뷰 검색 데드라인 : %s',
                              review_id, review_body[:20], review_visited_at, review_created_at, review_search_deadline, review_visited_at < review_search_deadline)
                    break_check = True
                    break

                # 2.2 리뷰 상태 및 내용 검사
                if review_status != 'ACTIVE' or not review_body.strip():
                    continue

                logger.info('[naver_map_place_crawling/parse_naver_place_reviews] 리뷰 아이디 : %s, 리뷰 내용 : %s, 리뷰 방문일 : %s, 리뷰 생성일 : %s, 리뷰 검색 데드라인 : %s, 리뷰 방문일 >= 리뷰 검색 데드라인 : %s',
                          review_id, review_body[:20], review_visited_at, review_created_at, review_search_deadline, review_visited_at >= review_search_deadline)

                # 3. 리뷰 기본 정보 설정
                review_post_id = f'NAVER_MAP_{review_id}'
                review_author_info = json_data.get('author', {})
                review_author_object_id = review_author_info.get('objectId', '')
                review_author_nickname = review_author_info.get('nickname', '')
                review_author_link = f'https://m.place.naver.com/my/{review_author_object_id}/reviewfeed?sort=VISIT_DATE_TIME_DESC&mediaFilter=ALL&reviewId={review_id}&v=2'

                logger.info('[naver_map_place_crawling/parse_naver_place_reviews] 리뷰 생성일: %s, 마지막 크롤링 성공일: %s, 리뷰 생성일 >= 마지막 크롤링 성공일: %s', review_created_at, store_channel_recent_crawling_at, review_created_at >= store_channel_recent_crawling_at)

                # 4. 답글 정보 처리
                reply_info = json_data.get('reply', {})
                reply_date = reply_info.get('date', '0')
                reply_date = '0' if reply_date is None else reply_date
                reply_created_at = crawl_utils.fix_date(reply_date)
                
                review_reply = None
                if reply_info.get('body'):
                    review_reply = {
                        "reply": reply_info.get('body', ''),
                        "author": reply_info.get('replyTitle', ''),
                        "authorDtm": reply_created_at
                    }
                    logger.info('[naver_map_place_crawling/parse_naver_place_reviews] 리뷰 답글 생성일: %s, 마지막 크롤링 성공일: %s, 리뷰 답글 생성일 >= 마지막 크롤링 성공일: %s', reply_created_at, store_channel_recent_crawling_at, reply_created_at >= store_channel_recent_crawling_at)

                # 5. 새 리뷰 처리
                if review_created_at >= store_channel_recent_crawling_at:
                    post_create_req = rep_dto.PostCreateReq(
                        postId=review_post_id,
                        score=initial_value_score,
                        title="",
                        content=review_body,
                        channelId=channel_id,
                        storeName=store_name,
                        storeDetail=store_detail,
                        author=review_author_nickname,
                        authorLink=review_author_link,
                        authorDtm=review_created_at,
                        type="REVIEW",
                        captureImg="",
                        isInsulting=initial_value_isinsulting,
                        isDefamatory=initial_value_isdefamatory,
                        postWordAddReqList=initial_value_postWordAddReqList,
                    )
                    logger.info('[naver_map_place_crawling/parse_naver_place_reviews] new_post_list 데이터 추가: 리뷰 내용 = %s, 매장명 = %s, 매장 상세 = %s, 리뷰 작성자 = %s, 리뷰 아이디 = %s, 리뷰 생성일 = %s, 마지막 크롤링 성공일 = %s', 
                              review_body[:20], store_name, store_detail, review_author_nickname, review_post_id, review_created_at, store_channel_recent_crawling_at)
                    new_post_list.append(post_create_req.model_dump())

                # 6. 기존 리뷰의 새 답글 처리
                if reply_info.get('body') and reply_created_at >= store_channel_recent_crawling_at:
                    review_reply = {
                        "postId": review_post_id,
                        "reply": reply_info.get('body', ''),
                        "author": reply_info.get('replyTitle', ''),
                        "authorDtm": reply_created_at
                    }
                    logger.info('[naver_map_place_crawling/parse_naver_place_reviews] new_post_reply_add_req_list 데이터 추가: 리뷰 내용 = %s, 리뷰 아이디 = %s, 리뷰 작성자 = %s, 리뷰 답글 생성일 = %s, 마지막 크롤링 성공일 = %s', 
                              review_reply['reply'][:20], review_reply['postId'], review_reply['author'], review_reply['authorDtm'], store_channel_recent_crawling_at)
                    new_post_reply_add_req_list.append(review_reply)

            except Exception as e:
                logger.exception('[naver_map_place_crawling/parse_naver_place_reviews] Error: %s, store_name: %s, channel_id: %s', e, store_name, channel_id)
                break

        # 두번의 리스트 호출 후 보냄
        if (idx % 2) == 0:
            if len(new_post_list) != 0:
                await crawl_utils.send_success_messages(app_user_id, channel_id, store_name, store_detail, new_post_list, new_post_reply_add_req_list, complete_json, False)
                logger.info('[naver_map_place_crawling/parse_naver_place_reviews] intermediate send_success_messages - app_user_id: %s, channel_id: %s, store_name: %s, new_post_list: %s, new_post_reply_add_req_list: %s',
                            app_user_id, channel_id, store_name, len(new_post_list), len(new_post_reply_add_req_list))

                new_post_list.clear()
                new_post_reply_add_req_list.clear()
                
            await asyncio.sleep(CRAWLING_SLEEP_TIME)

        # 초기 크롤링 크기 만큼 크롤링 후 종료. 1년 이상 등의 조건은 추후 필요성을 보고 추가 예정
        if idx * NAVER_CRAWLING_SIZE >= NAVER_INITIAL_CRAWLING_TOTAL_SIZE:
            logger.info('[naver_map_place_crawling/parse_naver_place_reviews] Meet initial crawling total size: %s, idx: %s, store_name: %s, channel_id: %s', idx * NAVER_CRAWLING_SIZE, idx, store_name, channel_id)
            break

        if break_check:
            break

    # 크롤링 소요 시간 기록
    crawling_end_time = time.time()
    crawling_time = crawling_end_time - crawling_start_time
    logger.info('[naver_map_place_crawling/parse_naver_place_reviews] Crawling Time: %s seconds, store_name: %s, channel_id: %s', crawling_time, store_name, channel_id)

    return new_post_list, new_post_reply_add_req_list