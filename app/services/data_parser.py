from bs4 import BeautifulSoup, Tag
import json
import os
from blog_automation.config import DATA_PROCESSED_DIR

def _get_section(soup: BeautifulSoup, title: str):
    """Helper function to find a section container by its title."""
    spans = soup.find_all('span', class_='text-cs-s-dark')
    for span in spans:
        if title in span.get_text(strip=True):
            # Find the main container of the section
            parent = span.find_parent('div', class_='bg-white')
            if parent:
                return parent
    return None

def _safe_get_text(element: Tag, selector: str, default=None):
    """Safely get text from an element."""
    try:
        return element.select_one(selector).text.strip()
    except (AttributeError, IndexError):
        return default

def _get_table_data(table: Tag):
    """Extracts data from a table into a list of lists."""
    data = []
    if not table:
        return data
    rows = table.find_all('tr')
    for row in rows:
        cols = [ele.text.strip() for ele in row.find_all('td')]
        if cols:
            data.append(cols)
    return data

def parse_html_file(html_file_path: str):
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'lxml')
    data = {}

    # 1. 노출 점수
    exposure_score_section = _get_section(soup, '노출 점수')
    if exposure_score_section:
        score_table = _get_table_data(exposure_score_section.select_one('.table-keyword'))
        if len(score_table) > 0 and len(score_table[0]) >= 4:
            data['posting_score'] = score_table[0][1]
            data['exposure_score'] = score_table[0][3]
        else:
            data['posting_score'] = None
            data['exposure_score'] = None
    else:
        data['posting_score'] = None
        data['exposure_score'] = None


    # 2. 포스팅 점수
    posting_score_section = _get_section(soup, '포스팅 점수')
    if posting_score_section:
        table = posting_score_section.select_one('.table-keyword')
        posting_scores = []
        if table:
            rows = table.select('tr')
            for row in rows:
                cols = row.select('td')
                if len(cols) >= 3:
                    posting_scores.append({
                        'category': cols[0].text.strip().split('\n')[0],
                        'score': cols[1].text.strip(),
                        'evaluation': cols[2].text.strip(),
                        'percentage': row.select_one('.progress-bar span').text.strip() if row.select_one('.progress-bar span') else None
                    })
        data['posting_score_details'] = posting_scores if posting_scores else None
    else:
        data['posting_score_details'] = None

    # 3. 노출 평가
    exposure_eval_section = _get_section(soup, '노출 평가')
    if exposure_eval_section:
        eval_box = exposure_eval_section.select_one('.border.py-3')
        if eval_box:
            publish_time_p = eval_box.find('p', string=lambda t: t and '발행시간' in t)
            data['publish_time'] = publish_time_p.text.split(':')[1].strip() if publish_time_p else None
            status_p = eval_box.select_one('p.text-danger')
            data['exposure_status'] = status_p.text.strip() if status_p else None
        else:
            data['publish_time'] = None
            data['exposure_status'] = None
    else:
        data['publish_time'] = None
        data['exposure_status'] = None

    # 4. 포스팅 분석
    posting_analysis_section = _get_section(soup, '포스팅 분석')
    if posting_analysis_section:
        analysis_tables = posting_analysis_section.select('.table-bordered')
        if len(analysis_tables) >= 2:
            char_table = _get_table_data(analysis_tables[0])
            if char_table and len(char_table[0]) >= 4:
                data['text_analysis'] = {
                    'title_with_space': char_table[0][0],
                    'title_without_space': char_table[0][1],
                    'content_with_space': char_table[0][2],
                    'content_without_space': char_table[0][3]
                }
            else:
                 data['text_analysis'] = None

            morpheme_table = _get_table_data(analysis_tables[1])
            if morpheme_table and len(morpheme_table[0]) >= 4:
                data['morpheme_counts'] = {
                    'morpheme_count': morpheme_table[0][0],
                    'syllable_count': morpheme_table[0][1],
                    'word_count': morpheme_table[0][2],
                    'abusing_word_count': morpheme_table[0][3]
                }
            else:
                data['morpheme_counts'] = None
        else:
            data['text_analysis'] = None
            data['morpheme_counts'] = None
    else:
        data['text_analysis'] = None
        data['morpheme_counts'] = None
        
    # 5. 이미지 분석
    image_analysis_section = _get_section(soup, '이미지 분석')
    if image_analysis_section:
        image_table = _get_table_data(image_analysis_section.select_one('table'))
        if image_table and len(image_table[0]) >= 5:
            data['image_analysis'] = {
                'valid_images': image_table[0][0],
                'invalid_images': image_table[0][1],
                'material_images': image_table[0][2],
                'uploaded_images': image_table[0][3],
                'content_images': image_table[0][4],
            }
        else:
            data['image_analysis'] = None

        invalid_image_urls = [img['data-image'] for img in soup.select('#invaildImageBox img') if img.has_attr('data-image')]
        if data.get('image_analysis'):
            data['image_analysis']['invalid_image_urls'] = invalid_image_urls
    else:
        data['image_analysis'] = None
        
    # 6. 포스팅 제목 및 본문
    data['post_title'] = _safe_get_text(soup, '#post_diagnosis_result_title')
    data['post_content'] = _safe_get_text(soup, '#post_diagnosis_result_content')

    # 7. 어뷰징 단어
    data['abusing_words'] = {
        'adult_words': _safe_get_text(soup, '#adult-word-last'),
        'harmful_words': _safe_get_text(soup, '#harmful-word-last'),
        'commercial_words': _safe_get_text(soup, '#commercial-word-last'),
        'excessive_words': _safe_get_text(soup, '#advertising-word-last')
    }

    # 8. 카테고리 분석
    category_section = _get_section(soup, '카테고리 분석')
    if category_section:
        category_box = category_section.select('#category-diagnosis-box ul')
        categories = []
        for i, ul in enumerate(category_box):
            category_name_tag = ul.select_one('[class*="border-category-"]')
            percentage_tag = ul.select_one('.text-primary')
            keywords_tag = soup.select_one(f'#category_data_{i}')
            
            if category_name_tag and percentage_tag and keywords_tag:
                categories.append({
                    'category_name': category_name_tag.text.strip(),
                    'percentage': percentage_tag.text.strip(),
                    'keywords': keywords_tag.text.strip()
                })
        data['category_analysis'] = categories if categories else None
    else:
        data['category_analysis'] = None

    # 9. 형태소 분석
    morpheme_section = _get_section(soup, '형태소 분석')
    if morpheme_section:
        morpheme_box = morpheme_section.select('#morpheme-diagnosis-box ul')
        morphemes = []
        for i, ul in enumerate(morpheme_box):
            morpheme_type_tag = ul.select_one('[class*="border-morpheme-"]')
            percentage_tag = ul.select_one('.text-primary')
            words_tag = soup.select_one(f'#morpheme_data_{i}')

            if morpheme_type_tag and percentage_tag and words_tag:
                morphemes.append({
                    'morpheme_type': morpheme_type_tag.text.strip(),
                    'percentage': percentage_tag.text.strip(),
                    'words': words_tag.text.strip()
                })
        data['morpheme_analysis_details'] = morphemes if morphemes else None
    else:
        data['morpheme_analysis_details'] = None

    # 10. 단어 분석
    word_analysis_section = _get_section(soup, '단어 분석')
    if word_analysis_section:
        total_words = _safe_get_text(word_analysis_section, '.bg-primary.bg-opacity-25 span')
        word_list = word_analysis_section.select('.sub-list li')
        words = []
        if len(word_list) > 1:
            for i in range(0, len(word_list), 2):
                word = word_list[i].text.strip()
                count = word_list[i+1].text.strip()
                if word:
                    words.append({'word': word, 'count': count})
        data['word_analysis'] = {
            'total_words': total_words,
            'words': words
        } if words else None
    else:
        data['word_analysis'] = None

    # 11. 키워드 추출
    keyword_section = _get_section(soup, '키워드 추출')
    if keyword_section:
        keyword_table = keyword_section.select_one('.table-keyword tbody')
        keywords = []
        if keyword_table:
            rows = keyword_table.select('tr')
            for row in rows:
                cells = row.select('td')
                if len(cells) >= 11:
                    keywords.append({
                        'rank': cells[0].text.strip(),
                        'keyword': cells[1].text.strip(),
                        'search_volume_total': cells[2].text.strip(),
                        'search_volume_pc': cells[3].text.strip(),
                        'search_volume_mobile': cells[4].text.strip(),
                        'clicks_pc': cells[5].text.strip(),
                        'clicks_mobile': cells[6].text.strip(),
                        'ctr_pc': cells[7].text.strip(),
                        'ctr_mobile': cells[8].text.strip(),
                        'competition': cells[9].text.strip(),
                        'ad_count': cells[10].text.strip()
                    })
        data['extracted_keywords'] = keywords if keywords else None
    else:
        data['extracted_keywords'] = None
        
    # 12. 동일 주제 포스팅
    same_topic_section = _get_section(soup, '동일 주제 포스팅')
    if same_topic_section:
        data['same_topic_posting'] = _safe_get_text(same_topic_section, '.border.border-1')
    else:
        data['same_topic_posting'] = None

    # Save processed data to a JSON file
    file_name = os.path.basename(html_file_path).replace('.html', '.json')
    json_path = os.path.join(DATA_PROCESSED_DIR, file_name)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
    return json_path 