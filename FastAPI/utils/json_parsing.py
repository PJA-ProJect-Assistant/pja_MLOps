# utils/json_parsing.py
import json
import ast
import re
import logging

# 로거 설정
logger = logging.getLogger(__name__)

def clean_escaped_quotes(text):
    """이스케이프된 따옴표를 정리하는 함수"""
    
    if not isinstance(text, str):
        return str(text)  # 문자열이 아니면 그대로 반환

    # 방법 1: 정규식으로 \' -> ' 변환
    cleaned = re.sub(r"\\'", "'", text)
    # 방법 2: 정규식으로 \" -> " 변환
    cleaned = re.sub(r'\\"', '"', cleaned)
    
    return str(cleaned)

def clean_backslashes(text):
    """백슬래시를 완전히 제거하는 함수"""
    
    if not isinstance(text, str):
        return str(text)
    
    # 모든 백슬래시 제거 (이스케이프 문자 포함)
    cleaned = text.replace('\\', '')
    
    return cleaned

def extract_json_from_response(content):
    """응답에서 순수 JSON만 추출하는 함수"""
    
    if not isinstance(content, str):
        return str(content)
    
    # 마크다운 코드 블록 제거
    content = re.sub(r'```json\s*', '', content)
    content = re.sub(r'```\s*', '', content)
    
    # 앞뒤 공백 및 불필요한 텍스트 제거
    content = content.strip()
    
    # JSON 시작점 찾기 (배열 또는 객체)
    json_start_array = content.find('[')
    json_start_object = content.find('{')
    
    # 가장 먼저 나오는 JSON 구조 선택
    if json_start_array == -1 and json_start_object == -1:
        logger.warning("JSON 시작점을 찾을 수 없습니다")
        return content
    
    if json_start_array == -1:
        json_start = json_start_object
        start_char = '{'
        end_char = '}'
    elif json_start_object == -1:
        json_start = json_start_array
        start_char = '['
        end_char = ']'
    else:
        json_start = min(json_start_array, json_start_object)
        start_char = content[json_start]
        end_char = '}' if start_char == '{' else ']'
    
    # JSON 끝점 찾기 (중괄호/대괄호 균형 맞추기)
    brace_count = 0
    json_end = -1
    
    for i in range(json_start, len(content)):
        if content[i] == start_char:
            brace_count += 1
        elif content[i] == end_char:
            brace_count -= 1
            if brace_count == 0:
                json_end = i + 1
                break
    
    if json_end == -1:
        logger.warning("JSON 끝점을 찾을 수 없습니다")
        return content
    
    return content[json_start:json_end]

def safe_parse_requirements(text):
    """안전하게 requirements 문자열을 파싱하는 함수"""

    # 이미 딕셔너리나 리스트인 경우 그대로 반환
    if isinstance(text, (dict, list)):
        return str(text)
    
    # 문자열이 아닌 경우 처리
    if not isinstance(text, str):
        logger.warning(f"예상치 못한 타입: {type(text)}")
        return str(text)
    
    try:
        # 1단계: 이스케이프된 따옴표 정리
        cleaned = clean_escaped_quotes(text)
        
        # 2단계: ast.literal_eval로 파싱 시도
        parsed = ast.literal_eval(cleaned)
        
        return str(parsed)
    
    except (ValueError, SyntaxError) as e:
        logger.warning(f"ast.literal_eval 실패: {e}")
        
        try:
            # 3단계: 추가 정리 후 재시도
            # 바깥쪽 따옴표 제거 (만약 있다면)
            if cleaned.startswith("'") and cleaned.endswith("'"):
                cleaned = cleaned[1:-1]
            elif cleaned.startswith('"') and cleaned.endswith('"'):
                cleaned = cleaned[1:-1]
            
            # 다시 파싱 시도
            parsed = ast.literal_eval(cleaned)
            return str(parsed)
            
        except Exception as e2:
            logger.error(f"2차 파싱도 실패: {e2}")
            return None

def safe_parse_json(content):
    """안전하게 JSON 문자열을 파싱하는 함수 (범용)"""
    
    if not isinstance(content, str):
        logger.warning(f"JSON 파싱 대상이 문자열이 아닙니다: {type(content)}")
        return None
    
    try:
        # 1단계: 순수 JSON 추출
        clean_content = extract_json_from_response(content)
        
        # 2단계: 백슬래시 정리
        clean_content = clean_escaped_quotes(clean_content)
        
        # 3단계: JSON 파싱 시도
        json_data = json.loads(clean_content)
        return json_data
        
    except json.JSONDecodeError as e:
        logger.warning(f"JSON 파싱 실패, 백슬래시 제거 후 재시도: {e}")
        
        try:
            # 4단계: 백슬래시 완전 제거 후 재시도
            clean_content = clean_backslashes(content)
            clean_content = extract_json_from_response(clean_content)
            json_data = json.loads(clean_content)
            return json_data
            
        except json.JSONDecodeError as e2:
            logger.warning(f"백슬래시 제거 후에도 JSON 파싱 실패: {e2}")
            
            try:
                # 5단계: ast.literal_eval 시도
                clean_content = extract_json_from_response(content)
                clean_content = clean_escaped_quotes(clean_content)
                parsed_data = ast.literal_eval(clean_content)
                return parsed_data
                
            except (ValueError, SyntaxError) as e3:
                logger.error(f"ast.literal_eval도 실패: {e3}")
                return None
    
    except Exception as e:
        logger.error(f"예상치 못한 오류: {e}")
        return None

def validate_json_structure(data, required_keys=None):
    """JSON 구조가 올바른지 검증하는 함수"""
    
    if not isinstance(data, (dict, list)):
        logger.error("데이터가 딕셔너리나 리스트가 아닙니다")
        return False
    
    # 백슬래시 포함 여부 검증
    json_str = json.dumps(data, ensure_ascii=False)
    if '\\' in json_str:
        logger.warning("JSON에 백슬래시가 포함되어 있습니다")
        return False
    
    # 필수 키 검증 (딕셔너리인 경우)
    if required_keys and isinstance(data, dict):
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            logger.error(f"필수 키가 누락되었습니다: {missing_keys}")
            return False
    
    return True

def clean_and_parse_response(content, response_type="json"):
    """
    AI 응답을 정리하고 파싱하는 통합 함수
    
    Args:
        content (str): AI 모델의 원본 응답
        response_type (str): "json", "requirements", "list" 등 응답 타입
    
    Returns:
        parsed_data: 파싱된 데이터 또는 None
    """
    
    if not content:
        logger.error("빈 응답입니다")
        return None
    
    logger.info(f"응답 타입: {response_type}, 원본 길이: {len(content)}")
    
    if response_type == "requirements":
        return safe_parse_requirements(content)
    elif response_type in ["json", "dict", "object"]:
        return safe_parse_json(content)
    elif response_type == "list":
        parsed = safe_parse_json(content)
        if isinstance(parsed, list):
            return parsed
        else:
            logger.warning("리스트 타입이 아닙니다")
            return None
    else:
        # 기본적으로 JSON 파싱 시도
        return safe_parse_json(content)

# 사용 예시 및 테스트 함수들
def test_parsing_functions():
    """파싱 함수들을 테스트하는 함수"""
    
    # 테스트 케이스 1: 정상적인 JSON
    test1 = '{"name": "test", "value": 123}'
    result1 = safe_parse_json(test1)
    print(f"테스트 1 결과: {result1}")
    
    # 테스트 케이스 2: 마크다운 블록이 포함된 JSON
    test2 = '```json\n{"name": "test", "value": 123}\n```'
    result2 = safe_parse_json(test2)
    print(f"테스트 2 결과: {result2}")
    
    # 테스트 케이스 3: 백슬래시가 포함된 JSON
    test3 = r'{"name": "test\'s", "path": "/api/users"}'
    result3 = safe_parse_json(test3)
    print(f"테스트 3 결과: {result3}")
    
    # 테스트 케이스 4: Requirements 형태
    test4 = "[{'requirementType': 'FUNCTIONAL', 'content': 'test content'}]"
    result4 = safe_parse_requirements(test4)
    print(f"테스트 4 결과: {result4}")

if __name__ == "__main__":
    # 테스트 실행
    test_parsing_functions()