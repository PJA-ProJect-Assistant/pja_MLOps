"""
3개 모델 성능 비교: 파인튜닝 vs GPT-4o-mini vs GPT-4o
파인튜닝 효과를 명확하게 측정하기 위한 완전한 비교
"""

import os
import json
import time
import pandas as pd
from tqdm import tqdm
import openai
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
import numpy as np
import ast
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ========================================================================================
# 전역 설정 및 상수 정의
# ========================================================================================

# 테스트할 3개 모델 정의
FINETUNED_MODEL = "ft:gpt-4o-mini-2024-07-18:test:pja-erd-finetuning-model:BmOgyrDW:ckpt-step-124"
BASELINE_MINI_MODEL = "gpt-4o-mini"  # 파인튜닝 베이스 모델
BASELINE_4O_MODEL = "gpt-4o"         # 플래그십 모델

# API 호출 설정
TEMPERATURE = 0.2
CSV_FILE_PATH = "hehe.csv"

# 시스템 프롬프트
SYSTEM_PROMPT = """당신은 숙련된 데이터베이스 설계 전문가입니다. 
사용자의 프로젝트 설명과 요구사항을 바탕으로 ERD(Entity Relationship Diagram) 데이터를 생성해주세요.
각 테이블의 구조, 컬럼 정보, 관계를 명확하게 정의해야 합니다.
JSON 형태로 구조화된 ERD 정보를 제공해주세요."""

# 의미적 유사도 계산을 위한 모델
try:
    semantic_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    SEMANTIC_SIMILARITY_AVAILABLE = True
except:
    print("⚠️ 의미적 유사도 모델을 로드할 수 없습니다. 해당 메트릭은 제외됩니다.")
    SEMANTIC_SIMILARITY_AVAILABLE = False

# ========================================================================================
# 유틸리티 함수
# ========================================================================================

def safe_parse_json_string(json_str):
    """JSON 문자열을 안전하게 파싱하는 함수"""
    if pd.isna(json_str) or json_str == '':
        return None
    
    try:
        if isinstance(json_str, (dict, list)):
            return json_str
        
        if isinstance(json_str, str):
            try:
                return ast.literal_eval(json_str)
            except:
                return json.loads(json_str)
    except Exception as e:
        print(f"JSON 파싱 오류: {e}")
        return json_str

def format_data_for_prompt(data):
    """데이터를 프롬프트에 적합한 형태로 포맷팅"""
    if isinstance(data, str):
        return data
    elif isinstance(data, (dict, list)):
        return json.dumps(data, ensure_ascii=False, indent=2)
    else:
        return str(data)

def extract_text_content(data):
    """JSON 데이터에서 텍스트 내용을 추출하여 비교 가능한 형태로 변환"""
    if isinstance(data, str):
        return data.strip()
    elif isinstance(data, dict):
        text_parts = []
        for key, value in data.items():
            if isinstance(value, str):
                text_parts.append(f"{key}: {value}")
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        text_parts.append(item)
                    elif isinstance(item, dict):
                        text_parts.append(json.dumps(item, ensure_ascii=False))
        return " ".join(text_parts)
    elif isinstance(data, list):
        text_parts = []
        for item in data:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                text_parts.append(json.dumps(item, ensure_ascii=False))
        return " ".join(text_parts)
    else:
        return str(data)

# ========================================================================================
# 핵심 함수 정의
# ========================================================================================

def openai_chat_completion(model, user_input, project_info):
    """OpenAI Chat Completion API를 호출하는 함수"""
    try:
        formatted_project_info = format_data_for_prompt(project_info)
        formatted_user_input = format_data_for_prompt(user_input)
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"""
프로젝트 정보:
{formatted_project_info}

요구사항:
{formatted_user_input}

위 정보를 바탕으로 ERD 데이터를 생성해주세요.
테이블 구조, 컬럼 정보, 관계를 포함한 JSON 형태로 제공해주세요.
"""}
        ]
        
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=2000
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"API 호출 오류 ({model}): {e}")
        return ""

def calculate_individual_metrics(pred_text, ref_text):
    """
    개별 샘플에 대한 메트릭을 계산하는 함수
    """
    try:
        # 텍스트 정리
        pred_clean = extract_text_content(pred_text).strip()
        ref_clean = extract_text_content(ref_text).strip()
        
        if not pred_clean or not ref_clean:
            return {
                "BLEU": 0.0,
                "ROUGE-L": 0.0,
                "의미적_유사도": 0.0
            }
        
        # 1. BLEU 점수 (개별 샘플용)
        try:
            bleu_score = corpus_bleu([pred_clean], [[ref_clean]]).score
        except:
            bleu_score = 0.0
        
        # 2. ROUGE-L 점수
        try:
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            rouge_score = scorer.score(ref_clean, pred_clean)['rougeL'].fmeasure * 100
        except:
            rouge_score = 0.0
        
        # 3. 의미적 유사도
        try:
            if SEMANTIC_SIMILARITY_AVAILABLE:
                pred_embedding = semantic_model.encode([pred_clean])
                ref_embedding = semantic_model.encode([ref_clean])
                similarity = cosine_similarity(pred_embedding, ref_embedding)[0][0] * 100
            else:
                similarity = 0.0
        except:
            similarity = 0.0
        
        return {
            "BLEU": round(bleu_score, 2),
            "ROUGE-L": round(rouge_score, 2),
            "의미적_유사도": round(similarity, 2)
        }
    
    except Exception as e:
        print(f"개별 메트릭 계산 오류: {e}")
        return {
            "BLEU": 0.0,
            "ROUGE-L": 0.0,
            "의미적_유사도": 0.0
        }
    """의미적 유사도를 계산하는 함수"""
        
def calculate_semantic_similarity(predictions, references):
    """
    의미적 유사도를 계산하는 함수
    """
    if not SEMANTIC_SIMILARITY_AVAILABLE:
        return 0.0
    
    try:
        # 길이 맞추기 (안전장치)
        min_length = min(len(predictions), len(references))
        predictions = predictions[:min_length]
        references = references[:min_length]
        
        pred_texts = [extract_text_content(pred) for pred in predictions]
        ref_texts = [extract_text_content(ref) for ref in references]
        
        # 빈 텍스트 필터링
        valid_pairs = [(p, r) for p, r in zip(pred_texts, ref_texts) if p.strip() and r.strip()]
        
        if not valid_pairs:
            return 0.0
        
        valid_preds, valid_refs = zip(*valid_pairs)
        
        pred_embeddings = semantic_model.encode(list(valid_preds))
        ref_embeddings = semantic_model.encode(list(valid_refs))
        
        similarities = []
        for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
            similarity = cosine_similarity([pred_emb], [ref_emb])[0][0]
            similarities.append(similarity)
        
        return np.mean(similarities) * 100
    except Exception as e:
        print(f"의미적 유사도 계산 오류: {e}")
        print(f"predictions 길이: {len(predictions) if 'predictions' in locals() else 'N/A'}")
        print(f"references 길이: {len(references) if 'references' in locals() else 'N/A'}")
        return 0.0
    """
    각 샘플별 상세 성능 분석 데이터프레임 생성
    """
    print("\n📊 샘플별 상세 성능 분석 데이터프레임 생성 중...")
    
    # 기본 정보 컬럼들
    detailed_data = []
    
    model_names = {
        FINETUNED_MODEL: "파인튜닝",
        BASELINE_MINI_MODEL: "4o-mini", 
        BASELINE_4O_MODEL: "4o"
    }
    
    for i in range(len(references)):
        row_data = {
            'sample_id': i + 1,
            'user_input_length': len(str(test_df.iloc[i]['user_input'])),
            'reference_length': len(str(references[i])),
        }
        
        # 각 모델별 성능 계산
        for model_id, model_name in model_names.items():
            if i < len(results[model_id]):
                prediction = results[model_id][i]
                reference = references[i]
                
                # 개별 메트릭 계산
                metrics = calculate_individual_metrics(prediction, reference)
                
                # 컬럼명에 모델명 추가
                row_data[f'{model_name}_BLEU'] = metrics['BLEU']
                row_data[f'{model_name}_ROUGE-L'] = metrics['ROUGE-L']
                row_data[f'{model_name}_의미적유사도'] = metrics['의미적_유사도']
                row_data[f'{model_name}_output_length'] = len(str(prediction))
        
        # 모델 간 성능 차이 계산
        if '파인튜닝_BLEU' in row_data and '4o-mini_BLEU' in row_data:
            row_data['파인튜닝_vs_mini_BLEU'] = row_data['파인튜닝_BLEU'] - row_data['4o-mini_BLEU']
            row_data['파인튜닝_vs_mini_ROUGE'] = row_data['파인튜닝_ROUGE-L'] - row_data['4o-mini_ROUGE-L']
            row_data['파인튜닝_vs_mini_의미적'] = row_data['파인튜닝_의미적유사도'] - row_data['4o-mini_의미적유사도']
        
        if '파인튜닝_BLEU' in row_data and '4o_BLEU' in row_data:
            row_data['파인튜닝_vs_4o_BLEU'] = row_data['파인튜닝_BLEU'] - row_data['4o_BLEU']
            row_data['파인튜닝_vs_4o_ROUGE'] = row_data['파인튜닝_ROUGE-L'] - row_data['4o_ROUGE-L']
            row_data['파인튜닝_vs_4o_의미적'] = row_data['파인튜닝_의미적유사도'] - row_data['4o_의미적유사도']
        
        detailed_data.append(row_data)
    
    df_detailed = pd.DataFrame(detailed_data)
    return df_detailed

def analyze_dataframe_insights(df_detailed):
    """
    데이터프레임에서 인사이트 추출 및 분석
    """
    print("\n🔍 데이터프레임 인사이트 분석")
    print("=" * 80)
    
    # 1. 기본 통계
    print("📈 기본 통계:")
    print(f"  총 샘플 수: {len(df_detailed)}")
    print(f"  평균 입력 길이: {df_detailed['user_input_length'].mean():.0f}자")
    print(f"  평균 정답 길이: {df_detailed['reference_length'].mean():.0f}자")
    
    # 2. 모델별 평균 성능
    print(f"\n📊 모델별 평균 성능:")
    models = ['파인튜닝', '4o-mini', '4o']
    metrics = ['BLEU', 'ROUGE-L', '의미적유사도']
    
    for model in models:
        print(f"\n  🤖 {model}:")
        for metric in metrics:
            col_name = f'{model}_{metric}'
            if col_name in df_detailed.columns:
                avg_score = df_detailed[col_name].mean()
                std_score = df_detailed[col_name].std()
                print(f"    {metric}: {avg_score:.2f} (±{std_score:.2f})")
    
    # 3. 파인튜닝이 우수한 샘플 비율
    print(f"\n🏆 파인튜닝 우수 샘플 분석:")
    
    if '파인튜닝_vs_mini_BLEU' in df_detailed.columns:
        mini_wins = {
            'BLEU': (df_detailed['파인튜닝_vs_mini_BLEU'] > 0).sum(),
            'ROUGE-L': (df_detailed['파인튜닝_vs_mini_ROUGE'] > 0).sum(),
            '의미적유사도': (df_detailed['파인튜닝_vs_mini_의미적'] > 0).sum()
        }
        
        print(f"  vs GPT-4o-mini:")
        for metric, wins in mini_wins.items():
            win_rate = (wins / len(df_detailed)) * 100
            print(f"    {metric} 우수: {wins}/{len(df_detailed)} ({win_rate:.1f}%)")
    
    if '파인튜닝_vs_4o_BLEU' in df_detailed.columns:
        gpt4o_wins = {
            'BLEU': (df_detailed['파인튜닝_vs_4o_BLEU'] > 0).sum(),
            'ROUGE-L': (df_detailed['파인튜닝_vs_4o_ROUGE'] > 0).sum(),
            '의미적유사도': (df_detailed['파인튜닝_vs_4o_의미적'] > 0).sum()
        }
        
        print(f"  vs GPT-4o:")
        for metric, wins in gpt4o_wins.items():
            win_rate = (wins / len(df_detailed)) * 100
            print(f"    {metric} 우수: {wins}/{len(df_detailed)} ({win_rate:.1f}%)")
    
    # 4. 최고/최저 성능 샘플
    print(f"\n🎯 극값 분석:")
    
    if '파인튜닝_BLEU' in df_detailed.columns:
        # 파인튜닝 모델 최고 성능
        best_idx = df_detailed['파인튜닝_BLEU'].idxmax()
        worst_idx = df_detailed['파인튜닝_BLEU'].idxmin()
        
        print(f"  파인튜닝 BLEU 최고: 샘플 {best_idx+1} ({df_detailed.loc[best_idx, '파인튜닝_BLEU']:.2f}점)")
        print(f"  파인튜닝 BLEU 최저: 샘플 {worst_idx+1} ({df_detailed.loc[worst_idx, '파인튜닝_BLEU']:.2f}점)")
    
    # 5. 성능 분포 분석
    print(f"\n📈 성능 분포 분석:")
    if '파인튜닝_BLEU' in df_detailed.columns:
        bleu_ranges = {
            '90-100': (df_detailed['파인튜닝_BLEU'] >= 90).sum(),
            '80-89': ((df_detailed['파인튜닝_BLEU'] >= 80) & (df_detailed['파인튜닝_BLEU'] < 90)).sum(),
            '70-79': ((df_detailed['파인튜닝_BLEU'] >= 70) & (df_detailed['파인튜닝_BLEU'] < 80)).sum(),
            '60-69': ((df_detailed['파인튜닝_BLEU'] >= 60) & (df_detailed['파인튜닝_BLEU'] < 70)).sum(),
            '50-59': ((df_detailed['파인튜닝_BLEU'] >= 50) & (df_detailed['파인튜닝_BLEU'] < 60)).sum(),
            '0-49': (df_detailed['파인튜닝_BLEU'] < 50).sum()
        }
        
        print(f"  파인튜닝 BLEU 점수 분포:")
        for range_name, count in bleu_ranges.items():
            percentage = (count / len(df_detailed)) * 100
            print(f"    {range_name}점: {count}개 ({percentage:.1f}%)")
    
    return df_detailed

def calculate_metrics(predictions, references):
    """모델 성능을 평가하는 다양한 메트릭을 계산하는 함수"""
    pred_texts = [extract_text_content(pred) for pred in predictions]
    ref_texts = [extract_text_content(ref) for ref in references]
    
    valid_pairs = [(p, r) for p, r in zip(pred_texts, ref_texts) if p.strip() and r.strip()]
    
    if not valid_pairs:
        return {
            "BLEU": 0.0,
            "ROUGE-L": 0.0,
            "정확일치율(%)": 0.0,
            "의미적_유사도(%)": 0.0
        }
    
    valid_preds, valid_refs = zip(*valid_pairs)
    
    # BLEU 점수 계산
    try:
        bleu_score = corpus_bleu(list(valid_preds), [list(valid_refs)]).score
    except:
        bleu_score = 0.0
    
    # ROUGE-L 점수 계산
    try:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_scores = [scorer.score(ref, pred)['rougeL'].fmeasure 
                       for ref, pred in zip(valid_refs, valid_preds)]
        rouge_l_score = np.mean(rouge_scores) * 100
    except:
        rouge_l_score = 0.0
    
    # 정확 일치율 계산
    exact_matches = sum(1 for p, r in zip(valid_preds, valid_refs) 
                       if p.strip() == r.strip())
    exact_match_rate = (exact_matches / len(valid_pairs)) * 100
    
    # 의미적 유사도 계산
    semantic_similarity = calculate_semantic_similarity(predictions, references)
    
    return {
        "BLEU": round(bleu_score, 2),
        "ROUGE-L": round(rouge_l_score, 2),
        "정확일치율(%)": round(exact_match_rate, 2),
        "의미적_유사도(%)": round(semantic_similarity, 2)
    }

def run_three_model_evaluation():
    """3개 모델 비교 평가 실행 함수"""
    print("📊 3개 모델 성능 비교: 파인튜닝 vs GPT-4o-mini vs GPT-4o")
    print("=" * 80)
    
    # ========================================================================================
    # 1. 데이터 로드 및 검증
    # ========================================================================================
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        print(f"✅ 데이터 로드 완료: {len(df)}개 샘플")
        print(f"📋 CSV 컬럼: {list(df.columns)}")
            
    except Exception as e:
        print(f"❌ CSV 파일 로드 실패: {e}")
        return
    
    # 데이터 전처리
    print("\n🔄 데이터 전처리 중...")
    for col in ['user_input', 'project_info', 'total_requirements', 'ERD_data']:
        if col in df.columns:
            df[col] = df[col].apply(safe_parse_json_string)
    
    # ========================================================================================
    # 2. 3개 모델 추론 실행 (전체 데이터)
    # ========================================================================================
    
    # 전체 데이터 사용 (사용자 선택 가능)
    use_full_data = input("전체 데이터를 사용하시겠습니까? (y/n): ").lower().strip()
    
    if use_full_data == 'y':
        test_df = df.copy()
        print(f"\n🔄 전체 {len(test_df)}개 샘플에 대해 3개 모델 추론 시작...")
        print(f"⚠️ 예상 소요 시간: 약 {len(test_df) * 0.5}분 (API 호출 지연 포함)")
        print(f"💰 예상 비용: 약 ${len(test_df) * 0.03:.2f} (GPT-4o 기준)")
        
        proceed = input("계속 진행하시겠습니까? (y/n): ").lower().strip()
        if proceed != 'y':
            print("테스트가 취소되었습니다.")
            return
    else:
        test_samples = min(10, len(df))  # 기본값을 10개로 증가
        test_df = df.head(test_samples).copy()
        print(f"\n🔄 {test_samples}개 샘플에 대해 3개 모델 추론 시작...")
    
    # 진행 상황 체크포인트 저장 기능 추가
    checkpoint_file = "evaluation_checkpoint.json"
    
    # 기존 체크포인트 로드 시도
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            if len(checkpoint_data.get('completed_indices', [])) > 0:
                print(f"📋 기존 진행상황 발견: {len(checkpoint_data['completed_indices'])}개 샘플 완료")
                resume = input("이어서 진행하시겠습니까? (y/n): ").lower().strip()
                
                if resume == 'y':
                    completed_indices = set(checkpoint_data['completed_indices'])
                    results = checkpoint_data.get('results', {
                        FINETUNED_MODEL: [],
                        BASELINE_MINI_MODEL: [],
                        BASELINE_4O_MODEL: []
                    })
                else:
                    completed_indices = set()
                    results = {
                        FINETUNED_MODEL: [],
                        BASELINE_MINI_MODEL: [],
                        BASELINE_4O_MODEL: []
                    }
            else:
                completed_indices = set()
                results = {
                    FINETUNED_MODEL: [],
                    BASELINE_MINI_MODEL: [],
                    BASELINE_4O_MODEL: []
                }
        except:
            completed_indices = set()
            results = {
                FINETUNED_MODEL: [],
                BASELINE_MINI_MODEL: [],
                BASELINE_4O_MODEL: []
            }
    else:
        completed_indices = set()
        results = {
            FINETUNED_MODEL: [],
            BASELINE_MINI_MODEL: [],
            BASELINE_4O_MODEL: []
        }
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="모델 추론 진행"):
        # 이미 완료된 샘플은 건너뛰기
        if idx in completed_indices:
            continue
            
        user_input = row['user_input']
        project_info = row['project_info']
        
        print(f"\n📝 샘플 {idx+1}/{len(test_df)} 처리 중...")
        
        try:
            # 1. 파인튜닝 모델 추론
            print(f"   🎯 파인튜닝 모델 추론...")
            ft_result = openai_chat_completion(FINETUNED_MODEL, user_input, project_info)
            
            # 2. GPT-4o-mini 원본 추론
            print(f"   🤖 GPT-4o-mini 추론...")
            mini_result = openai_chat_completion(BASELINE_MINI_MODEL, user_input, project_info)
            
            # 3. GPT-4o 추론
            print(f"   🔥 GPT-4o 추론...")
            gpt4o_result = openai_chat_completion(BASELINE_4O_MODEL, user_input, project_info)
            
            # 결과 저장
            results[FINETUNED_MODEL].append(ft_result)
            results[BASELINE_MINI_MODEL].append(mini_result)
            results[BASELINE_4O_MODEL].append(gpt4o_result)
            
            # 완료된 인덱스 추가
            completed_indices.add(idx)
            
            # 체크포인트 저장 (10개마다)
            if len(completed_indices) % 10 == 0:
                checkpoint_data = {
                    'completed_indices': list(completed_indices),
                    'results': results
                }
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
                print(f"   💾 체크포인트 저장됨: {len(completed_indices)}개 완료")
            
            # API 레이트 리미트 방지
            time.sleep(1.0)
            
            # 진행상황 출력
            if len(completed_indices) % 5 == 0:
                remaining = len(test_df) - len(completed_indices)
                eta_minutes = remaining * 0.5
                print(f"   ⏱️ 진행률: {len(completed_indices)}/{len(test_df)} ({len(completed_indices)/len(test_df)*100:.1f}%) - 예상 잔여시간: {eta_minutes:.1f}분")
                
        except Exception as e:
            print(f"   ❌ 샘플 {idx+1} 처리 중 오류: {e}")
            print(f"   ⏭️ 다음 샘플로 건너뜁니다...")
            continue
    
    # ========================================================================================
    # 3. 성능 메트릭 계산 및 비교
    # ========================================================================================
    
    print("\n📈 3개 모델 성능 평가 결과")
    print("=" * 80)
    
    # 결과 길이 맞추기 (실패한 샘플 제외)
    min_length = min(len(results[model]) for model in results.keys())
    for model in results.keys():
        results[model] = results[model][:min_length]
    
    # 해당하는 정답 데이터도 맞춰서 자르기
    references = test_df['ERD_data'].tolist()[:min_length]
    test_df_matched = test_df.head(min_length).copy()
    
    print(f"📊 실제 분석 대상: {min_length}개 샘플")
    
    # ========================================================================================
    # 3.1 상세 데이터프레임 생성
    # ========================================================================================
    
    df_detailed = create_detailed_dataframe(test_df_matched, results, references)
    
    # 상세 데이터프레임 저장
    try:
        df_detailed.to_csv("샘플별_상세_성능분석.csv", index=False, encoding="utf-8")
        print(f"💾 상세 분석 결과가 '샘플별_상세_성능분석.csv'로 저장되었습니다.")
        
        # 엑셀 파일로도 저장 (더 보기 좋음)
        df_detailed.to_excel("샘플별_상세_성능분석.xlsx", index=False)
        print(f"📊 Excel 파일도 '샘플별_상세_성능분석.xlsx'로 저장되었습니다.")
    except Exception as e:
        print(f"⚠️ 상세 분석 파일 저장 실패: {e}")
    
    # 데이터프레임 인사이트 분석
    df_detailed = analyze_dataframe_insights(df_detailed)
    
    # ========================================================================================
    # 3.2 전체 평균 성능 계산
    # ========================================================================================
    
    # 최종 체크포인트 저장
    if completed_indices:
        checkpoint_data = {
            'completed_indices': list(completed_indices),
            'results': results
        }
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 전체 추론 완료: {len(completed_indices)}/{len(test_df)} 샘플")
    
    # 결과 길이 맞추기 (실패한 샘플 제외)
    min_length = min(len(results[model]) for model in results.keys())
    for model in results.keys():
        results[model] = results[model][:min_length]
    
    # 해당하는 정답 데이터도 맞춰서 자르기
    references = test_df['ERD_data'].tolist()[:min_length]
    
    # 모델별 성능 계산 (전체 평균)
    final_results = {}
    model_names = {
        FINETUNED_MODEL: "파인튜닝_모델",
        BASELINE_MINI_MODEL: "GPT-4o-mini", 
        BASELINE_4O_MODEL: "GPT-4o"
    }
    
    for model_id, predictions in results.items():
        model_name = model_names[model_id]
        
        print(f"\n🔍 {model_name} 출력 샘플 확인:")
        for i, pred in enumerate(predictions[:2]):
            print(f"  출력 {i+1}: {str(pred)[:150]}...")
        
        metrics = calculate_metrics(predictions, references)
        
        print(f"\n🤖 {model_name} 전체 평균 성능:")
        for metric_name, value in metrics.items():
            print(f"   {metric_name}: {value}")
        
        final_results[model_name] = metrics
    
    # ========================================================================================
    # 4. 종합 비교 분석
    # ========================================================================================
    
    print(f"\n📊 종합 성능 비교 분석")
    print("=" * 80)
    
    # 성능 비교 표 출력
    metrics_list = ["BLEU", "ROUGE-L", "의미적_유사도(%)"]
    
    print(f"{'메트릭':<15} {'파인튜닝':<12} {'4o-mini':<12} {'GPT-4o':<12} {'vs mini':<12} {'vs 4o':<12}")
    print("-" * 80)
    
    for metric in metrics_list:
        ft_score = final_results["파인튜닝_모델"][metric]
        mini_score = final_results["GPT-4o-mini"][metric]
        gpt4o_score = final_results["GPT-4o"][metric]
        
        vs_mini = ft_score - mini_score
        vs_4o = ft_score - gpt4o_score
        
        print(f"{metric:<15} {ft_score:<12.2f} {mini_score:<12.2f} {gpt4o_score:<12.2f} {vs_mini:<+12.2f} {vs_4o:<+12.2f}")
    
    # ========================================================================================
    # 5. 파인튜닝 효과 분석
    # ========================================================================================
    
    print(f"\n🎯 파인튜닝 효과 분석")
    print("=" * 80)
    
    ft_metrics = final_results["파인튜닝_모델"]
    mini_metrics = final_results["GPT-4o-mini"]
    gpt4o_metrics = final_results["GPT-4o"]
    
    # 베이스 모델(4o-mini) 대비 개선도
    print("📈 베이스 모델(GPT-4o-mini) 대비 파인튜닝 개선도:")
    mini_improvements = 0
    for metric in metrics_list:
        diff = ft_metrics[metric] - mini_metrics[metric]
        improvement_rate = (diff / mini_metrics[metric]) * 100 if mini_metrics[metric] != 0 else 0
        print(f"  {metric}: {diff:+.2f} ({improvement_rate:+.1f}%)")
        if diff > 0:
            mini_improvements += 1
    
    # 플래그십 모델(GPT-4o) 대비 성능
    print(f"\n🔥 플래그십 모델(GPT-4o) 대비 파인튜닝 성능:")
    gpt4o_wins = 0
    for metric in metrics_list:
        diff = ft_metrics[metric] - gpt4o_metrics[metric]
        print(f"  {metric}: {diff:+.2f} ({'우수' if diff > 0 else '열세'})")
        if diff > 0:
            gpt4o_wins += 1
    
    # ========================================================================================
    # 6. 결론 및 권장사항
    # ========================================================================================
    
    print(f"\n🏆 최종 결론")
    print("=" * 80)
    
    if mini_improvements >= 2:
        print("✅ 파인튜닝이 베이스 모델 대비 명확한 성능 향상을 보여줍니다!")
    else:
        print("⚠️ 파인튜닝 효과가 제한적입니다.")
    
    if gpt4o_wins >= 2:
        print("🔥 파인튜닝 모델이 플래그십 모델도 능가하는 놀라운 성과입니다!")
    elif gpt4o_wins >= 1:
        print("👍 파인튜닝 모델이 일부 지표에서 플래그십 모델과 경쟁합니다!")
    else:
        print("📝 플래그십 모델 대비로는 아직 개선 여지가 있습니다.")
    
    # 비용 효율성 분석
    print(f"\n💰 비용 효율성 분석:")
    samples_processed = len(completed_indices)
    print(f"  처리된 샘플: {samples_processed}개")
    print(f"  파인튜닝 모델 비용: GPT-4o의 ~10% (대폭 절약)")
    print(f"  추론 속도: GPT-4o보다 빠름")
    if gpt4o_wins >= 1:
        print(f"  성능: 일부 지표에서 GPT-4o 수준 또는 그 이상")
        print(f"  → 🎯 매우 높은 ROI!")
    
    # 체크포인트 파일 정리
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"\n🧹 임시 체크포인트 파일이 정리되었습니다.")
    
    # ========================================================================================
    # 7. 결과 저장 (상세 분석 포함)
    # ========================================================================================
    
    try:
        # 전체 평균 결과 저장
        with open("3모델_성능비교_결과.json", "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 전체 평균 결과가 '3모델_성능비교_결과.json' 파일로 저장되었습니다.")
        
        # 상세 출력 결과 저장
        output_df = test_df_matched.copy()
        output_df['파인튜닝_모델_출력'] = results[FINETUNED_MODEL]
        output_df['GPT4o_mini_출력'] = results[BASELINE_MINI_MODEL]
        output_df['GPT4o_출력'] = results[BASELINE_4O_MODEL]
        output_df.to_csv("3모델_전체_출력_비교.csv", index=False, encoding="utf-8")
        print(f"📄 전체 출력 비교가 '3모델_전체_출력_비교.csv'로 저장되었습니다.")
        
        # 요약 리포트 생성
        summary_report = {
            "실험_정보": {
                "총_샘플수": len(completed_indices),
                "성공_샘플수": min_length,
                "테스트_날짜": time.strftime("%Y-%m-%d %H:%M:%S"),
                "모델_정보": {
                    "파인튜닝": FINETUNED_MODEL,
                    "4o-mini": BASELINE_MINI_MODEL,
                    "4o": BASELINE_4O_MODEL
                }
            },
            "전체_평균_성능": final_results,
            "파인튜닝_개선도": {
                "vs_4o_mini": {
                    metric: final_results["파인튜닝_모델"][metric] - final_results["GPT-4o-mini"][metric]
                    for metric in ["BLEU", "ROUGE-L", "의미적_유사도(%)"]
                },
                "vs_4o": {
                    metric: final_results["파인튜닝_모델"][metric] - final_results["GPT-4o"][metric]
                    for metric in ["BLEU", "ROUGE-L", "의미적_유사도(%)"]
                }
            }
        }
        
        with open("실험_요약_리포트.json", "w", encoding="utf-8") as f:
            json.dump(summary_report, f, ensure_ascii=False, indent=2)
        print(f"📋 실험 요약 리포트가 '실험_요약_리포트.json'으로 저장되었습니다.")
        
    except Exception as e:
        print(f"❌ 결과 저장 실패: {e}")
    
    # ========================================================================================
    # 8. 최종 요약 출력
    # ========================================================================================
    
    print(f"\n🎊 최종 실험 요약")
    print("=" * 80)
    print(f"📊 분석 완료: {min_length}개 샘플")
    print(f"📁 생성된 파일:")
    print(f"   • 샘플별_상세_성능분석.csv/.xlsx - 각 샘플별 개별 점수")
    print(f"   • 3모델_성능비교_결과.json - 전체 평균 성능")
    print(f"   • 3모델_전체_출력_비교.csv - 모든 모델 출력 비교")
    print(f"   • 실험_요약_리포트.json - 종합 실험 리포트")
    
    # 핵심 성과 요약
    ft_bleu = final_results["파인튜닝_모델"]["BLEU"]
    mini_bleu = final_results["GPT-4o-mini"]["BLEU"]
    gpt4o_bleu = final_results["GPT-4o"]["BLEU"]
    
    print(f"\n🏆 핵심 성과:")
    print(f"   파인튜닝 BLEU: {ft_bleu:.2f}")
    print(f"   vs 4o-mini: +{ft_bleu - mini_bleu:.2f}점 ({((ft_bleu - mini_bleu)/mini_bleu*100):+.1f}%)")
    print(f"   vs GPT-4o: +{ft_bleu - gpt4o_bleu:.2f}점 ({((ft_bleu - gpt4o_bleu)/gpt4o_bleu*100):+.1f}%)")
    
    return df_detailed, final_results

# ========================================================================================
# 메인 실행 부분
# ========================================================================================

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY 환경변수를 설정해주세요.")
        exit(1)
    
    print("🚀 3개 모델 성능 비교 테스트를 시작합니다...")
    print(f"📁 테스트 데이터: {CSV_FILE_PATH}")
    print(f"🎯 파인튜닝 모델: {FINETUNED_MODEL}")
    print(f"🤖 GPT-4o-mini: {BASELINE_MINI_MODEL}")
    print(f"🔥 GPT-4o: {BASELINE_4O_MODEL}")
    print(f"🌡️ 온도 설정: {TEMPERATURE}")
    
    run_three_model_evaluation()
    
    print("\n✅ 3개 모델 성능 비교 테스트가 완료되었습니다!")