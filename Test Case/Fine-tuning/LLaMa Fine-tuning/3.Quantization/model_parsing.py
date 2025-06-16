
import time
from huggingface_hub import HfApi
from dotenv import load_dotenv
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.cuda.empty_cache()
torch.cuda.synchronize()

# 메모리 상태 확인
allocated = torch.cuda.memory_allocated() / 1024**3
reserved = torch.cuda.memory_reserved() / 1024**3
print(f"GPU 메모리: {allocated:.1f}GB 사용 중, {reserved:.1f}GB 예약됨")

def section_by_section(input_text: str, tokenizer, model) -> str:
    print("🚀 섹션별 생성 시작...")
    
    sections = []
    
    # 각 섹션별 프롬프트
    section_prompts = [
        {
            "name": "프로젝트 상세 정보",
            "prompt": f"{input_text}\n\n위 프로젝트에 대해 다음 형식으로 프로젝트 상세 정보만 생성해주세요:\n**프로젝트 상세 정보:**\n{{완전한 딕셔너리 형태}}"
        },
        {
            "name": "관계 데이터", 
            "prompt": f"{input_text}\n\n위 프로젝트에 대해 데이터베이스 관계 데이터만 생성해주세요:\n**관계 데이터:**\n[완전한 리스트 형태]"
        },
        {
            "name": "ERD 데이터",
            "prompt": f"{input_text}\n\n위 프로젝트에 대해 ERD 데이터만 생성해주세요:\n**ERD 데이터:**\n[완전한 리스트 형태]"
        },
        {
            "name": "API 명세",
            "prompt": f"{input_text}\n\n위 프로젝트에 대해 API 명세 데이터만 생성해주세요:\n**API 명세 데이터:**\n{{완전한 딕셔너리 형태}}"
        }
    ]
    
    for i, section in enumerate(section_prompts):
        start_time = time.time()
        print(f"[{i+1}/4] {section['name']} 생성 중...")
        
        inputs = tokenizer(section['prompt'], return_tensors='pt', truncation=True, max_length=512)
        eos_token_id = tokenizer.convert_tokens_to_ids('<|eot_id|>')
        
        with torch.no_grad():
          outputs = model.generate(
              input_ids=inputs["input_ids"].to('cuda'),
              max_new_tokens=2048,        # 3072 → 2048 (적당히 줄이기)
              do_sample=False,            # True → False (그리디, 가장 빠름)
              use_cache=True,             # 캐시 사용
              num_beams=1,               # 빔 서치 끄기
              early_stopping=False,
              pad_token_id=tokenizer.eos_token_id,
              eos_token_id=None
          )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        section_content = response[len(section['prompt']):].strip()
        sections.append(section_content)
        
        elapsed = time.time() - start_time
        print(f"    ✅ 완료: {len(section_content)} 문자, {elapsed:.1f}초")
    
    # 모든 섹션 결합
    full_response = input_text + "\n\nAssistant: " + "\n\n".join(sections)
    print(f"🎉 전체 완료! 총 {len(full_response)} 문자 생성")
    return full_response

api = HfApi()
username="Min-kyu"
MODEL_NAME = "PJA_LLM_MODEL_8bit"

load_dotenv()

api_key = os.getenv('HUG_API_KEY')

# 학습된 모델
tokenizer = AutoTokenizer.from_pretrained(f"{username}/{MODEL_NAME}")
model = AutoModelForCausalLM.from_pretrained(f"{username}/{MODEL_NAME}")

# 추론
input_text = """
이 프로젝트는 3인칭 카드게임기반 MMORPG 게임 개발 프로젝트입니다. 창업을 목적으로 게임 데이터를 끌어다가 프로젝트를 하려고합니다. 주제는 간단하게 3인칭 카드게임을 mmorpg형식으로 만드려고 합니다. 전체적으로 어떻게 만들생각이냐면 시간이 10초 지날 때마다 하나씩 카드가 드롭되게 하는 형식으로 게임을 만들고싶습니다. 또한, 스토리가 있었으면 좋겠고 선택하는 스토리라인에 따라서 드롭되는 카드의 형식이 달랐으면 좋겠습니다. 이 게임을 통해서 사용자가 랜덤 가챠 + 순간적인 판단으로 컨트롤하는 능력이 늘었으면 좋겠다는 생각으로 프로젝트를 기획하였습니다.
"""

inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=512)
eos_token_id = tokenizer.convert_tokens_to_ids('<|eot_id|>')

with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs["input_ids"].to('cuda'),
        max_new_tokens=500,
        eos_token_id=eos_token_id,
        do_sample=True,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response.replace("', '", "',\n'").replace("}, {", "},\n{"))