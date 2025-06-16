"""
QLoRA 어댑터를 베이스 모델과 병합하여 배포용 모델을 생성하는 스크립트
파일명: merge_qlora.py

실행 방법:
python merge_qlora.py
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
import gc
import os
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

def merge_qlora_adapter(
    base_model_name: str,
    adapter_path: str,
    output_path: str,
    push_to_hub: bool = False,
    hub_model_name: str = None
):
    """
    QLoRA 어댑터를 베이스 모델과 병합
    
    Args:
        base_model_name: 베이스 모델 이름/경로
        adapter_path: LoRA 어댑터 경로
        output_path: 병합된 모델 저장 경로
        push_to_hub: Hugging Face Hub에 푸시 여부
        hub_model_name: Hub에 푸시할 모델 이름
    """
    
    print("🚀 QLoRA 어댑터 병합 시작...")
    
    # 메모리 정리
    torch.cuda.empty_cache()
    gc.collect()
    
    # 1. 베이스 모델 로드 (bfloat16으로 로드)
    print(f"📦 베이스 모델 로드: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # 2. 토크나이저 로드
    print("🔤 토크나이저 로드...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    
    # 3. LoRA 어댑터 로드
    print(f"🔗 LoRA 어댑터 로드: {adapter_path}")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch.bfloat16
    )
    
    # 4. 어댑터 병합
    print("🔄 어댑터 병합 중...")
    merged_model = model.merge_and_unload()
    
    # 5. 병합된 모델 저장
    print(f"💾 병합된 모델 저장: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    
    merged_model.save_pretrained(
        output_path,
        safe_serialization=True,  # SafeTensors 형식으로 저장
        max_shard_size="5GB"      # 큰 모델을 여러 파일로 분할
    )
    
    tokenizer.save_pretrained(output_path)
    
    # 6. 모델 카드 생성
    create_model_card(output_path, base_model_name, adapter_path)
    
    # 7. Hugging Face Hub에 푸시 (선택사항)
    if push_to_hub and hub_model_name:
        print(f"📤 Hugging Face Hub에 푸시: {hub_model_name}")
        
        # API 키 확인
        api_key = os.getenv('HUG_API_KEY')
        if api_key:
            login(token=api_key)
            
            merged_model.push_to_hub(
                hub_model_name,
                private=False,  # 공개 설정
                commit_message="Upload merged QLoRA model"
            )
            tokenizer.push_to_hub(
                hub_model_name,
                commit_message="Upload tokenizer"
            )
            print(f"✅ Hub 업로드 완료: https://huggingface.co/{hub_model_name}")
        else:
            print("⚠️  HUG_API_KEY가 설정되지 않아 Hub 업로드를 건너뜁니다.")
    
    # 8. 메모리 정리
    del merged_model, model, base_model
    torch.cuda.empty_cache()
    gc.collect()
    
    print("🎉 어댑터 병합 완료!")
    return output_path

def create_model_card(output_path: str, base_model: str, adapter_path: str):
    """모델 카드 생성"""
    model_card = f"""
---
license: apache-2.0
base_model: {base_model}
tags:
- llama
- qlora
- korean
- project-analysis
- fine-tuned
language:
- ko
- en
pipeline_tag: text-generation
---

# Korean Project Analysis Model (QLoRA Fine-tuned)

이 모델은 {base_model}을 QLoRA 방식으로 파인튜닝한 한국어 프로젝트 분석 전문 모델입니다.

## 모델 정보

- **베이스 모델**: {base_model}
- **파인튜닝 방법**: QLoRA (4-bit quantization + LoRA adapters)
- **어댑터 경로**: {adapter_path}
- **주요 기능**: 프로젝트 아이디어 분석, ERD 설계, API 명세 생성

## 사용법

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "your-username/your-model-name"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 추론 예시
messages = [
    {{"role": "system", "content": "당신은 프로젝트 분석 전문가입니다..."}},
    {{"role": "user", "content": "온라인 학습 플랫폼을 개발하고 싶습니다."}}
]

input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=4096,
        temperature=0.1,
        do_sample=True
    )

response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(response)
```

## 출력 형식

모델은 다음과 같은 구조화된 형식으로 응답합니다:

1. **프로젝트 상세 정보**: 종합 분석 및 기술 스택
2. **관계 데이터**: 데이터베이스 관계 정의
3. **ERD 데이터**: 완전한 스키마 설계
4. **API 명세 데이터**: OpenAPI 3.0 표준 문서

## 성능 특징

- QLoRA 기반으로 메모리 효율적
- 한국어 프로젝트 분석에 특화
- 실무 수준의 기술 문서 생성
- JSON 형식의 구조화된 출력

## 라이선스

Apache 2.0 License

## 학습 데이터

프로젝트 아이디어와 상응하는 기술 문서 쌍으로 구성된 데이터셋으로 학습되었습니다.
"""

    with open(os.path.join(output_path, "README.md"), "w", encoding="utf-8") as f:
        f.write(model_card.strip())

def test_merged_model(model_path: str, test_prompt: str = None):
    """병합된 모델 테스트"""
    print(f"🧪 병합된 모델 테스트: {model_path}")
    
    # 모델 및 토크나이저 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 기본 테스트 프롬프트
    if test_prompt is None:
        test_prompt = "모바일 앱 기반 배달 서비스를 개발하고 싶습니다. 실시간 주문 처리와 GPS 추적 기능이 포함되어야 합니다."
    
    messages = [
        {
            "role": "system", 
            "content": "당신은 프로젝트 아이디어를 체계적으로 분석하고 구조화하여 구체적인 개발 계획을 제시하는 전문 AI 어시스턴트입니다."
        },
        {
            "role": "user", 
            "content": test_prompt
        }
    ]
    
    # 추론 실행
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    print("🤖 생성 중...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    print(f"📝 입력: {test_prompt}")
    print(f"🎯 출력 (처음 500자): {response[:500]}...")
    
    # 기본 품질 체크
    quality_indicators = ["project_summary", "ERD", "API"]
    found_indicators = sum(1 for indicator in quality_indicators if indicator in response)
    
    if found_indicators >= 2:
        print(f"✅ 품질 체크: 통과 ({found_indicators}/3 지표 포함)")
    else:
        print(f"⚠️  품질 체크: 검토 필요 ({found_indicators}/3 지표 포함)")
    
    # 메모리 정리
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # 설정
    BASE_MODEL = "allganize/Llama-3-Alpha-Ko-8B-Instruct"
    ADAPTER_PATH = "./llama-3.1-korean-8b-qlora/lora_adapter"
    OUTPUT_PATH = "./llama-3.1-korean-8b-merged"
    HUB_MODEL_NAME = "your-username/llama-3.1-korean-project-analyzer"  # 원하는 이름으로 변경
    
    # 어댑터 병합 실행
    merged_path = merge_qlora_adapter(
        base_model_name=BASE_MODEL,
        adapter_path=ADAPTER_PATH,
        output_path=OUTPUT_PATH,
        push_to_hub=False,  # True로 변경하면 Hub에 업로드
        hub_model_name=HUB_MODEL_NAME
    )
    
    # 병합된 모델 테스트
    test_merged_model(merged_path)
    
    print("\n📋 사용법:")
    print(f"1. 로컬 사용: python inference.py --model_path {OUTPUT_PATH}")
    print(f"2. Hub 업로드 후: python inference.py --model_name {HUB_MODEL_NAME}")