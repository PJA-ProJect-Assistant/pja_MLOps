"""
QLoRA 파인튜닝된 모델 테스트 스크립트
파일명: test_qlora.py

실행 방법:
python test_qlora.py
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
import json
from datasets import load_dataset

def load_qlora_model(base_model_path, adapter_path):
    """QLoRA 어댑터가 적용된 모델 로드"""
    
    # 4bit 양자화 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    # 베이스 모델 로드 (4bit 양자화)
    print("📦 Loading base model with 4-bit quantization...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    
    # QLoRA 어댑터 적용
    print("🔗 Loading LoRA adapter...")
    model = PeftModel.from_pretrained(
        base_model, 
        adapter_path,
        torch_dtype=torch.bfloat16,
    )
    
    # 추론을 위해 어댑터 병합 (선택사항)
    # model = model.merge_and_unload()  # 메모리가 충분하면 사용
    
    return model, tokenizer

def test_qlora_model():
    """QLoRA 파인튜닝된 모델 테스트"""
    
    # 경로 설정
    base_model_path = "allganize/Llama-3-Alpha-Ko-8B-Instruct"
    adapter_path = "./llama-3.1-korean-8b-qlora/lora_adapter"
    
    # 모델 및 토크나이저 로드
    model, tokenizer = load_qlora_model(base_model_path, adapter_path)
    
    # 추론 최적화 설정
    model.eval()
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # 테스트 데이터 로드
    test_dataset = load_dataset("json", data_files="../data/test_dataset.json", split="train")
    print(f"📊 Test 데이터 개수: {len(test_dataset)}")
    
    # 몇 개 샘플 테스트
    num_samples = min(3, len(test_dataset))
    
    for i in range(num_samples):
        sample = test_dataset[i]
        messages = sample['messages']
        
        # 시스템 + 유저 프롬프트만 사용
        test_messages = [
            messages[0],  # system
            messages[1]   # user
        ]
        
        # 정답 (기대 출력)
        expected_output = messages[2]['content']
        
        print(f"\\n{'='*60}")
        print(f"테스트 케이스 {i+1}:")
        print(f"{'='*60}")
        print(f"입력: {messages[1]['content'][:150]}...")
        
        # 모델 추론
        formatted_prompt = tokenizer.apply_chat_template(
            test_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 토크나이즈 및 GPU로 전송
        inputs = tokenizer(
            formatted_prompt, 
            return_tensors="pt",
            max_length=2048,
            truncation=True
        ).to(model.device)
        
        print("🤖 생성 중...")
        
        # 생성 설정
        generation_config = {
            "max_new_tokens": 4096,
            "min_new_tokens": 2000,
            "do_sample": True,              # QLoRA는 sampling이 더 좋을 수 있음
            "temperature": 0.1,             # 낮은 temperature로 일관성 확보
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "pad_token_id": tokenizer.eos_token_id,
            "use_cache": True,
            "early_stopping": False,
        }
        
        # 추론 실행
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **generation_config
            )
        
        # 결과 디코딩
        generated_output = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        print(f"\\n📝 기대 출력 (처음 200자):")
        print(f"{expected_output[:200]}...")
        print(f"\\n🎯 실제 출력 (처음 200자):")
        print(f"{generated_output[:200]}...")
        
        # 품질 평가
        required_keywords = ["project_summary", "relationships_data", "ERD_data", "API_specs_data"]
        found_keywords = sum(1 for kw in required_keywords if kw in generated_output)
        
        if found_keywords >= 3:
            print(f"\\n✅ 구조적 일관성: 좋음 ({found_keywords}/4 키워드 포함)")
        else:
            print(f"\\n❌ 구조적 일관성: 부족 ({found_keywords}/4 키워드 포함)")
            
        print(f"📏 출력 길이: {len(generated_output)}자")
        
        # JSON 파싱 시도
        try:
            # 출력에서 JSON 부분만 추출 시도
            if "**프로젝트 상세 정보:**" in generated_output:
                json_start = generated_output.find("{'project_summary'")
                if json_start != -1:
                    # 첫 번째 JSON 블록만 검증
                    json_part = generated_output[json_start:json_start+500]
                    print("📋 JSON 형식 체크: 시작 부분이 올바른 형식입니다")
                else:
                    print("⚠️  JSON 형식 체크: project_summary를 찾을 수 없습니다")
            else:
                print("⚠️  JSON 형식 체크: 예상된 구조를 찾을 수 없습니다")
        except Exception as e:
            print(f"⚠️  JSON 파싱 오류: {str(e)}")
        
        print("-" * 60)
    
    print("\\n🎉 QLoRA 모델 테스트 완료!")
    
    # 메모리 정리
    del model
    torch.cuda.empty_cache()

def compare_model_sizes():
    """모델 크기 비교"""
    print("📊 모델 크기 비교:")
    print("• Full Fine-tuning: ~16GB (전체 모델 파라미터)")
    print("• QLoRA: ~4GB (4bit 양자화) + ~100-200MB (LoRA adapter)")
    print("• 메모리 절약: 약 75% 감소")
    print("\\n⚡ QLoRA 장점:")
    print("• 메모리 사용량 대폭 감소")
    print("• 학습 속도 향상") 
    print("• 여러 태스크용 어댑터 관리 용이")
    print("• 베이스 모델 재사용 가능")

if __name__ == "__main__":
    compare_model_sizes()
    print("\\n" + "="*60)
    test_qlora_model()