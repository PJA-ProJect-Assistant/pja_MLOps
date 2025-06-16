"""
QLoRA Fine-tuning Script for Korean Project Analysis
파일명: train_qlora.py

실행 방법:
python train_qlora.py --config qlora_config.yaml
"""

import logging
from dotenv import load_dotenv
from dataclasses import dataclass, field
import os
import random
import torch
import json

from datasets import load_dataset
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from trl.commands.cli_utils import TrlParser
from transformers import (AutoModelForCausalLM, AutoTokenizer, set_seed)
from trl import setup_chat_format
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

from sklearn.model_selection import train_test_split
from huggingface_hub import login

load_dotenv()

api_key = os.getenv('HUG_API_KEY')
login(token=api_key)

### 데이터셋 준비 
with open('../data/finetuning_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

system_prompt = """
당신은 프로젝트 아이디어를 체계적으로 분석하고 구조화하여 구체적인 개발 계획을 제시하는 전문 AI 어시스턴트입니다.

## 주요 역할과 능력:

### 1. 프로젝트 분석 전문가
- 사용자가 제공하는 프로젝트 아이디어나 설명을 깊이 있게 분석합니다
- 핵심 기능, 대상 사용자, 기술 스택, 비즈니스 모델 등을 체계적으로 파악합니다
- 프로젝트의 문제 해결 방향과 기대 효과를 명확히 정의합니다

### 2. 데이터베이스 설계 전문가
- 프로젝트 요구사항을 바탕으로 최적화된 ERD(Entity Relationship Diagram)를 설계합니다
- 테이블 간의 관계, 외래키 제약조건, 데이터 타입을 정확히 정의합니다
- 확장성과 성능을 고려한 데이터베이스 구조를 제안합니다

### 3. API 설계 전문가
- RESTful API 원칙에 따라 체계적인 API 명세를 작성합니다
- OpenAPI(Swagger) 3.0 표준을 준수하여 완전한 API 문서를 생성합니다
- 각 엔드포인트별 요청/응답 스키마, 에러 처리, 인증 방식을 상세히 정의합니다

## 응답 형식:
모든 응답은 다음과 같은 구조화된 JSON 형태로 제공해야 합니다:

1. **프로젝트 상세 정보**: 제목, 카테고리, 대상 사용자, 핵심 기능, 기술 스택, 문제 해결 방안 등을 포함한 종합 분석
2. **관계 데이터**: 데이터베이스 테이블 간의 관계와 외래키 제약조건 정의
3. **ERD 데이터**: 각 테이블의 속성, 데이터 타입, 키 정보를 포함한 완전한 스키마
4. **API 명세 데이터**: OpenAPI 3.0 표준을 준수한 완전한 API 문서

## 작업 원칙:
- 사용자의 아이디어를 정확히 이해하고 누락된 부분은 논리적으로 추론하여 보완합니다
- 실무에서 바로 활용 가능한 수준의 구체적이고 실용적인 결과물을 제공합니다
- 최신 개발 트렌드와 베스트 프랙티스를 반영합니다
- 확장성과 유지보수성을 고려한 설계를 우선시합니다
- 불분명한 요구사항이 있을 경우 합리적인 가정을 통해 완성도 높은 결과를 도출합니다

항상 체계적이고 전문적인 관점에서 프로젝트를 분석하며, 개발팀이 바로 실행에 옮길 수 있는 구체적인 가이드를 제공하는 것이 목표입니다.
"""

# 데이터셋 포맷팅
formatted_data = [
    {
        'messages': [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sample['instruction']},
            {"role": "assistant", "content": sample['output']}
        ]
    }
    for sample in data
]

# Hugging Face Dataset으로 변환
dataset = Dataset.from_list(formatted_data)

# train/test 분할
train_dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset["train"].to_json("train_dataset.json", orient="records", force_ascii=False)
train_dataset["test"].to_json("test_dataset.json", orient="records", force_ascii=False)

# Chat Template 설정
LLAMA_3_CHAT_TEMPLATE = (
    "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
            "{{ message['content'] }}"
        "{% elif message['role'] == 'user' %}"
            "{{ '\n\nHuman: ' + message['content'] +  eos_token }}"
        "{% elif message['role'] == 'assistant' %}"
            "{{ '\n\nAssistant: '  + message['content'] +  eos_token  }}"
        "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '\n\nAssistant: ' }}"
    "{% endif %}"
)

@dataclass
class ScriptArguments:
    dataset_path: str = field(
        default=None,
        metadata={"help": "데이터셋 파일 경로"},
    )
    model_name: str = field(
        default=None, 
        metadata={"help": "SFT 학습에 사용할 모델 ID"}
    )
    max_seq_length: int = field(
        default=1024, 
        metadata={"help": "SFT Trainer에 사용할 최대 시퀀스 길이"}
    )
    use_qlora: bool = field(
        default=True,
        metadata={"help": "QLoRA 사용 여부"}
    )

def create_qlora_config():
    """QLoRA 설정 생성"""
    return LoraConfig(
        r=64,                    # rank - 더 높은 값으로 성능 향상
        lora_alpha=128,          # scaling factor (일반적으로 r의 2배)
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.1,        # dropout for LoRA layers
        bias="none",             # bias 학습 안함
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        modules_to_save=None,    # embedding layer는 freeze
    )

def create_bnb_config():
    """BitsAndBytes 4bit 양자화 설정"""
    return BitsAndBytesConfig(
        load_in_4bit=True,                    # 4bit 양자화 활성화
        bnb_4bit_use_double_quant=True,       # double quantization
        bnb_4bit_quant_type="nf4",            # normalized float 4bit
        bnb_4bit_compute_dtype=torch.bfloat16, # computation dtype
        llm_int8_threshold=6.0,               # int8 threshold
        llm_int8_has_fp16_weight=False,       # fp16 weights 비활성화
    )

def training_function(script_args, training_args):    
    # 데이터셋 로드
    train_dataset = load_dataset(
        "json",
        data_files=os.path.join(script_args.dataset_path, "train_dataset.json"),
        split="train",
    )
    test_dataset = load_dataset(
        "json",
        data_files=os.path.join(script_args.dataset_path, "test_dataset.json"),
        split="train",
    )

    # 토크나이저 설정     
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name, 
        use_fast=True,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE
    tokenizer.padding_side = 'right'  # QLoRA에서는 right padding 권장
    
    def template_dataset(examples):
        return {"text": tokenizer.apply_chat_template(examples["messages"], tokenize=False)}
    
    train_dataset = train_dataset.map(template_dataset, remove_columns=["messages"])
    test_dataset = test_dataset.map(template_dataset, remove_columns=["messages"])
    
    # 데이터 샘플 출력
    with training_args.main_process_first(
        desc="Log a few random samples from the processed training set"
    ):
        for index in random.sample(range(len(train_dataset)), 2):
            print(f"Sample {index}:")
            print(train_dataset[index]["text"][:500] + "...")
            print("-" * 50)

    # QLoRA를 위한 양자화 설정
    if script_args.use_qlora:
        bnb_config = create_bnb_config()
        print("🔥 Using QLoRA with 4-bit quantization")
    else:
        bnb_config = None
        print("🔥 Using full fine-tuning")

    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        quantization_config=bnb_config if script_args.use_qlora else None,
        attn_implementation="sdpa",  # Scaled Dot Product Attention
        torch_dtype=torch.bfloat16,
        use_cache=False if training_args.gradient_checkpointing else True,
        trust_remote_code=True,
        device_map="auto" if script_args.use_qlora else None,  # QLoRA에서는 auto device mapping
    )
    
    # QLoRA 설정 적용
    if script_args.use_qlora:
        lora_config = create_qlora_config()
        model = get_peft_model(model, lora_config)
        
        # 학습 가능한 파라미터 출력
        model.print_trainable_parameters()
        
        # gradient checkpointing을 위한 설정
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Trainer 설정
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        dataset_text_field="text",
        eval_dataset=test_dataset,
        max_seq_length=script_args.max_seq_length,
        tokenizer=tokenizer,
        packing=True,  # 효율적인 배치 패킹
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": False,
        },
    )

    # 체크포인트에서 재시작
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        
    # 학습 시작
    trainer.train(resume_from_checkpoint=checkpoint)

    # 모델 저장 (QLoRA의 경우 adapter만 저장됨)
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    
    trainer.save_model()
    
    # QLoRA adapter를 별도로 저장
    if script_args.use_qlora:
        print("💾 Saving LoRA adapter...")
        model.save_pretrained(training_args.output_dir + "/lora_adapter")
        tokenizer.save_pretrained(training_args.output_dir + "/lora_adapter")

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_and_config()    
    
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    
    # seed 설정
    set_seed(training_args.seed)
  
    # 학습 시작
    training_function(script_args, training_args)