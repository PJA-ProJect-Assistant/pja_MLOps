import logging
from dotenv import load_dotenv
from dataclasses import dataclass, field
import os
import random
import torch
import json

from datasets import load_dataset
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments
from trl.commands.cli_utils import TrlParser
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed,)
from trl import setup_chat_format
from peft import LoraConfig
from trl import (SFTTrainer)

from sklearn.model_selection import train_test_split

# Load dataset from the hub

from huggingface_hub import login

load_dotenv()

api_key = os.getenv('HUG_API_KEY')

login(
    token=api_key,
)

### 3.5.3. 데이터셋 준비 
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

# 방법 1: 리스트 컴프리헨션 사용
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

### 3.5.4. Llama3 모델 파라미터 설정 
@dataclass
class ScriptArguments:
    dataset_path: str = field(
        default=None,
        metadata={
            "help": "데이터셋 파일 경로"
        },
    )
    model_name: str = field(
    default=None, metadata={"help": "SFT 학습에 사용할 모델 ID"}
    )
    max_seq_length: int = field(
        default=512, metadata={"help": "SFT Trainer에 사용할 최대 시퀀스 길이"}
    )
    question_key: str = field(
    default=None, metadata={"help": "지시사항 데이터셋의 질문 키"}
    )
    answer_key: str = field(
    default=None, metadata={"help": "지시사항 데이터셋의 답변 키"}
    )


def training_function(script_args, training_args):    
    # 데이터셋 불러오기 
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

    # 토크나이저 및 데이터셋 chat_template으로 변경하기      
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE
    tokenizer.padding_side = 'right'
    
    def template_dataset(examples):
        return{"text":  tokenizer.apply_chat_template(examples["messages"], tokenize=False)}
    
    train_dataset = train_dataset.map(template_dataset, remove_columns=["messages"])
    test_dataset = test_dataset.map(template_dataset, remove_columns=["messages"])
    
    # 데이터가 변화되었는지 확인하기 위해 2개만 출력하기 
    with training_args.main_process_first(
        desc="Log a few random samples from the processed training set"
    ):
        for index in random.sample(range(len(train_dataset)), 2):
            print(train_dataset[index]["text"])

    # Model 및 파라미터 설정하기 
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        attn_implementation="sdpa", 
        torch_dtype=torch.bfloat16,
        use_cache=False if training_args.gradient_checkpointing else True,  
    )
    
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Train 설정 
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        dataset_text_field="text",
        eval_dataset=test_dataset,
        max_seq_length=script_args.max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        dataset_kwargs={
            "add_special_tokens": False,  
            "append_concat_token": False, 
        },
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()
    
if __name__ == "__main__":

    parser = TrlParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_and_config()    
    
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    
    # set seed
    set_seed(training_args.seed)
  
    # launch training
    training_function(script_args, training_args)