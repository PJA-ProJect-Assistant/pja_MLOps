"""
GPU 메모리에 맞는 QLoRA 설정 자동 생성기
파일명: optimize_gpu_config.py

실행 방법:
python optimize_gpu_config.py
"""

import torch
import gc
import psutil
import GPUtil
from peft import LoraConfig
from transformers import BitsAndBytesConfig

def get_gpu_info():
    """GPU 정보 확인"""
    if torch.cuda.is_available():
        gpu = GPUtil.getGPUs()[0]
        total_memory = gpu.memoryTotal
        free_memory = gpu.memoryFree
        used_memory = gpu.memoryUsed
        
        print(f"🎮 GPU: {gpu.name}")
        print(f"📊 총 메모리: {total_memory}MB")
        print(f"🟢 사용 가능: {free_memory}MB")
        print(f"🔴 사용 중: {used_memory}MB")
        
        return total_memory, free_memory
    else:
        print("❌ CUDA를 사용할 수 없습니다")
        return 0, 0

def optimize_qlora_config_for_gpu(gpu_memory_mb):
    """GPU 메모리에 따른 QLoRA 설정 최적화"""
    
    configs = {}
    
    if gpu_memory_mb >= 40000:  # 40GB+ (A100, A6000)
        configs = {
            "name": "High-End GPU (40GB+)",
            "max_seq_length": 4096,
            "per_device_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "lora_r": 128,
            "lora_alpha": 256,
            "dataloader_num_workers": 8
        }
    elif gpu_memory_mb >= 24000:  # 24GB (RTX 4090, RTX 3090)
        configs = {
            "name": "High-End Consumer (24GB)",
            "max_seq_length": 2048,
            "per_device_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "lora_r": 64,
            "lora_alpha": 128,
            "dataloader_num_workers": 4
        }
    elif gpu_memory_mb >= 16000:  # 16GB (RTX 4080, RTX 4070 Ti)
        configs = {
            "name": "Mid-Range Consumer (16GB)",
            "max_seq_length": 1024,
            "per_device_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "lora_r": 32,
            "lora_alpha": 64,
            "dataloader_num_workers": 2
        }
    elif gpu_memory_mb >= 12000:  # 12GB (RTX 4070, RTX 3080 Ti)
        configs = {
            "name": "Entry Consumer (12GB)",
            "max_seq_length": 512,
            "per_device_batch_size": 1,
            "gradient_accumulation_steps": 32,
            "lora_r": 16,
            "lora_alpha": 32,
            "dataloader_num_workers": 1
        }
    else:
        configs = {
            "name": "Low Memory (8GB)",
            "max_seq_length": 256,
            "per_device_batch_size": 1,
            "gradient_accumulation_steps": 64,
            "lora_r": 8,
            "lora_alpha": 16,
            "dataloader_num_workers": 1
        }
    
    return configs

def create_memory_efficient_bnb_config(gpu_memory_mb):
    """메모리 효율적인 BitsAndBytes 설정"""
    
    if gpu_memory_mb < 16000:  # 16GB 미만
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,  # bfloat16 대신 float16
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
    else:  # 16GB 이상
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )

def create_memory_efficient_lora_config(configs):
    """메모리 효율적인 LoRA 설정"""
    return LoraConfig(
        r=configs["lora_r"],
        lora_alpha=configs["lora_alpha"],
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False,
    )

def memory_cleanup():
    """메모리 정리"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def monitor_memory_usage():
    """메모리 사용량 모니터링"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        
        print(f"🔋 GPU 메모리 - 할당: {allocated:.2f}GB, 예약: {reserved:.2f}GB")
    
    # CPU 메모리
    cpu_memory = psutil.virtual_memory()
    cpu_used = cpu_memory.used / 1024**3  # GB
    cpu_total = cpu_memory.total / 1024**3  # GB
    
    print(f"💾 CPU 메모리 - 사용: {cpu_used:.2f}GB / {cpu_total:.2f}GB")

def get_recommended_settings():
    """현재 시스템에 맞는 권장 설정 출력"""
    total_memory, free_memory = get_gpu_info()
    
    if total_memory == 0:
        print("⚠️  GPU를 찾을 수 없습니다.")
        return
    
    configs = optimize_qlora_config_for_gpu(total_memory)
    
    print(f"\n📋 {configs['name']} 권장 설정:")
    print(f"├─ max_seq_length: {configs['max_seq_length']}")
    print(f"├─ per_device_batch_size: {configs['per_device_batch_size']}")
    print(f"├─ gradient_accumulation_steps: {configs['gradient_accumulation_steps']}")
    print(f"├─ lora_r: {configs['lora_r']}")
    print(f"├─ lora_alpha: {configs['lora_alpha']}")
    print(f"└─ dataloader_num_workers: {configs['dataloader_num_workers']}")
    
    # 실효 배치 사이즈 계산
    effective_batch_size = configs['per_device_batch_size'] * configs['gradient_accumulation_steps']
    print(f"\n🎯 실효 배치 사이즈: {effective_batch_size}")
    
    # 메모리 사용량 추정
    estimated_memory = estimate_memory_usage(configs, total_memory)
    print(f"📊 예상 메모리 사용량: {estimated_memory:.1f}GB")
    
    if estimated_memory > total_memory * 0.95 / 1024:  # MB to GB
        print("⚠️  메모리 부족 위험! 설정을 낮춰주세요.")
    else:
        print("✅ 메모리 여유 충분!")
    
    return configs

def estimate_memory_usage(configs, gpu_memory_mb):
    """메모리 사용량 추정"""
    base_model_memory = 3.5  # 8B 모델 4bit 기준 약 3.5GB
    
    # LoRA 파라미터 메모리
    lora_memory = configs['lora_r'] * 0.01  # 대략적 추정
    
    # 배치 및 시퀀스 길이에 따른 메모리
    batch_memory = configs['per_device_batch_size'] * configs['max_seq_length'] * 0.001
    
    # Gradient checkpointing 고려
    gradient_memory = batch_memory * 0.5  # 대략 절반으로 감소
    
    total_estimated = base_model_memory + lora_memory + batch_memory + gradient_memory
    
    return total_estimated

def create_yaml_config(configs, output_file="auto_qlora_config.yaml"):
    """자동 생성된 설정을 YAML 파일로 저장"""
    yaml_content = f"""### Auto-generated QLoRA Config for {configs['name']}
model_name: "allganize/Llama-3-Alpha-Ko-8B-Instruct"
dataset_path: "."
max_seq_length: {configs['max_seq_length']}
output_dir: "./llama-3.1-korean-8b-qlora-auto"
report_to: "wandb"

# 최적화된 학습 설정
learning_rate: 0.0002
lr_scheduler_type: "cosine"
num_train_epochs: 3
per_device_train_batch_size: {configs['per_device_batch_size']}
per_device_eval_batch_size: {configs['per_device_batch_size']}
gradient_accumulation_steps: {configs['gradient_accumulation_steps']}
optim: "paged_adamw_8bit"

# 로깅 및 평가
logging_steps: 5
save_strategy: "steps"
save_steps: 25
eval_strategy: "steps"
eval_steps: 25

# 정규화
weight_decay: 0.01
max_grad_norm: 1.0
warmup_ratio: 0.03

# 메모리 최적화
bf16: true
tf32: true
gradient_checkpointing: true
dataloader_num_workers: {configs['dataloader_num_workers']}
group_by_length: true
dataloader_pin_memory: false

# 모델 저장
load_best_model_at_end: false
save_total_limit: 1

# QLoRA 활성화
use_qlora: yes
lora_r: {configs['lora_r']}
lora_alpha: {configs['lora_alpha']}

# 기타
remove_unused_columns: true
dataloader_drop_last: true
seed: 42
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"📄 설정 파일 저장: {output_file}")

if __name__ == "__main__":
    print("🔍 시스템 정보 확인 중...")
    
    # 현재 메모리 상황 체크
    monitor_memory_usage()
    
    # 권장 설정 생성
    configs = get_recommended_settings()
    
    if configs:
        # YAML 설정 파일 자동 생성
        create_yaml_config(configs)
        
        print(f"\n🚀 사용법:")
        print(f"python train_qlora.py --config auto_qlora_config.yaml")
        
        # 메모리 정리
        memory_cleanup()
        print("\n✨ 메모리 정리 완료!")