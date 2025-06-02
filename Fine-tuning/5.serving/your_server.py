from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import uvicorn
from pydantic import BaseModel
import huggingface_hub
from dotenv import load_dotenv
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
api_key = os.getenv('HUG_API_KEY')

app = FastAPI()

tokenizer = None
model = None
device = None

@app.on_event("startup")
async def load_model():
    global tokenizer, model, device
    try:
        logger.info("모델 로딩 시작...")
        
        # 디바이스 설정
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"사용 디바이스: {device}")
        
        if api_key:
            huggingface_hub.login(token=api_key)
        
        tokenizer = AutoTokenizer.from_pretrained("Min-kyu/PJA_LLM_MODEL_8bit")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            "Min-kyu/PJA_LLM_MODEL_8bit",
            torch_dtype=torch.float16,
            device_map="auto",  # 또는 device_map={"": 0}
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        logger.info("모델 로드 완료")
        
    except Exception as e:
        logger.error(f"모델 로딩 실패: {str(e)}")

class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 100
    temperature: float = 0.7

@app.post("/generate")
async def generate_text(request: GenerateRequest):
    global tokenizer, model, device
    
    try:
        if tokenizer is None or model is None:
            raise HTTPException(status_code=503, detail="모델이 아직 로드되지 않았습니다")
        
        logger.info(f"생성 요청: {request.prompt}")
        
        # 입력 토큰화하고 GPU로 이동
        inputs = tokenizer(request.prompt, return_tensors="pt")
        
        # 모든 입력 텐서를 모델과 같은 디바이스로 이동
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        logger.info(f"입력 디바이스: {inputs['input_ids'].device}")
        logger.info(f"모델 디바이스: {model.device}")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=request.max_length,
                do_sample=True,
                temperature=request.temperature,
                pad_token_id=tokenizer.eos_token_id
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"생성 완료")
        
        return {"generated_text": result}
        
    except Exception as e:
        logger.error(f"생성 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"생성 실패: {str(e)}")

@app.get("/health")
async def health_check():
    global tokenizer, model
    status = "healthy" if (tokenizer is not None and model is not None) else "loading"
    return {"status": status}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)