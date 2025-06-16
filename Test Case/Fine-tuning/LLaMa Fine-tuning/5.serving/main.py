from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# 전역 변수로 모델과 토크나이저 선언
tokenizer = None
model = None

@app.on_event("startup")
async def load_model():
    """서버 시작 시 모델 로드"""
    global tokenizer, model
    try:
        model_name = "Min-kyu/PJA_LLM_MODEL_8bit"
        logger.info(f"Loading model: {model_name}")
        
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # pad_token이 없으면 eos_token으로 설정
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 모델 로드 (8bit 모델이므로 device_map 설정)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise e

class PromptRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 0.7

@app.post("/generate")
def generate_text(request: PromptRequest):
    global tokenizer, model
    
    try:
        # 모델이 로드되었는지 확인
        if model is None or tokenizer is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        logger.info(f"Generating text for prompt: {request.prompt[:50]}...")
        
        # 입력 토큰화
        inputs = tokenizer(
            request.prompt, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        eos_token_id = tokenizer.convert_tokens_to_ids('<|eot_id|>')

        # 모델이 GPU에 있다면 입력도 GPU로 이동
        if next(model.parameters()).is_cuda:
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 텍스트 생성
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                do_sample=True,
                eos_token_id= eos_token_id,
                pad_token_id= tokenizer.eos_token_id,
            )
        
        # 생성된 텍스트만 추출 (입력 프롬프트 제외)
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        logger.info("Text generation completed")
        return {"response": result.strip()}
        
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/health")
def health_check():
    """헬스체크 엔드포인트"""
    return {
        "status": "healthy",
        "model_loaded": model is not None and tokenizer is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)