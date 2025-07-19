#!/usr/bin/env python3
"""
vLLM Inference Server with OpenAI-compatible API
"""

import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import json
import base64

# Request/Response models
class GenerateRequest(BaseModel):
    prompt: Optional[str] = None
    prompt_token_ids: Optional[List[int]] = None
    multi_modal_data: Optional[Dict[str, Any]] = None
    sampling_params: Optional[Dict[str, Any]] = None
    use_lora: Optional[bool] = True

class GenerateResponse(BaseModel):
    generated_text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int

# Global variables
llm = None
default_sampling_params = None
lora_request = None

app = FastAPI(title="vLLM Server", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    global llm, default_sampling_params, lora_request
    
    # Engine arguments from environment variables
    engine_args = {
        "model": os.environ.get("BASE_MODEL_PATH", "Qwen/Qwen2.5-VL-3B-Instruct"),
        "gpu_memory_utilization": float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.5")),
        "max_model_len": int(os.environ.get("MAX_MODEL_LEN", "16384")),
        "tensor_parallel_size": int(os.environ.get("TENSOR_PARALLEL_SIZE", "1")),
        "dtype": os.environ.get("DTYPE", "auto"),
        "trust_remote_code": True,
        "enable_lora": True,
        "limit_mm_per_prompt": {"image": 4, "video": 2, "audio": 2}
    }
    
    print("Initializing vLLM with engine args:", engine_args)
    
    # Initialize LLM
    llm = LLM(**engine_args)
    
    # Default sampling parameters
    default_sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512
    )
    
    # LoRA request (if LoRA path exists)
    lora_path = os.environ.get("LORA_PATH")
    if lora_path and os.path.exists(lora_path):
        lora_request = LoRARequest("default", 1, lora_path)
        print(f"Using LoRA adapter from: {lora_path}")
    else:
        print("No LoRA adapter found or path invalid, using base model only")
    
    print("vLLM initialized successfully!")
    print(f"Server starting on {os.environ.get('HOST', '0.0.0.0')}:{os.environ.get('PORT', '8000')}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": llm is not None}

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    global llm, default_sampling_params, lora_request
    
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare sampling parameters
        if request.sampling_params:
            sampling_params = SamplingParams(**request.sampling_params)
        else:
            sampling_params = default_sampling_params
        
        # Prepare input for vLLM
        if request.prompt_token_ids is not None:
            # For multimodal with token IDs, we need to convert images to PIL format
            from PIL import Image
            import io
            
            # Convert base64 images back to PIL Images
            processed_multi_modal_data = {}
            if request.multi_modal_data and "image" in request.multi_modal_data:
                processed_images = []
                for img_data in request.multi_modal_data["image"]:
                    if isinstance(img_data, str):
                        # Base64 string - decode to PIL Image
                        try:
                            img_bytes = base64.b64decode(img_data)
                            pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                            processed_images.append(pil_image)
                        except Exception as e:
                            print(f"Error processing image: {e}")
                            continue
                    else:
                        # Already a PIL Image or other format
                        processed_images.append(img_data)
                
                processed_multi_modal_data["image"] = processed_images
            
            # Create input data structure for vLLM
            input_data = {
                "prompt_token_ids": request.prompt_token_ids,
                "multi_modal_data": processed_multi_modal_data
            }
            
        elif request.prompt is not None:
            # Use text prompt - simpler case
            input_data = request.prompt
        else:
            raise HTTPException(status_code=400, detail="Either prompt or prompt_token_ids must be provided")
        
        # Select LoRA request
        current_lora_request = lora_request if request.use_lora else None
        
        print(f"Generating with input type: {'token_ids' if request.prompt_token_ids else 'text'}")
        print(f"Using LoRA: {current_lora_request is not None}")
        
        # Generate response
        result = llm.generate(
            input_data,
            sampling_params=sampling_params,
            lora_request=current_lora_request,
        )
        
        # Extract response
        if result and len(result) > 0:
            output = result[0]
            generated_text = output.outputs[0].text
            finish_reason = output.outputs[0].finish_reason
            prompt_tokens = len(output.prompt_token_ids) if output.prompt_token_ids else 0
            completion_tokens = len(output.outputs[0].token_ids) if output.outputs[0].token_ids else 0
            
            return GenerateResponse(
                generated_text=generated_text,
                finish_reason=finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens
            )
        else:
            raise HTTPException(status_code=500, detail="No output generated")
            
    except Exception as e:
        import traceback
        print(f"Generation error: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/generate_batch")
async def generate_batch(requests: List[GenerateRequest]):
    """Batch generation endpoint"""
    results = []
    for req in requests:
        try:
            result = await generate_text(req)
            results.append(result)
        except Exception as e:
            results.append({"error": str(e)})
    return results

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )
