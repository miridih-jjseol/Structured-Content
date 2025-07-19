# 구조화된 콘텐츠 파이프라인

FastAPI 서버-클라이언트 아키텍처를 사용하여 XML 및 이미지 입력으로부터 구조화된 콘텐츠를 생성하는 비전-언어 모델(VLM) 파이프라인입니다.

## 개요

이 파이프라인은 XML 레이아웃 데이터와 썸네일 이미지를 처리하여 FastAPI vLLM 서버를 사용한 구조화된 콘텐츠 예측을 생성합니다. 시스템은 확장성과 효율성을 위해 완전한 서버-클라이언트 분리로 설계되었습니다.

## 특징

- **서버-클라이언트 아키텍처**: 추론 서버와 클라이언트 처리의 완전한 분리
- **비전-언어 모델 지원**: LoRA 어댑터와 함께 Qwen2.5-VL 사용
- **구조화된 출력 생성**: XML 레이아웃 데이터를 의미론적 그룹으로 변환
- **실시간 예측**: FastAPI 기반 추론과 멀티모달 데이터 처리
- **WandB 통합**: 결과의 자동 로깅 및 시각화

## 시스템 아키텍처

### 서버-클라이언트 구조

```
[서버 측]                           [클라이언트 측]
┌─────────────────┐                ┌──────────────────────────────────┐
│  vLLM 서버      │                │  XML + 이미지 입력               │
│  - Qwen2.5-VL   │                │           ↓                      │
│  - LoRA 어댑터  │◄──────────────►│  vLLM 입력 전처리                │
│  - FastAPI      │                │           ↓                      │
│  - 재중님 모델  │                │  서버로 요청 전송                │
└─────────────────┘                │           ↓                      │
                                   │  응답 파싱 (Response Parsing)    │
                                   │           ↓                      │
                                   │  시각화 (Visualization)          │
                                   └──────────────────────────────────┘
```

### 처리 흐름

1. **서버**: `bash start_vllm_server.sh` 실행
   - vLLM 서버 시작
   - Qwen2.5-VL 모델 로드
   - LoRA 어댑터 적용
   - FastAPI 엔드포인트 활성화

2. **클라이언트**: 데이터 처리 및 예측
   - XML 파일 → vLLM 입력 형식으로 전처리
   - 이미지 전처리 및 인코딩
   - 서버로 멀티모달 요청 전송
   - 재중님 vLLM 모델에서 예측 수행
   - 응답 파싱 및 구조화
   - 결과 시각화 생성

## 설치

### 필수 조건

- Python 3.8+
- CUDA 호환 GPU
- LLaMA-Factory
- FastAPI 및 관련 의존성

### 의존성 설치

```bash
pip install -r requirements.txt
```

### LLaMA-Factory 설치

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```

## 서버 설정

### 1. FastAPI vLLM 서버 시작

학습된 모델로 FastAPI 서버를 시작하려면 다음 스크립트를 사용하세요:

```bash
# 서버 시작 스크립트
bash start_vllm_server.sh
```

### 2. 서버 실행 확인

서버가 접근 가능한지 확인:

```bash
curl http://localhost:8000/health
```

## 클라이언트 사용법

### CLI 배치 처리 스크립트

가장 간단한 사용법은 배치 처리 스크립트를 사용하는 것입니다:

```bash
# 기본 사용법
python run_batch_xml_pipeline.py \
    --api_endpoint http://211.47.48.147:8000/generate \
    --enable_realtime_prediction

# 추가 옵션과 함께
python run_batch_xml_pipeline.py \
    --api_endpoint http://211.47.48.147:8000/generate \
    --enable_realtime_prediction \
    --project_name "your-wandb-project" \
    --table_name "layout_results"
```

#### CLI 옵션 설명

- `--api_endpoint`: FastAPI 서버의 엔드포인트 URL
- `--enable_realtime_prediction`: 실시간 예측 활성화
- `--project_name`: WandB 프로젝트 이름 (선택사항)
- `--table_name`: WandB 테이블 이름 (선택사항)

### 프로그래밍 방식 사용법

#### 기본 사용법

```python
from integrated_pipeline import IntegratedStructuredContentPipeline

# FastAPI 서버 엔드포인트로 파이프라인 초기화
pipeline = IntegratedStructuredContentPipeline(
    project_name="your-wandb-project",
    table_name="layout_results",
    enable_realtime_prediction=True,
    api_endpoint="http://localhost:8000/generate"
)

# XML과 이미지 처리
result = pipeline.process_xml_and_image(
    xml_file_path="path/to/layout.xml",
    thumbnail_image_path="path/to/image.jpg",
    template_id="template_001"
)

# 결과를 WandB에 저장
pipeline.save_to_wandb([result])
```

#### 배치 처리

```python
# 여러 파일 처리
xml_files = ["layout1.xml", "layout2.xml", "layout3.xml"]
image_files = ["image1.jpg", "image2.jpg", "image3.jpg"]

results = []
for xml_path, img_path in zip(xml_files, image_files):
    result = pipeline.process_xml_and_image(xml_path, img_path)
    if result:
        results.append(result)

# 모든 결과 저장
pipeline.save_to_wandb(results)
```

## 클라이언트 처리 과정 상세

### 1. XML → vLLM 입력 전처리
- XML 파일에서 elements_metadata 추출
- 멀티모달 데이터 구조로 변환
- 이미지 전처리 및 크기 조정
- 토큰화 및 프롬프트 생성

### 2. 서버 통신
- FastAPI 엔드포인트로 요청 전송
- 재중님 vLLM 모델에서 추론 수행
- JSON 형태 응답 수신

### 3. Response Parsing
- 모델 응답에서 구조화된 출력 추출
- semantic_group → layout_functions 변환
- 오류 처리 및 fallback 로직

### 4. Visualization (클라이언트)
- MORDOR 형식: Layout functions 시각화
- MESSI 형식: 3-panel view 생성 (원본 + 요소 박스 + 그룹 박스)
- WandB 이미지 객체 생성

## 설정

### 서버 설정

`llamafactory-cli api` 명령어에서 서버 설정 수정:

- `--api_port`: 서버 포트 (기본값: 8000)
- `--model_name_or_path`: 기본 모델 경로
- `--adapter_name_or_path`: LoRA 어댑터 경로
- `--template`: 사용할 채팅 템플릿

### 클라이언트 설정

다양한 옵션으로 파이프라인 설정:

```python
pipeline = IntegratedStructuredContentPipeline(
    project_name="MORDOR-structured-output-validation",
    table_name="layout_results",
    enable_realtime_prediction=True,
    api_endpoint="http://your-server:8000/generate",
    api_model_name="default",
    api_key="your_api_key"  # 선택사항
)
```

## 입력 형식

### XML 레이아웃 파일

```xml
<?xml version="1.0" encoding="UTF-8"?>
<root>
    <SIMPLE_TEXT Priority="1" TbpeId="text_001">
        <Position Left="100" Top="200" Right="400" Bottom="250"/>
        <TextBody>{"c": [{"c": [{"c": ["Sample Text"]}]}]}</TextBody>
    </SIMPLE_TEXT>
    <SHAPESVG Priority="2" TbpeId="shape_001">
        <Position Left="50" Top="50" Right="150" Bottom="100"/>
    </SHAPESVG>
</root>
```

### 썸네일 이미지

- 지원 형식: JPG, PNG
- 권장 크기: 임의 크기 (자동으로 크기 조정됨)
- 품질: 더 나은 결과를 위해 고품질 권장

## 출력 형식

파이프라인은 다음 구성 요소로 구조화된 출력을 생성합니다:

### WandB 테이블 컬럼

- `template_id`: 템플릿의 고유 식별자
- `elements_metadata`: 요소 위치 및 속성의 JSON 문자열
- `predicted_output`: 모델 예측의 JSON 문자열
- `structured_output`: MORDOR 형식의 생성된 시각화 (_draw_layout_function을 사용한 레이아웃 함수 시각화)
- `model_predicted`: MESSI 형식의 생성된 시각화 (3-panel view를 통한 실시간 예측 결과)

### 출력 예시

```python
{
    'template_id': 'template_001',
    'elements_metadata': '{"element_1": {"x": 100, "y": 200, "w": 300, "h": 50, "tag": "TEXT", "text_content": "Sample Text"}}',
    'predicted_output': '{"Parent Group 1": {"Subgroup 1": {"element_1": null}}}',
    'structured_output': <wandb.Image>,  # MORDOR 형식 시각화
    'model_predicted': <wandb.Image>     # MESSI 형식 시각화
}
```

## API 엔드포인트

FastAPI 서버는 다음 엔드포인트를 제공합니다:

- `POST /generate`: 메인 추론 엔드포인트
  - 단일 텍스트/이미지 입력에 대한 추론 수행
  - `prompt` 또는 `prompt_token_ids` + `multi_modal_data` 지원
  - LoRA 어댑터 사용 여부 선택 가능
- `POST /generate_batch`: 배치 추론 엔드포인트  
  - 여러 요청을 한 번에 처리
  - 각 요청은 `/generate`와 동일한 형식
- `GET /health`: 상태 확인 엔드포인트
  - 서버 상태 및 모델 로드 여부 확인

### API 파라미터 설명

#### `/generate` 요청 파라미터

| 파라미터 | 타입 | 필수 | 설명 |
|---------|------|------|------|
| `prompt` | string | 선택 | 텍스트 프롬프트 (prompt_token_ids와 배타적) |
| `prompt_token_ids` | List[int] | 선택 | 토큰 ID 배열 (prompt와 배타적) |
| `multi_modal_data` | object | 선택 | 멀티모달 데이터 (이미지 등) |
| `multi_modal_data.image` | List[string] | 선택 | Base64 인코딩된 이미지 배열 |
| `sampling_params` | object | 선택 | 샘플링 파라미터 |
| `sampling_params.temperature` | float | 선택 | 생성 온도 (0.0~2.0, 기본값: 0.7) |
| `sampling_params.top_p` | float | 선택 | Top-p 샘플링 (0.0~1.0, 기본값: 0.9) |
| `sampling_params.max_tokens` | int | 선택 | 최대 생성 토큰 수 (기본값: 512) |
| `sampling_params.stop` | List[string] | 선택 | 중단 문자열 배열 |
| `use_lora` | boolean | 선택 | LoRA 어댑터 사용 여부 (기본값: true) |

#### 응답 형식

| 필드 | 타입 | 설명 |
|------|------|------|
| `generated_text` | string | 생성된 텍스트 |
| `finish_reason` | string | 완료 사유 (stop, length 등) |
| `prompt_tokens` | int | 입력 토큰 수 |
| `completion_tokens` | int | 생성된 토큰 수 |

### API 사용 예시

#### 1. 서버 상태 확인

```bash
curl -X GET http://211.47.48.147:8000/health
```

**응답:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### 2. 텍스트 프롬프트로 추론

```bash
curl -X POST http://211.47.48.147:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Analyze this layout and group elements semantically.",
    "sampling_params": {
      "temperature": 0.0,
      "top_p": 1.0,
      "max_tokens": 2048
    },
    "use_lora": true
  }'
```

#### 3. 멀티모달 추론 (이미지 + 텍스트)

```bash
curl -X POST http://211.47.48.147:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "<image>\nPlease group these elements: {\"element_1\": {\"x\": 100, \"y\": 200}}",
    "multi_modal_data": {
      "image": ["base64_encoded_image_string"]
    },
    "sampling_params": {
      "temperature": 0.0,
      "max_tokens": 2048
    },
    "use_lora": true
  }'
```

#### 4. 토큰 ID 기반 추론

```bash
curl -X POST http://211.47.48.147:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt_token_ids": [151644, 872, 198, 151645, 198],
    "multi_modal_data": {
      "image": ["base64_encoded_image_string"]
    },
    "sampling_params": {
      "temperature": 0.0,
      "max_tokens": 2048,
      "stop": []
    },
    "use_lora": true
  }'
```

#### 5. 배치 추론

```bash
curl -X POST http://211.47.48.147:8000/generate_batch \
  -H "Content-Type: application/json" \
  -d '[
    {
      "prompt": "First request",
      "sampling_params": {"temperature": 0.0, "max_tokens": 1024},
      "use_lora": true
    },
    {
      "prompt": "Second request", 
      "sampling_params": {"temperature": 0.0, "max_tokens": 1024},
      "use_lora": true
    }
  ]'
```

**응답 형식:**
```json
{
  "generated_text": "Generated response text here",
  "finish_reason": "stop",
  "prompt_tokens": 25,
  "completion_tokens": 150
}
```

## 문제 해결

### 일반적인 문제

1. **서버 연결 오류**
   ```
   ConnectionError: Unable to connect to API server
   ```
   - 지정된 포트에서 서버가 실행 중인지 확인
   - API 엔드포인트 URL 확인

2. **이미지 처리 오류**
   ```
   Error in template_obj image processing
   ```
   - 이미지 파일이 유효하고 접근 가능한지 확인
   - 이미지 파일 권한 확인

3. **모델 로딩 오류**
   ```
   Failed to load model or adapter
   ```
   - 모델 및 어댑터 경로 확인
   - GPU 메모리 가용성 확인

### 성능 팁

- 더 빠른 추론을 위해 GPU 가속 사용
- 가능한 경우 여러 파일 배치 처리
- 서버 리소스 사용량 모니터링
- 처리 속도가 중요한 경우 더 작은 이미지 사용

### WandB 설정

```python
import wandb

# 커스텀 WandB 설정
wandb.login(key="your_wandb_api_key")

pipeline = IntegratedStructuredContentPipeline(
    project_name="custom-project-name",
    table_name="custom-table-name"
)
```

## 서버 시작 스크립트

`start_vllm_server.sh` 스크립트 예시:


## 기여

1. 저장소 포크
2. 기능 브랜치 생성
3. 변경 사항 작성
4. 해당하는 경우 테스트 추가
5. 풀 리퀘스트 제출

## 라이선스

이 프로젝트는 MIT 라이선스 하에 라이선스가 부여됩니다 - 자세한 내용은 LICENSE 파일을 참조하세요.
