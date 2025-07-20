#!/usr/bin/env python3
"""
Integrated Pipeline for Structured Content Generation
Based on test_clean_code_pipeline.ipynb with process_wandb_csv.py functionality

=== SERVER-CLIENT SEPARATION METHODS ===

This pipeline supports multiple methods for server-client separation with vLLM:

METHOD 1: Native LlamaFactory API (RECOMMENDED - Complete Server-Client Separation)
- Server: Run vLLM server using LlamaFactory
- Client: Use generate_new_img_realtime_native_api() method
- Advantages: Proper image_grid_thw handling, complete separation
- Command to start server:
  llamafactory-cli api \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --adapter_name_or_path saves/qwen2_5vl-3b/lora/sft \
    --template qwen2_vl \
    --api_port 8000

METHOD 2: Direct vLLM Engine (Fallback - Partial Separation)
- Server: vLLM server running
- Client: Instantiate VllmEngine in client (less ideal)
- Use: generate_new_img_realtime_direct_vllm() method

METHOD 3: OpenAI API Format (Final Fallback)
- Server: Any OpenAI-compatible server
- Client: Use generate_new_img_realtime_openai_api() method
- Limitations: May have image_grid_thw issues with vLLM

METHOD 4: HuggingFace Engine (Alternative Backend)
- Server: Use HuggingFace engine instead of vLLM
- Command:
  llamafactory-cli api \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --adapter_name_or_path saves/qwen2_5vl-3b/lora/sft \
    --template qwen2_vl \
    --api_port 8000 \
    --use_hf_engine

USAGE:
pipeline = IntegratedStructuredContentPipeline(
    enable_realtime_prediction=True,
    api_endpoint="http://211.47.48.147:8000/v1/chat/completions"
)
result = pipeline.process_xml_and_image("test.xml", "image.jpg")
"""

import json
import os
import sys
import pandas as pd
import tempfile
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional, Tuple
from io import BytesIO
import base64
from PIL import Image
import wandb
import ast
import re
import requests

# Add aippt-jailbreak directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'aippt-jailbreak'))

# Import the required modules
from semanticGroup2StructuredOutput import semanticGroup2StructuredOutput, semanticGroup2LayoutFunction
from visualize_structured_output import StructuredContentVisualizer

# Import util functions for real-time processing
from util import xml_to_mllm_converter, clean_and_parse_json, draw_element_boxes, draw_group_boxes

# LLM related imports (need to be configured based on your setup)
try:
    # Add your LLM model imports here
    # from your_llm_module import llm, tokenizer, template_obj, sampling_params, lora_request
    pass
except ImportError:
    print("LLM modules not available. Real-time prediction will be disabled.")

def extract_text_from_list(list_str: str) -> str:
    """
    리스트 문자열에서 텍스트 추출
    예: ["Sub Topic 1", "", "Description"] -> "Sub Topic 1 Description"
    """
    import re
    
    try:
        # 리스트 내용 추출
        content = list_str[1:-1]  # [ ] 제거
        
        # 각 항목을 추출 (따옴표로 감싸진 문자열들)
        items = []
        current_item = ""
        in_quotes = False
        i = 0
        
        while i < len(content):
            char = content[i]
            if char == '"' and (i == 0 or content[i-1] != '\\'):
                in_quotes = not in_quotes
                if not in_quotes and current_item:
                    items.append(current_item)
                    current_item = ""
            elif in_quotes:
                current_item += char
            elif char == ',' and not in_quotes:
                # 쉼표로 구분되는 항목들
                pass
            i += 1
        
        # 마지막 항목 추가
        if current_item:
            items.append(current_item)
        
        # 빈 문자열이 아닌 항목들만 조인
        result = ' '.join([item.strip() for item in items if item.strip()])
        return result
        
    except Exception as e:
        print(f"Error extracting text from list: {e}")
        return ""

def parse_xml_to_elements_metadata(xml_file_path: str) -> Dict[str, Dict[str, Any]]:
    """
    XML 파일에서 elements_metadata 추출 (실제 XML 구조에 맞게 수정)
    """
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        elements_metadata = {}
        element_counter = 0
        
        # XML 구조에 따라 요소들 추출
        for element in root:
            if element.tag in ['SIMPLE_TEXT', 'SHAPESVG', 'GENERALSVG']:
                element_counter += 1
                elem_id = f"element_{element_counter}"
                
                # Position 정보 추출
                position_elem = element.find('Position')
                if position_elem is not None:
                    left = float(position_elem.get('Left', 0))
                    top = float(position_elem.get('Top', 0))
                    right = float(position_elem.get('Right', 0))
                    bottom = float(position_elem.get('Bottom', 0))
                    
                    x = int(left)
                    y = int(top)
                    w = int(right - left)
                    h = int(bottom - top)
                else:
                    x = y = w = h = 0
                
                # 요소 타입 결정
                if element.tag == 'SIMPLE_TEXT':
                    tag = 'TEXT'
                    # 텍스트 내용 추출 (TextBody에서)
                    text_content = ""
                    text_body = element.find('TextBody')
                    if text_body is not None and text_body.text:
                        try:
                            # JSON 형태의 텍스트 본문 파싱
                            import json
                            text_data = json.loads(text_body.text)
                            # 텍스트 내용을 간단히 추출
                            text_content = extract_text_from_json_structure(text_data)
                        except:
                            text_content = text_body.text[:50] + "..." if len(text_body.text) > 50 else text_body.text
                    
                elif element.tag in ['SHAPESVG', 'GENERALSVG']:
                    tag = 'SVG'
                    text_content = ""
                else:
                    tag = 'GROUP'
                    text_content = ""
                
                # Priority 정보 추출
                priority = int(element.get('Priority', element_counter))
                
                # TbpeId 추출
                tbpe_id = element.get('TbpeId', f"tbpe_{element_counter}")
                
                elements_metadata[elem_id] = {
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h,
                    'tag': tag,
                    'text_content': text_content,
                    'priority': priority,
                    'tbpe_id': tbpe_id
                }
        
        return elements_metadata
        
    except Exception as e:
        print(f"Error parsing XML file {xml_file_path}: {e}")
        import traceback
        traceback.print_exc()
        return {}

def extract_text_from_json_structure(text_data: Dict[str, Any]) -> str:
    """
    JSON 구조에서 텍스트 내용을 추출
    """
    try:
        text_parts = []
        
        if isinstance(text_data, dict) and 'c' in text_data:
            for content in text_data['c']:
                if isinstance(content, dict) and 'c' in content:
                    for run in content['c']:
                        if isinstance(run, dict) and 'c' in run:
                            for text_part in run['c']:
                                if isinstance(text_part, str):
                                    text_parts.append(text_part)
        
        result = ' '.join(text_parts).strip()
        # 너무 긴 텍스트는 잘라내기
        if len(result) > 100:
            result = result[:100] + "..."
        
        return result
    
    except Exception as e:
        print(f"Error extracting text from JSON structure: {e}")
        return ""

def generate_sample_predicted_output(elements_metadata: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    elements_metadata를 기반으로 샘플 predicted_output 생성
    """
    element_ids = list(elements_metadata.keys())
    
    # 간단한 그룹 구조 생성
    if len(element_ids) <= 2:
        return {
            "Parent Group 1": {
                "Subgroup 1": {elem_id: None for elem_id in element_ids}
            }
        }
    else:
        mid_point = len(element_ids) // 2
        return {
            "Parent Group 1": {
                "Subgroup 1": {elem_id: None for elem_id in element_ids[:mid_point]},
                "Subgroup 2": {elem_id: None for elem_id in element_ids[mid_point:]}
            }
        }

class IntegratedStructuredContentPipeline:
    """
    통합 구조화된 콘텐츠 생성 파이프라인
    """
    
    def __init__(self, project_name: str = "MORDOR-structured-output-validation", 
                 table_name: str = "layout_results",
                 enable_realtime_prediction: bool = False,
                 api_endpoint: str = "http://211.47.48.147:8000/v1/chat/completions",
                 api_model_name: str = "default",
                 api_key: Optional[str] = None,
                 output_dir: Optional[str] = None):
        self.project_name = project_name
        self.table_name = table_name
        self.visualizer = StructuredContentVisualizer()
        self.enable_realtime_prediction = enable_realtime_prediction
        self.output_dir = output_dir
        
        # Create output directory if specified
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"Output directory set to: {self.output_dir}")
        
        # API configuration
        self.api_endpoint_config = api_endpoint
        self.api_model_name_config = api_model_name
        self.api_key_config = api_key
        
        # Initialize LLM components if needed
        if enable_realtime_prediction:
            self.setup_llm_components()
    
    def setup_llm_components(self):
        """
        LLM 컴포넌트를 초기화합니다. (API endpoint 기반)
        """
        try:
            print("=" * 60)
            print("LLM COMPONENTS SETUP")
            print("=" * 60)
            print("🔧 Initializing LLM components for API endpoint...")
            
            # Import required modules for tokenizer only
            from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
            from llamafactory.data import get_template_and_fix_tokenizer
            from llamafactory.hparams import get_infer_args
            from llamafactory.model import load_tokenizer
            
            # Model configuration
            model_name_or_path = "Qwen/Qwen2.5-VL-3B-Instruct"
            adapter_name_or_path = "saves/qwen2_5vl-3b/lora/sft"
            dataset = "mllm_eval_dataset"
            dataset_dir = "data"
            template = "qwen2_vl"
            cutoff_len = 15000
            max_samples = None
            default_system = None
            enable_thinking = True
            
            # Image processing parameters
            self.image_max_pixels = 262144
            self.image_min_pixels = 32 * 32
            
            # API endpoint configuration
            self.api_endpoint = self.api_endpoint_config
            self.api_model_name = self.api_model_name_config
            self.api_headers = {
                "Content-Type": "application/json"
            }
            
            # Add API key if provided
            if self.api_key_config:
                self.api_headers["Authorization"] = f"Bearer {self.api_key_config}"
            
            # Sampling parameters for API
            self.api_sampling_params = {
                "temperature": 0.0,  # deterministic
                "top_p": 1.0,
                "top_k": -1,
                "max_tokens": 15000,
                "repetition_penalty": 1.0,
                "stop": [],
                "stream": False
            }
            
            # Get inference arguments for tokenizer and template
            model_args, data_args, finetuning_args, generating_args = get_infer_args(
                dict(
                    model_name_or_path=model_name_or_path,
                    adapter_name_or_path=adapter_name_or_path,
                    dataset=dataset,
                    dataset_dir=dataset_dir,
                    template=template,
                    cutoff_len=cutoff_len,
                    max_samples=max_samples,
                    preprocessing_num_workers=16,
                    default_system=default_system,
                    enable_thinking=enable_thinking,
                )
            )
            
            # Initialize tokenizer (still needed for preprocessing)
            print("📝 Loading tokenizer...")
            self.tokenizer_module = load_tokenizer(model_args)
            self.tokenizer = self.tokenizer_module["tokenizer"]
            
            # Initialize template
            print("📋 Loading template...")
            # training_args = Seq2SeqTrainingArguments(output_dir="dummy_dir")
            self.template_obj = get_template_and_fix_tokenizer(self.tokenizer, data_args)
            self.template_obj.mm_plugin.expand_mm_tokens = False
            
            # Store model and data args for direct vLLM engine
            self.model_args = model_args
            self.data_args = data_args
            self.finetuning_args = finetuning_args
            self.generating_args = generating_args
            
            # No need to load LLM - using API endpoint
            self.llm = None  # Placeholder, actual inference via API
            self.lora_request = None
            self.sampling_params = None
            
            print("✅ LLM components successfully initialized!")
            print(f"   Model: {model_name_or_path}")
            print(f"   Template: {template}")
            print(f"   API Endpoint: {self.api_endpoint}")
            print("🎯 Real-time prediction is now ENABLED (API mode)")
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ LLM setup failed: {e}")
            print("📋 Real-time prediction will be DISABLED")
            print("🔧 Using layout visualization as fallback for model_predicted")
            import traceback
            traceback.print_exc()
            self.enable_realtime_prediction = False
    

    
    def generate_new_img_realtime(self, xml_path: str, img_path: str) -> Tuple[Optional[Image.Image], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        실시간 new_img 생성 메서드 (FastAPI vLLM 서버 사용)
        Returns: (new_img, predicted_output_dict)
        """
        if not self.enable_realtime_prediction:
            print("🚫 Real-time prediction is disabled")
            return None, None, None
        
        print("🔄 Attempting real-time LLM prediction via FastAPI vLLM server...")
        
        # Use FastAPI vLLM server only
        return self.generate_new_img_realtime_fastapi(xml_path, img_path)
    
    def generate_new_img_realtime_fastapi(self, xml_path: str, img_path: str) -> Tuple[Optional[Image.Image], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        FastAPI vLLM 서버를 사용한 실시간 new_img 생성 (완전한 서버-클라이언트 분리)
        client_example.py에서 성공적으로 작동하는 방식을 사용합니다.
        Returns: (new_img, predicted_output_dict)
        """
        if not self.enable_realtime_prediction:
            print("🚫 Real-time prediction is disabled")
            return None, None, None
        
        print("🔄 Attempting real-time LLM prediction via FastAPI vLLM server...")
        
        # 필요한 컴포넌트들이 설정되었는지 확인
        required_components = ['tokenizer', 'template_obj', 'tokenizer_module', 'api_endpoint']
        missing_components = []
        
        for comp in required_components:
            if not hasattr(self, comp) or getattr(self, comp) is None:
                missing_components.append(comp)
        
        if missing_components:
            print("❌ FastAPI prediction failed: Missing components")
            print(f"   Missing components: {missing_components}")
            return None, None, None
        
        print("✅ All components available, proceeding with FastAPI prediction")
        
        try:
            import requests
            import base64
            from io import BytesIO
            
            # 1. input data processing
            mllm_dataset = xml_to_mllm_converter(xml_path, img_path)
            
            messages = mllm_dataset['messages']
            images = mllm_dataset['images']
            
            # 2. Process images and messages as in original code
            multi_modal_data = {
                "image": self.template_obj.mm_plugin._regularize_images(
                    images, 
                    image_max_pixels=self.image_max_pixels, 
                    image_min_pixels=self.image_min_pixels
                )["images"]
            }
            
            print(f"🔍 Regularized images: {len(multi_modal_data['image'])} images")
            
            # Process messages with regularized images
            processor = self.tokenizer_module.get('processor')
            processed_messages = self.template_obj.mm_plugin.process_messages(
                messages, 
                multi_modal_data["image"],  # type: ignore
                [], 
                [], 
                processor  # type: ignore
            )
            
            print(f"🔍 Processed messages: {len(processed_messages)} messages")
            
            # Encode messages to get prompt_ids
            prompt_ids, _ = self.template_obj.encode_oneturn(
                self.tokenizer, 
                processed_messages, 
                None, 
                None
            )
            
            print(f"🔍 Prompt IDs length: {len(prompt_ids)} tokens")
            
            # 3. Convert PIL Images to base64 format for FastAPI
            serializable_multi_modal_data = {"image": []}
            
            for img in multi_modal_data["image"]:
                if hasattr(img, 'save'):  # PIL Image
                    # Convert PIL Image to base64 string
                    img_buffer = BytesIO()
                    img.save(img_buffer, format='JPEG', quality=85)
                    img_bytes = img_buffer.getvalue()
                    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                    serializable_multi_modal_data["image"].append(img_base64)
                elif isinstance(img, str):
                    # If it's a file path, read and convert to base64
                    with open(img, 'rb') as f:
                        img_bytes = f.read()
                        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                        serializable_multi_modal_data["image"].append(img_base64)
                else:
                    # For other types, try to convert to string
                    serializable_multi_modal_data["image"].append(str(img))
            
            # 4. Prepare API request in FastAPI format (similar to client_example.py)
            api_payload = {
                "prompt_token_ids": prompt_ids,
                "multi_modal_data": serializable_multi_modal_data,
                "sampling_params": {
                    "temperature": 0.0,  # deterministic
                    "top_p": 1.0,
                    "max_tokens": 15000,
                    "stop": []
                },
                "use_lora": True
            }
            
            print(f"📡 Making FastAPI request to {self.api_endpoint}")
            print(f"🔍 Request payload: {len(prompt_ids)} tokens, {len(serializable_multi_modal_data['image'])} images")
            
            # 5. Make API request
            api_headers = {
                "Content-Type": "application/json"
            }
            
            # Add API key if provided
            if hasattr(self, 'api_key_config') and self.api_key_config:
                api_headers["Authorization"] = f"Bearer {self.api_key_config}"
            
            try:
                response = requests.post(
                    self.api_endpoint,
                    headers=api_headers,
                    json=api_payload,
                    timeout=300  # 5 minute timeout
                )
            except requests.exceptions.ConnectionError as e:
                print(f"❌ Connection error: Unable to connect to FastAPI server")
                print(f"   Error: {e}")
                print(f"   Make sure the vLLM server is running at {self.api_endpoint}")
                return None, None, None
            except requests.exceptions.Timeout:
                print(f"❌ Request timed out after 300 seconds")
                return None, None, None
            except Exception as e:
                print(f"❌ Request failed with error: {e}")
                return None, None, None
            
            if response.status_code != 200:
                print(f"❌ API request failed: {response.status_code}")
                print(f"   Response: {response.text[:500]}...")
                return None, None, None
            
            # 6. Parse response
            try:
                response_data = response.json()
                generated_text = response_data.get('generated_text', '')
                
                if not generated_text:
                    print("❌ No generated text in response")
                    return None, None, None
                
                print(f"✅ Generated text length: {len(generated_text)} characters")
                print(f"🔍 Generated text preview: {generated_text[:200]}...")
                
            except json.JSONDecodeError:
                print(f"❌ Failed to parse JSON response")
                print(f"   Response: {response.text[:500]}...")
                return None, None, None
            
            # 7. Parse and process the generated output (struct_label_group.json format)
            try:
                print("🔍 Parsing generated structured output...")
                
                # Remove thinking tags if present
                if '<thinking>' in generated_text and '</thinking>' in generated_text:
                    generated_text = re.sub(r'<thinking>.*?</thinking>', '', generated_text, flags=re.DOTALL)
                
                # Try to parse as direct JSON first (most common case)
                semantic_group = None
                try:
                    semantic_group = json.loads(generated_text.strip())
                    print("✅ Successfully parsed as direct JSON")
                except json.JSONDecodeError:
                    # If direct parsing fails, try to extract from code blocks
                    print("🔍 Direct JSON parsing failed, trying code blocks...")
                    code_blocks = re.findall(r'```(?:python|json)?\n(.*?)\n```', generated_text, re.DOTALL)
                    
                    if code_blocks:
                        for code_block in code_blocks:
                            try:
                                semantic_group = ast.literal_eval(code_block.strip())
                                break
                            except:
                                try:
                                    semantic_group = json.loads(code_block.strip())
                                    break
                                except:
                                    continue
                
                if semantic_group is None:
                    print("❌ Failed to parse structured output from generated text")
                    return None, None, None
                
                print(f"✅ Successfully parsed semantic group structure")
                print(f"🔍 Semantic group keys: {list(semantic_group.keys())}")
                
                # 8. Convert semantic group to layout functions using semanticGroup2LayoutFunction
                try:
                    from semanticGroup2StructuredOutput import semanticGroup2LayoutFunction
                    
                    # Get elements metadata from mllm_dataset messages content (excluding <image> part)
                    content = mllm_dataset['messages'][0]['content']
                    # Remove '<image>' part from the beginning
                    content_without_image = content.replace('<image>', '', 1).strip()
                    # Parse as JSON
                    elements_metadata = json.loads(content_without_image)
                    
                    print(f"🔍 Elements metadata count: {len(elements_metadata)}")
                    
                    # Convert semantic group to layout functions
                    layout_functions = semanticGroup2LayoutFunction(semantic_group, elements_metadata)
                    
                    print(f"✅ Successfully converted to layout functions")
                    print(f"🔍 Layout functions count: {len(layout_functions)}")
                    
                    # Create predicted_output_dict for compatibility
                    predicted_output_dict = {
                        'layout_functions': layout_functions,
                        'semantic_group': semantic_group
                    }
                    
                except Exception as e:
                    print(f"❌ Error converting semantic group to layout functions: {e}")
                    # Fallback: create a simple layout function structure
                    layout_functions = []
                    predicted_output_dict = {
                        'layout_functions': layout_functions,
                        'semantic_group': semantic_group
                    }
                
                if not layout_functions:
                    print("❌ No layout functions generated")
                    return None, None, None
                
                # Load original image
                original_image = Image.open(img_path).convert("RGB")
                
                # Generate component visualizations
                image = original_image.copy()
                drawed_element_img = draw_element_boxes(image, elements_metadata)
                drawed_group_box = draw_group_boxes(image, semantic_group, elements_metadata)
                
                # Create combined image
                new_img = Image.new('RGB', (image.width * 3, image.height))
                x_offset = 0
                for img in [image, drawed_element_img, drawed_group_box]:
                    new_img.paste(img, (x_offset, 0))
                    x_offset += img.width
                
                print(f"✅ Successfully generated new_img with size: {new_img.size}")
                return new_img, predicted_output_dict, elements_metadata
                
            except Exception as e:
                print(f"❌ Error processing generated output: {e}")
                import traceback
                traceback.print_exc()
                return None, None, None
                
        except Exception as e:
            print(f"❌ Error in FastAPI prediction: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    # (Other realtime prediction methods removed for simplicity)

    def parse_elements_metadata_regex(self, input_prompt_str: str) -> Dict[str, Dict[str, Any]]:
        """
        정규식을 사용한 견고한 파싱 방법
        """
        elements_metadata = {}
        
        try:
            # 전체 문자열 정제
            cleaned_str = input_prompt_str.strip()
            if cleaned_str.startswith('"') and cleaned_str.endswith('"'):
                cleaned_str = cleaned_str[1:-1]
            cleaned_str = cleaned_str.replace('""', '"')
            
            # 중괄호 매칭을 위한 더 정교한 파싱
            # JSON 객체의 시작과 끝을 정확히 찾기
            i = 0
            while i < len(cleaned_str):
                # 요소 ID 찾기
                elem_match = re.search(r'"([^"]+)": \{', cleaned_str[i:])
                if not elem_match:
                    break
                    
                elem_id = elem_match.group(1)
                start_pos = i + elem_match.start()
                content_start = i + elem_match.end()
                
                # 해당 요소의 끝점 찾기 (중괄호 매칭)
                brace_count = 1
                j = content_start
                while j < len(cleaned_str) and brace_count > 0:
                    if cleaned_str[j] == '{':
                        brace_count += 1
                    elif cleaned_str[j] == '}':
                        brace_count -= 1
                    j += 1
                
                if brace_count == 0:
                    # 요소 내용 추출
                    element_content = cleaned_str[content_start:j-1]
                    
                    # 각 속성 파싱
                    try:
                        tag_match = re.search(r'"tag": "([^"]+)"', element_content)
                        x_match = re.search(r'"x": (-?\d+)', element_content)
                        y_match = re.search(r'"y": (-?\d+)', element_content)
                        w_match = re.search(r'"w": (-?\d+)', element_content)
                        h_match = re.search(r'"h": (-?\d+)', element_content)
                        tbpe_id_match = re.search(r'"tbpe_id": "([^"]+)"', element_content)
                        priority_match = re.search(r'"priority": "?(-?\d+)"?', element_content)
                        
                        # text_content 추출
                        text_content = ""
                        text_match = re.search(r'"text_content": (.+?)(?=, "tbpe_id")', element_content)
                        if text_match:
                            text_raw = text_match.group(1).strip()
                            if text_raw == '"None"':
                                text_content = ""
                            elif text_raw.startswith('"') and text_raw.endswith('"'):
                                text_content = text_raw[1:-1]
                            elif text_raw.startswith('['):
                                # 리스트 형태 처리 - 끝까지 찾기
                                bracket_count = 0
                                text_end = 0
                                for k, char in enumerate(text_raw):
                                    if char == '[':
                                        bracket_count += 1
                                    elif char == ']':
                                        bracket_count -= 1
                                        if bracket_count == 0:
                                            text_end = k + 1
                                            break
                                
                                if text_end > 0:
                                    list_content = text_raw[:text_end]
                                    text_content = extract_text_from_list(list_content)
                            else:
                                text_content = text_raw.replace('"', '')
                        
                        # 모든 필수 속성 확인
                        if all([tag_match, x_match, y_match, w_match, h_match, tbpe_id_match]):
                            elements_metadata[elem_id] = {
                                'x': int(x_match.group(1)),  # type: ignore
                                'y': int(y_match.group(1)),  # type: ignore
                                'w': int(w_match.group(1)),  # type: ignore
                                'h': int(h_match.group(1)),  # type: ignore
                                'tag': tag_match.group(1),  # type: ignore
                                'text_content': text_content,
                                'priority': int(priority_match.group(1)) if priority_match else 0,
                                'tbpe_id': tbpe_id_match.group(1)  # type: ignore
                            }
                        else:
                            missing_attrs = []
                            if not tag_match: missing_attrs.append('tag')
                            if not x_match: missing_attrs.append('x')
                            if not y_match: missing_attrs.append('y')
                            if not w_match: missing_attrs.append('w')
                            if not h_match: missing_attrs.append('h')
                            if not tbpe_id_match: missing_attrs.append('tbpe_id')
                            print(f"Missing attributes for {elem_id}: {missing_attrs}")
                            
                    except Exception as e:
                        print(f"Error parsing element {elem_id}: {e}")
                    
                    # 다음 요소로 이동
                    i = j
                else:
                    # 중괄호가 매칭되지 않으면 다음 위치에서 시도
                    i += 1
        
        except Exception as e:
            print(f"Error in regex parsing: {e}")
        
        return elements_metadata

    def parse_predicted_output(self, predicted_output_str: str) -> Dict[str, Any]:
        """
        predicted_output 문자열을 파싱하여 구조화된 그룹으로 변환합니다.
        """
        try:
            # 문자열 정제
            cleaned_str = predicted_output_str.strip()
            
            # 만약 문자열이 따옴표로 시작하고 끝나면 제거
            if cleaned_str.startswith('"') and cleaned_str.endswith('"'):
                cleaned_str = cleaned_str[1:-1]
            
            # 이스케이프된 따옴표 처리
            cleaned_str = cleaned_str.replace('""', '"')
            
            # 줄바꿈 제거
            cleaned_str = cleaned_str.replace('\n', '').replace('\r', '')
            
            # 불필요한 텍스트 제거 (마지막 } 이후)
            last_brace_pos = cleaned_str.rfind('}')
            if last_brace_pos != -1:
                cleaned_str = cleaned_str[:last_brace_pos + 1]
            
            # JavaScript null을 Python None으로 변환
            cleaned_str = cleaned_str.replace('null', 'None')
            
            # 먼저 JSON 파싱 시도
            try:
                # null을 다시 복원하여 JSON 파싱
                json_str = cleaned_str.replace('None', 'null')
                predicted_output = json.loads(json_str)
                return predicted_output
            except json.JSONDecodeError:
                # JSON 파싱 실패 시 ast.literal_eval 사용
                predicted_output = ast.literal_eval(cleaned_str)
                return predicted_output
                
        except (ValueError, SyntaxError) as e:
            print(f"AST parsing error: {e}")
            print(f"Problematic string (first 200 chars): {predicted_output_str[:200]}")
            print(f"Problematic string (last 200 chars): {predicted_output_str[-200:]}")
            return {}
        except Exception as e:
            print(f"Error parsing predicted output: {e}")
            return {}

    def generate_layout_functions_text(self, layout_functions: List[Dict[str, Any]]) -> str:
        """
        Layout functions를 텍스트 형태로 변환합니다.
        """
        try:
            text_output = []
            for i, func in enumerate(layout_functions):
                layout_type = func.get('layoutType', 'Unknown')
                element_ids = func.get('elementIds', [])
                
                text_output.append(f"Function {i+1}:")
                text_output.append(f"  Layout Type: {layout_type}")
                text_output.append(f"  Element IDs: {', '.join(element_ids)}")
                text_output.append("")
            
            return '\n'.join(text_output)
        except Exception as e:
            print(f"Error generating layout functions text: {e}")
            return ""

    def convert_to_matplotlib_figure(self, layout_functions: List[Dict[str, Any]], 
                                    elements_metadata: Dict[str, Dict[str, Any]],
                                    background_image_path: Optional[str] = None) -> bytes:
        """
        Layout functions를 matplotlib figure로 변환하여 이미지 바이트로 반환합니다.
        """
        try:
            # 임시 파일로 이미지 저장
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            # 시각화 생성
            self.visualizer.visualize_layout_functions(
                layout_functions,
                elements_metadata,
                save_path=tmp_path,
                background_image_path=background_image_path if background_image_path and os.path.exists(background_image_path) else None
            )
            
            # 이미지 바이트로 읽기
            with open(tmp_path, 'rb') as f:
                image_bytes = f.read()
            
            # 임시 파일 삭제
            os.unlink(tmp_path)
            
            return image_bytes
        except Exception as e:
            print(f"Error converting to matplotlib figure: {e}")
            return b''

    def process_single_row(self, input_prompt: str, predicted_output: str, 
                          template_id: str, background_image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        단일 행 데이터를 처리하여 구조화된 출력을 생성합니다.
        """
        try:
            print(f"Processing template: {template_id}")
            
            # 1. 데이터 파싱
            elements_metadata = self.parse_elements_metadata_regex(input_prompt)
            predicted_output_dict = self.parse_predicted_output(predicted_output)
            
            if not elements_metadata or not predicted_output_dict:
                print(f"Skipping {template_id} due to parsing errors")
                return {}
            
            # 2. Layout functions 생성 (process_wandb_csv.py line 328)
            try:
                layout_functions = semanticGroup2LayoutFunction(predicted_output_dict, elements_metadata)
                print(f"Successfully generated {len(layout_functions)} layout functions")
            except Exception as e:
                print(f"Error in semanticGroup2LayoutFunction: {e}")
                return {}
            
            # 3. 텍스트 출력 생성 (process_wandb_csv.py line 339)
            # 3. 이미지 출력 생성 (layout functions visualization using _draw_layout_function)
            image_bytes = self.convert_to_matplotlib_figure(
                layout_functions, 
                elements_metadata, 
                background_image_path
            )
            
            # 4. wandb 이미지 객체 생성
            structured_output_wandb = None
            if image_bytes:
                try:
                    image = Image.open(BytesIO(image_bytes))
                    structured_output_wandb = wandb.Image(image)
                except Exception as e:
                    print(f"Error creating wandb image: {e}")
            
            # 5. 최종 출력을 row_data 형태로 저장 (process_wandb_csv.py의 나머지 부분)
            row_data = {
                'template_id': template_id,
                'input_prompt': input_prompt,
                'predicted_output': predicted_output,
                'layout_functions': layout_functions,  # 추가된 부분
                'structured_output': structured_output_wandb,  # Layout functions visualization (MORDOR format)
                'model_predicted': None  # CSV-based input doesn't have realtime prediction
            }
            
            return row_data
            
        except Exception as e:
            print(f"Error processing row for {template_id}: {e}")
            return {}

    def process_batch(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        배치 데이터를 처리합니다.
        """
        results = []
        
        for i, data in enumerate(data_list):
            print(f"Processing batch item {i+1}/{len(data_list)}")
            
            row_data = self.process_single_row(
                input_prompt=data.get('input_prompt', ''),
                predicted_output=data.get('predicted_output', ''),
                template_id=data.get('template_id', f'item_{i}'),
                background_image_path=data.get('background_image_path')
            )
            
            if row_data:
                results.append(row_data)
        
        return results

    def convert_pil_to_wandb_image(self, pil_image: Image.Image) -> Optional[Any]:
        """
        PIL Image를 wandb.Image로 변환합니다.
        """
        try:
            return wandb.Image(pil_image)
        except Exception as e:
            print(f"Error converting PIL image to wandb image: {e}")
            return None

    def process_xml_and_image(self, xml_file_path: str, thumbnail_image_path: str, 
                             template_id: Optional[str] = None,
                             model_predicted_output: Optional[Image.Image] = None) -> Dict[str, Any]:
        """
        XML 파일과 thumbnail image를 처리하여 구조화된 출력을 생성합니다.
        실시간 LLM 예측을 통해 new_img도 생성합니다.
        
        Args:
            xml_file_path: XML 파일 경로
            thumbnail_image_path: 썸네일 이미지 경로  
            template_id: 템플릿 ID
            model_predicted_output: 외부에서 제공된 모델 예측 이미지 (선택사항)
        """
        try:
            if not template_id:
                template_id = os.path.basename(xml_file_path).replace('.xml', '')
            
            print(f"Processing XML: {xml_file_path}")
            print(f"Processing thumbnail: {thumbnail_image_path}")
            
            # 1. 실시간 LLM 예측 및 new_img 생성 시도
            realtime_new_img = None
            predicted_output_dict = None
            
            if self.enable_realtime_prediction:
                try:
                    realtime_new_img, predicted_output_dict, elements_metadata = self.generate_new_img_realtime(xml_file_path, thumbnail_image_path)
                    
                    # 실시간 예측에서 predicted_output_dict도 추출
                    if realtime_new_img:
                        print("Using real-time LLM prediction")
                        
                except Exception as e:
                    print(f"Real-time prediction failed: {e}")
                    realtime_new_img = None
                    predicted_output_dict = None
            
            if not elements_metadata:
                print(f"No elements found in XML: {xml_file_path}")
                return {}
            
            # 3. predicted_output_dict 생성 (실시간 예측 실패 시 샘플 사용)
            if predicted_output_dict is None:
                predicted_output_dict = generate_sample_predicted_output(elements_metadata)
                print("Using sample predicted output")
            
            # 4. Layout functions 생성
            layout_functions = predicted_output_dict['layout_functions']
            
            # 6. new_img 처리 (실시간 생성 또는 외부 제공)
            new_img_wandb = None
            if model_predicted_output:
                # 외부에서 제공된 이미지를 new_img로 사용
                new_img_wandb = self.convert_pil_to_wandb_image(model_predicted_output)
                print("Using externally provided image as new_img")
            elif realtime_new_img:
                # 실시간 생성된 new_img 사용
                new_img_wandb = self.convert_pil_to_wandb_image(realtime_new_img)
                print("Using real-time generated new_img")
            else:
                # 플레이스홀더 new_img 생성
                placeholder_image = Image.new('RGB', (600, 200), (220, 220, 220))
                new_img_wandb = wandb.Image(placeholder_image)
                print("Using placeholder for new_img")
            
            # 5. 시각화 이미지 생성 (layout functions visualization using _draw_layout_function)
            structured_output_wandb = None
            image_bytes = self.convert_to_matplotlib_figure(
                layout_functions, 
                elements_metadata, 
                thumbnail_image_path if os.path.exists(thumbnail_image_path) else None
            )
            
            if image_bytes:
                try:
                    image = Image.open(BytesIO(image_bytes))
                    structured_output_wandb = wandb.Image(image)
                    # Save realtime_new_img to file if output_dir is specified
                    if self.output_dir:
                        try:
                            output_filename = f"{template_id}.png"
                            output_path = os.path.join(self.output_dir, output_filename)
                            image.save(output_path)
                            print(f"Saved realtime_new_img to: {output_path}")
                        except Exception as e:
                            print(f"Error saving realtime_new_img: {e}")
                    print("Generated layout functions visualization image (MORDOR format)")
                except Exception as e:
                    print(f"Error creating visualization image: {e}")
            else:
                # 시각화 생성 실패 시 다른 플레이스홀더
                viz_placeholder = Image.new('RGB', (800, 600), (240, 240, 240))
                try:
                    from PIL import ImageDraw, ImageFont
                    draw = ImageDraw.Draw(viz_placeholder)
                    
                    try:
                        font = ImageFont.truetype("arial.ttf", 24)
                    except:
                        font = ImageFont.load_default()
                    
                    text = "Layout Visualization Failed"
                    bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = bbox[2] - bbox[0]
                    x = (viz_placeholder.width - text_width) // 2
                    y = viz_placeholder.height // 2
                    
                    draw.text((x, y), text, fill=(100, 100, 100), font=font)
                    
                except Exception as e:
                    print(f"Error creating viz placeholder: {e}")
                
                structured_output_wandb = wandb.Image(viz_placeholder)
                print("Generated placeholder for layout visualization")
            
            # 6. 최종 출력을 row_data 형태로 저장
            row_data = {
                'template_id': template_id,
                'elements_metadata': json.dumps(elements_metadata),
                'predicted_output': json.dumps(predicted_output_dict),
                'layout_functions': layout_functions,
                'model_predicted': new_img_wandb,  # Realtime prediction result (MESSI format)
                'structured_output': structured_output_wandb,  # Layout functions visualization (MORDOR format)
            }
            
            return row_data
            
        except Exception as e:
            print(f"Error processing XML and image for {template_id}: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def save_to_wandb(self, results: List[Dict[str, Any]]):
        """
        결과를 wandb 테이블에 저장합니다.
        """
        if not results:
            print("No results to save")
            return
        
        # wandb 초기화
        wandb.init(project=self.project_name, name=f"process_{self.table_name}")
        
        try:
            # 첫 번째 결과를 확인하여 컬럼 구조 결정
            first_result = results[0]
            
            # 디버깅: 첫 번째 결과의 키들 출력
            print(f"DEBUG: First result keys: {list(first_result.keys())}")
            
            # XML 기반 입력인지 확인
            if 'elements_metadata' in first_result:
                # XML 기반 입력용 컬럼 정의 (이미지 컬럼 포함)
                columns = [
                    'template_id', 'elements_metadata', 'predicted_output', 'model_predicted', 'structured_output'
                ]
                print(f"DEBUG: Using XML-based columns: {columns}")
            else:
                # 기존 CSV 기반 입력용 컬럼 정의 (이미지 컬럼 포함)
                columns = [
                    'template_id', 'input_prompt', 'predicted_output', 
                    'model_predicted', 'structured_output'
                ]
                print(f"DEBUG: Using CSV-based columns: {columns}")
            
            # 디버깅: 각 컬럼의 데이터 존재 여부 확인
            print(f"DEBUG: Column data existence check:")
            for col in columns:
                value = first_result.get(col)
                value_exists = value is not None
                print(f"  {col}: {'EXISTS' if value_exists else 'MISSING'} (type: {type(value)})")
            
            # 이미지 컬럼들 확인 (XML 기반인 경우)
            if 'elements_metadata' in first_result:
                image_columns = ['model_predicted']
                print(f"DEBUG: Image columns (included in table):")
                for col in image_columns:
                    value = first_result.get(col)
                    value_exists = value is not None
                    print(f"  {col}: {'EXISTS' if value_exists else 'MISSING'} (type: {type(value)})")
            
            # 테이블 데이터 준비 (이미지 컬럼 제외)
            table_data = []
            for result in results:
                row = []
                for col in columns:
                    value = result.get(col, '')
                    row.append(value)
                table_data.append(row)
            
            print(f"DEBUG: Table data prepared with {len(table_data)} rows")
            
            # 디버깅: 첫 번째 행의 데이터 길이와 컬럼 수 확인
            if table_data:
                first_row = table_data[0]
                print(f"DEBUG: First row length: {len(first_row)}, Column count: {len(columns)}")
                print(f"DEBUG: First row data types: {[type(item) for item in first_row]}")
                
                # 각 컬럼의 실제 데이터 타입 확인
                for i, (col, value) in enumerate(zip(columns, first_row)):
                    print(f"DEBUG: Column {i}: {col} -> {type(value)}")
                    if col in ['model_predicted', 'model_predicted_output', 'new_img']:
                        print(f"DEBUG:   Image value exists: {value is not None}")
            
            try:
                # 테이블 생성 (간단한 방식)
                print(f"DEBUG: Creating table with {len(columns)} columns and {len(table_data)} rows")
                
                table = wandb.Table(columns=columns, data=table_data)
                print(f"DEBUG: Table created successfully")
                print(f"DEBUG: Table columns: {table.columns}")
                print(f"DEBUG: Table data length: {len(table.data)}")
                        
            except Exception as e:
                print(f"DEBUG: Error creating wandb table: {e}")
                import traceback
                traceback.print_exc()
                return
            
            # wandb에 테이블 로그
            wandb.log({self.table_name: table})
            
            print(f"Successfully saved {len(results)} results to wandb table '{self.table_name}'")
            print(f"Table columns: {columns}")
            print(f"DEBUG: Images included directly in table")
            
        except Exception as e:
            print(f"Error saving to wandb: {e}")
            import traceback
            traceback.print_exc()
        finally:
            wandb.finish()

    def run_pipeline(self, input_data: List[Dict[str, Any]], save_to_wandb: bool = True) -> List[Dict[str, Any]]:
        """
        전체 파이프라인을 실행합니다.
        """
        print(f"Starting pipeline with {len(input_data)} items")
        
        # 배치 처리
        results = self.process_batch(input_data)
        
        # wandb에 저장 (옵션)
        if save_to_wandb:
            self.save_to_wandb(results)
        
        return results

 
