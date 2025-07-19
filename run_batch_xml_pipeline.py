#!/usr/bin/env python3
"""
Run Batch XML-based Integrated Pipeline
Input: XML directory, thumbnail directory
Output: wandb.table (same as process_wandb_csv.py)
"""

import os
import sys
import glob
import argparse
from pathlib import Path
from PIL import Image
from typing import Optional, List, Dict, Any
from integrated_pipeline import IntegratedStructuredContentPipeline

def find_matching_files(xml_dir: str, thumbnail_dir: str):
    """
    XML 파일과 썸네일 이미지의 매칭 파일들을 찾습니다.
    """
    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
    thumbnail_files = glob.glob(os.path.join(thumbnail_dir, "*.png"))
    
    # 파일명에서 숫자 부분 추출하여 매칭
    xml_mapping = {}
    for xml_file in xml_files:
        filename = os.path.basename(xml_file)
        template_id = filename.replace('.xml', '')
        xml_mapping[template_id] = xml_file
    
    thumbnail_mapping = {}
    for thumb_file in thumbnail_files:
        filename = os.path.basename(thumb_file)
        template_id = filename.replace('.png', '')
        thumbnail_mapping[template_id] = thumb_file
    
    # 매칭되는 파일들만 반환
    matching_pairs = []
    for template_id in xml_mapping:
        if template_id in thumbnail_mapping:
            matching_pairs.append({
                'template_id': template_id,
                'xml_file': xml_mapping[template_id],
                'thumbnail_file': thumbnail_mapping[template_id]
            })
    
    return matching_pairs

def process_with_model_outputs(
    matching_pairs: List[Dict[str, Any]], 
    model_outputs: Optional[Dict[str, Image.Image]] = None,
    save_to_wandb: bool = True,
    limit: Optional[int] = None,
    table_name: str = "xml_model_results"
) -> List[Dict[str, Any]]:
    """
    모델 출력과 함께 XML들을 배치 처리합니다.
    
    Args:
        matching_pairs: XML과 썸네일 매칭 쌍들
        model_outputs: {template_id: PIL_Image} 형태의 모델 출력들
        save_to_wandb: wandb에 저장할지 여부
        limit: 처리할 파일 개수 제한
    """
    if limit:
        matching_pairs = matching_pairs[:limit]
    
    pipeline = IntegratedStructuredContentPipeline(
        project_name="MORDOR-structured-output-validation-jyjang",
        table_name=table_name
    )
    
    results = []
    failed_count = 0
    
    for i, pair in enumerate(matching_pairs):
        print(f"\nProcessing {i+1}/{len(matching_pairs)}: {pair['template_id']}")
        
        try:
            # 모델 출력이 있으면 사용, 없으면 None
            model_output = None
            if model_outputs and pair['template_id'] in model_outputs:
                model_output = model_outputs[pair['template_id']]
                print(f"Using model output for {pair['template_id']}")
            
            result = pipeline.process_xml_and_image(
                xml_file_path=pair['xml_file'],
                thumbnail_image_path=pair['thumbnail_file'],
                template_id=pair['template_id'],
                model_predicted_output=model_output
            )
            
            if result:
                results.append(result)
                print(f"✓ Successfully processed: {pair['template_id']}")
            else:
                failed_count += 1
                print(f"✗ Failed to process: {pair['template_id']}")
                
        except Exception as e:
            failed_count += 1
            print(f"✗ Error processing {pair['template_id']}: {e}")
    
    # 결과 저장
    if results and save_to_wandb:
        pipeline.save_to_wandb(results)
    
    print(f"\n" + "=" * 60)
    print(f"BATCH PROCESSING COMPLETED")
    print(f"=" * 60)
    print(f"Total processed: {len(results)}")
    print(f"Failed: {failed_count}")
    if len(results) + failed_count > 0:
        print(f"Success rate: {len(results)/(len(results)+failed_count)*100:.1f}%")
    
    return results

def main():
    """
    배치 처리 메인 함수
    """
    parser = argparse.ArgumentParser(description='Batch process XML files and thumbnail images')
    parser.add_argument('--xml_dir', default='/data/shared/jjseol/data/Mordor_validation_66/xml_sheets/', help='XML files directory')
    parser.add_argument('--thumbnail_dir', default='/data/shared/jjseol/data/Mordor_validation_66/thumbnails/', help='Thumbnail images directory')
    parser.add_argument('--table_name', default='xml_batch_results', help='Table name for wandb')
    parser.add_argument('--enable_realtime_prediction', action='store_true', help='Enable real-time LLM prediction to generate new_img')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of files to process')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--template_id', default=None, help='Process specific template ID only')
    parser.add_argument('--api_endpoint', default='http://211.47.48.147:8000/generate', help='vLLM API endpoint URL')
    parser.add_argument('--api_model_name', default='default', help='Model name for API requests')
    parser.add_argument('--api_key', default=None, help='API key for authentication (optional)')
    
    args = parser.parse_args()
    
    # 디렉토리 존재 확인
    if not os.path.exists(args.xml_dir):
        print(f"Error: XML directory not found: {args.xml_dir}")
        return
    
    if not os.path.exists(args.thumbnail_dir):
        print(f"Error: Thumbnail directory not found: {args.thumbnail_dir}")
        return
    
    
    # 매칭되는 파일들 찾기
    matching_pairs = find_matching_files(args.xml_dir, args.thumbnail_dir)
    
    if not matching_pairs:
        print("No matching XML and thumbnail files found")
        return
    
    # 특정 template_id만 처리하는 경우
    if args.template_id:
        matching_pairs = [pair for pair in matching_pairs if pair['template_id'] == args.template_id]
        if not matching_pairs:
            print(f"Template ID {args.template_id} not found")
            return
    
    # 처리할 파일 개수 제한
    if args.limit:
        matching_pairs = matching_pairs[:args.limit]
    
    print(f"Found {len(matching_pairs)} matching XML and thumbnail pairs")
    print(f"XML directory: {args.xml_dir}")
    print(f"Thumbnail directory: {args.thumbnail_dir}")
    print(f"Real-time prediction enabled: {args.enable_realtime_prediction}")
    print(f"Save to wandb: {not args.no_wandb}")
    print("=" * 60)
    
    # 파이프라인 초기화
    pipeline = IntegratedStructuredContentPipeline(
        project_name="MORDOR-structured-output-validation-jyjang",
        table_name=args.table_name,
        enable_realtime_prediction=args.enable_realtime_prediction,
        api_endpoint=args.api_endpoint,
        api_model_name=args.api_model_name,
        api_key=args.api_key
    )
    
    # 배치 처리
    results = []
    failed_count = 0
    
    for i, pair in enumerate(matching_pairs):
        print(f"\nProcessing {i+1}/{len(matching_pairs)}: {pair['template_id']}")
        
        try:
            # 모델 예측 이미지 로드 (new_img)
            model_predicted_image = None
            
            result = pipeline.process_xml_and_image(
                xml_file_path=pair['xml_file'],
                thumbnail_image_path=pair['thumbnail_file'],
                template_id=pair['template_id'],
                model_predicted_output=model_predicted_image  # new_img 사용
            )
            
            if result:
                results.append(result)
                print(f"✓ Successfully processed: {pair['template_id']}")
            else:
                failed_count += 1
                print(f"✗ Failed to process: {pair['template_id']}")
                
        except Exception as e:
            failed_count += 1
            print(f"✗ Error processing {pair['template_id']}: {e}")
    
    # 결과 저장
    if results:
        if not args.no_wandb:
            pipeline.save_to_wandb(results)
        
        print(f"\n" + "=" * 60)
        print(f"BATCH PROCESSING COMPLETED")
        print(f"=" * 60)
        print(f"Total processed: {len(results)}")
        print(f"Failed: {failed_count}")
        print(f"Success rate: {len(results)/(len(results)+failed_count)*100:.1f}%")
        
        # 샘플 결과 출력
        if results:
            sample_result = results[0]
            print(f"\nSample result:")
            print(f"Template ID: {sample_result['template_id']}")
            print(f"Layout Functions Count: {len(sample_result.get('layout_functions', []))}")
            print(f"Has Model Predicted: {'Yes' if sample_result.get('model_predicted') else 'No'}")
            print(f"Has Model Predicted Output: {'Yes' if sample_result.get('model_predicted_output') else 'No'}")
            
            # 디버깅: 실제 결과 키들을 출력
            print(f"\nDebugging - Result keys: {list(sample_result.keys())}")
            print(f"Model predicted type: {type(sample_result.get('model_predicted'))}")
            print(f"Model predicted output type: {type(sample_result.get('model_predicted_output'))}")
            print(f"New img type: {type(sample_result.get('new_img'))}")
            
            # wandb 테이블에 저장될 컬럼들 확인
            print(f"\nWandB table columns that will be saved:")
            if 'elements_metadata' in sample_result:
                # XML 기반 입력용 컬럼 정의
                columns = [
                    'template_id', 'elements_metadata', 'predicted_output', 
                    'structured_output', 'new_img', 'model_predicted', 'model_predicted_output'
                ]
                print(f"XML-based columns: {columns}")
                
                # 각 컬럼의 값 존재 여부 확인
                for col in columns:
                    value = sample_result.get(col)
                    print(f"  {col}: {'EXISTS' if value else 'MISSING'} (type: {type(value)})")
            else:
                print("CSV-based columns will be used")
    else:
        print("No successful results to save")

def test_single_file():
    """
    단일 파일 테스트용 함수
    """
    # 사용자가 제공한 ID로 테스트
    template_id = "e22aab65-9b1e-4cc8-8846-eb2abf2d478a"
    
    # 해당 ID의 파일이 있는지 확인
    xml_dir = "/data/shared/jjseol/data/Mordor_validation_66/xml_sheets/"
    thumbnail_dir = "/data/shared/jjseol/data/Mordor_validation_66/thumbnails/"
    
    matching_pairs = find_matching_files(xml_dir, thumbnail_dir)
    
    # 처음 몇 개 파일로 테스트
    if matching_pairs:
        print(f"Available template IDs (first 10): {[pair['template_id'] for pair in matching_pairs[:10]]}")
        
        # 첫 번째 파일로 테스트
        test_pair = matching_pairs[0]
        print(f"\nTesting with: {test_pair['template_id']}")
        
        pipeline = IntegratedStructuredContentPipeline(
            project_name="MORDOR-structured-output-validation-jyjang",
            table_name="xml_test_results",
            enable_realtime_prediction=False,  # 테스트에서는 기본적으로 비활성화
            api_endpoint="http://211.47.48.147:8000/generate",
            api_model_name="default",
            api_key=None
        )
        
        result = pipeline.process_xml_and_image(
            xml_file_path=test_pair['xml_file'],
            thumbnail_image_path=test_pair['thumbnail_file'],
            template_id=test_pair['template_id'],
            model_predicted_output=None  # 기본값으로 None 사용
        )
        
        if result:
            print(f"✓ Test successful!")
            print(f"Layout Functions Count: {len(result.get('layout_functions', []))}")
            print(f"Has Model Predicted: {'Yes' if result.get('model_predicted') else 'No'}")
            print(f"Has Model Predicted Output: {'Yes' if result.get('model_predicted_output') else 'No'}")
        else:
            print("✗ Test failed")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage:")
        print("  python run_batch_xml_pipeline.py                    # Process all files")
        print("  python run_batch_xml_pipeline.py --limit 5          # Process first 5 files")
        print("  python run_batch_xml_pipeline.py --template_id 21283 # Process specific ID")
        print("  python run_batch_xml_pipeline.py --no_wandb         # Don't save to wandb")
        print("  python run_batch_xml_pipeline.py --enable_realtime_prediction # Enable real-time LLM prediction")
        print("  python run_batch_xml_pipeline.py --api_endpoint http://211.47.48.147:8000/generate # Set vLLM API endpoint")
        print("  python run_batch_xml_pipeline.py --api_model_name default # Set model name")
        print("  python run_batch_xml_pipeline.py --api_key YOUR_API_KEY # Set API key")
        print("  python run_batch_xml_pipeline.py --test             # Run test with first file")
        print("  python run_batch_xml_pipeline.py --table_name xml_batch_results # Set table name for wandb")
        print()
        print("Model images directory should contain PNG files named {template_id}.png")
        print("(Generated from test_clean_code_pipeline.ipynb as new_img)")
        print()
        print("Real-time prediction will generate new_img on-the-fly using custom vLLM server")
        print("(Configure API endpoint, model name, and API key as needed)")
        print("Default endpoint: http://211.47.48.147:8000/generate (vLLM FastAPI server)")
        print()
        
        # 매칭되는 파일들 개수 확인
        matching_pairs = find_matching_files(
            "/data/shared/jjseol/data/Mordor_validation_66/xml_sheets/",
            "/data/shared/jjseol/data/Mordor_validation_66/thumbnails/"
        )
        print(f"Found {len(matching_pairs)} matching XML and thumbnail pairs")
        
        if matching_pairs:
            print(f"Sample template IDs: {[pair['template_id'] for pair in matching_pairs[:5]]}")
    elif '--test' in sys.argv:
        test_single_file()
    else:
        main() 