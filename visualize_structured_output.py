import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, ConnectionPatch
import matplotlib.font_manager as fm
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import math
import os
import glob
from PIL import Image
from semanticGroup2StructuredOutput import semanticGroup2StructuredOutput, semanticGroup2LayoutFunction
import platform

class StructuredContentVisualizer:
    def __init__(self, width=1920, height=1080, scale_factor=0.4):
        """
        구조화된 콘텐츠를 시각화하는 클래스
        
        Args:
            width: 캔버스 너비 (기본값, 자동 조정 가능)
            height: 캔버스 높이 (기본값, 자동 조정 가능)
            scale_factor: 스케일링 팩터
        """
        self.width = width
        self.height = height
        self.scale_factor = scale_factor
        self.fig = None
        self.ax = None
        
        # 한글 폰트 설정
        plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Malgun Gothic', 'Apple Gothic']
        if platform.system() == 'Windows':
            plt.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']
        elif platform.system() == 'Darwin':  # macOS
            plt.rcParams['font.family'] = ['Apple Gothic', 'DejaVu Sans']
        else:  # Linux
            plt.rcParams['font.family'] = ['DejaVu Sans', 'Liberation Sans']
            
        # 마이너스 기호 깨짐 방지
        plt.rcParams['axes.unicode_minus'] = False
        
        # 레이아웃 타입별 색상
        self.layout_colors = {
            'VStack': '#3b82f6',   # 파란색
            'HStack': '#10b981',   # 초록색  
            'ZStack': '#f59e0b',   # 주황색
            'Group': '#8b5cf6',    # 보라색
            'Graph': '#ef4444',    # 빨간색
            'SVG': '#10b981',      # 초록색
            'Image': '#ef4444',    # 빨간색
            'Chart': '#8b5cf6',    # 보라색
            'Table': '#06b6d4',    # 청록색
            'Video': '#f97316',    # 주황색
            'Text': '#374151',     # 회색
            'Component': '#ec4899' # 분홍색
        }
    
    def _calculate_optimal_bounds(self, layout_functions: List[Dict[str, Any]], elements_metadata: Dict[str, Dict[str, Any]]) -> Tuple[int, int, int, int]:
        """
        레이아웃 함수들과 요소 메타데이터를 기반으로 최적의 시각화 영역을 계산
        
        Returns:
            (min_x, min_y, max_x, max_y): 최적 영역의 경계
        """
        min_x = float('inf')
        min_y = float('inf') 
        max_x = float('-inf')
        max_y = float('-inf')
        
        # 모든 요소들의 위치를 수집
        for func in layout_functions:
            element_ids = func.get('elementIds', [])
            
            # Graph 타입인 경우 노드 좌표 사용
            if func.get('layoutType') == 'Graph':
                graph_info = func.get('graphInfo', {})
                nodes = graph_info.get('nodes', [])
                for node in nodes:
                    x, y = node.get('x', 0), node.get('y', 0)
                    # 노드 크기 고려 (반지름 20 + 여백)
                    min_x = min(min_x, x - 40)
                    min_y = min(min_y, y - 40)
                    max_x = max(max_x, x + 40)
                    max_y = max(max_y, y + 40)
            
            # 일반 요소들의 경우
            for elem_id in element_ids:
                if elem_id in elements_metadata:
                    metadata = elements_metadata[elem_id]
                    x, y, w, h = metadata['x'], metadata['y'], metadata['w'], metadata['h']
                    
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    max_x = max(max_x, x + w)
                    max_y = max(max_y, y + h)
        
        # 유효하지 않은 경우 기본값 반환
        if min_x == float('inf'):
            return 0, 0, self.width, self.height
        
        # 여백 추가 (10%)
        padding_x = max(50, (max_x - min_x) * 0.1)
        padding_y = max(50, (max_y - min_y) * 0.1)
        
        min_x = max(0, min_x - padding_x)
        min_y = max(0, min_y - padding_y)
        max_x = max_x + padding_x
        max_y = max_y + padding_y
        
        print(f"Optimal bounds calculated: ({min_x:.0f}, {min_y:.0f}) to ({max_x:.0f}, {max_y:.0f})")
        return int(min_x), int(min_y), int(max_x), int(max_y)

    def create_figure(self, title="Structured Content Visualization", background_image_path=None, layout_functions=None, elements_metadata=None):
        """시각화를 위한 matplotlib figure 생성"""
        # 최적 영역 계산 (레이아웃 함수가 제공된 경우)
        if layout_functions and elements_metadata:
            min_x, min_y, max_x, max_y = self._calculate_optimal_bounds(layout_functions, elements_metadata)
            # 계산된 영역으로 캔버스 크기 업데이트
            self.width = max_x - min_x
            self.height = max_y - min_y
            self.offset_x = min_x
            self.offset_y = min_y
        else:
            # 기본값 사용
            self.offset_x = 0
            self.offset_y = 0
        
        self.fig, self.ax = plt.subplots(1, 1, figsize=(
            self.width * self.scale_factor / 100, 
            self.height * self.scale_factor / 100
        ))
        
        # 좌표 범위 설정 (오프셋 적용)
        self.ax.set_xlim(self.offset_x, self.offset_x + self.width)
        self.ax.set_ylim(self.offset_y, self.offset_y + self.height)
        self.ax.invert_yaxis()  # Y축 뒤집기 (웹 좌표계와 맞추기)
        self.ax.set_aspect('equal')
        
        # 좌표계 및 격자 제거
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_xlabel('')
        self.ax.set_ylabel('')
        
        # 여백 제거 - 그림이 화면 가득 차도록
        self.ax.margins(0)
        
        # 배경 이미지 로드 및 표시
        if background_image_path and os.path.exists(background_image_path):
            self._load_background_image(background_image_path)
        else:
            # 배경 이미지가 없으면 연한 회색 배경
            self.ax.set_facecolor('#f8fafc')
        
        # 제목만 표시 (좌표계는 제거)
        if title:
            self.fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        return self.fig, self.ax
    
    def _load_background_image(self, image_path):
        """배경 이미지 로드 및 표시"""
        try:
            # PIL로 이미지 로드
            pil_image = Image.open(image_path)
            
            # 이미지 크기를 캔버스 크기에 맞게 조정
            pil_image = pil_image.resize((self.width, self.height), Image.Resampling.LANCZOS)
            
            # numpy 배열로 변환
            img_array = np.array(pil_image)
            
            # 이미지를 원래 위치(0,0)에 배치 (투명도 조정)
            self.ax.imshow(img_array, extent=[0, self.width, self.height, 0], alpha=0.7, aspect='equal')
            
            print(f"Background image loaded: {image_path}")
            
        except Exception as e:
            print(f"Error loading background image {image_path}: {e}")
            self.ax.set_facecolor('#f8fafc')
    
    def _find_background_image(self, dataset_path, dataset_name):
        """데이터셋 폴더에서 배경 이미지 찾기"""
        if not dataset_path:
            return None
            
        # 일반적인 이미지 파일 확장자들
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        
        # 가능한 이미지 파일 패턴들
        possible_patterns = [
            f"{dataset_name}.*",  # dataset_name.png 등
            "0.*",                # 0.png (첫 번째 페이지)
            "slide_0.*",          # slide_0.png
            "page_0.*",           # page_0.png
            "image.*",            # image.png
            "preview.*",          # preview.png
            "thumbnail.*"         # thumbnail.png
        ]
        
        for pattern in possible_patterns:
            for ext in image_extensions:
                # 패턴에서 확장자 부분을 실제 확장자로 교체
                file_pattern = pattern.replace(".*", ext)
                full_path = os.path.join(dataset_path, file_pattern)
                
                if os.path.exists(full_path):
                    return full_path
        
        # 폴더 내 모든 이미지 파일 검색
        for file in os.listdir(dataset_path):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                return os.path.join(dataset_path, file)
        
        return None
    
    def visualize_structured_output(self, structured_data: Dict[str, Any], save_path: Optional[str] = None, 
                                   background_image_path: Optional[str] = None, dataset_path: Optional[str] = None,
                                   dataset_name: Optional[str] = None):
        """
        구조화된 출력 데이터를 시각화
        
        Args:
            structured_data: semanticGroup2StructuredOutput의 출력 결과
            save_path: 저장할 파일 경로 (선택적)
            background_image_path: 배경 이미지 경로 (선택적)
            dataset_path: 데이터셋 폴더 경로 (배경 이미지 자동 검색용)
            dataset_name: 데이터셋 이름 (배경 이미지 자동 검색용)
        """
        # 배경 이미지 자동 검색
        if not background_image_path and dataset_path:
            background_image_path = self._find_background_image(dataset_path, dataset_name or "")
        
        self.create_figure("Structured Content Layout", background_image_path)
        
        # 요소들을 평면화하여 수집
        elements = self._flatten_elements(structured_data)
        
        # 레이아웃 컨테이너들 수집
        layout_containers = self._collect_layout_containers(structured_data)
        
        # 1. 레이아웃 컨테이너들 먼저 그리기 (뒤쪽 레이어)
        for container in layout_containers:
            self._draw_layout_container(container)
        
        # 2. 개별 요소들 그리기 (앞쪽 레이어)
        for element in elements:
            self._draw_element(element)
        
        # 3. 그래프 구조가 있다면 노드와 엣지 그리기
        if structured_data.get('type') == 'Graph':
            self._draw_graph_structure(structured_data)
        
        # 범례 추가 (배경에 따라 투명도 조정)
        self._add_legend(background_image_path is not None)
        
        # 여백 최소화하여 화면 가득 채우기
        plt.tight_layout(pad=0.1)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.02)
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def visualize_layout_functions(self, layout_functions: List[Dict[str, Any]], 
                                 elements_metadata: Dict[str, Dict[str, Any]], 
                                 save_path: Optional[str] = None,
                                 background_image_path: Optional[str] = None,
                                 dataset_path: Optional[str] = None,
                                 dataset_name: Optional[str] = None):
        """
        레이아웃 함수 배열을 시각화
        
        Args:
            layout_functions: semanticGroup2LayoutFunction의 출력 결과
            elements_metadata: 요소 메타데이터
            save_path: 저장할 파일 경로 (선택적)
            background_image_path: 배경 이미지 경로 (선택적)
            dataset_path: 데이터셋 폴더 경로 (배경 이미지 자동 검색용)
            dataset_name: 데이터셋 이름 (배경 이미지 자동 검색용)
        """
        # 배경 이미지 자동 검색
        if not background_image_path and dataset_path:
            background_image_path = self._find_background_image(dataset_path, dataset_name or "")
        
        self.create_figure("Layout Functions Visualization", background_image_path)
        
        # 요소 메타데이터를 시각화 (투명도 높게) - 비활성화됨
        # for elem_id, metadata in elements_metadata.items():
        #     self._draw_element_from_metadata(elem_id, metadata, alpha=0.3)
        
        # 레이아웃 함수들을 시각화
        for i, func in enumerate(layout_functions):
            self._draw_layout_function(func, elements_metadata, i, layout_functions)
        
        # 범례 추가
        self._add_legend(background_image_path is not None)
        
        # 여백 최소화하여 화면 가득 채우기
        plt.tight_layout(pad=0.1)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.02)
            print(f"Layout functions visualization saved to: {save_path}")
        
        plt.show()
    
    def _flatten_elements(self, data: Dict[str, Any], elements: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """구조화된 데이터에서 모든 요소를 평면화"""
        if elements is None:
            elements = []
        
        # 현재 요소가 리프 요소인지 확인
        if self._is_leaf_element(data):
            elements.append(data)
        
        # 자식 요소들 재귀적으로 처리
        if 'children' in data and isinstance(data['children'], list):
            for child in data['children']:
                self._flatten_elements(child, elements)
        
        return elements
    
    def _collect_layout_containers(self, data: Dict[str, Any], containers: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """레이아웃 컨테이너들을 수집"""
        if containers is None:
            containers = []
        
        # 현재 요소가 레이아웃 컨테이너인지 확인
        if data.get('type') in ['VStack', 'HStack', 'ZStack', 'Group', 'Graph']:
            containers.append(data)
        
        # 자식 요소들 재귀적으로 처리
        if 'children' in data and isinstance(data['children'], list):
            for child in data['children']:
                self._collect_layout_containers(child, containers)
        
        return containers
    
    def _is_leaf_element(self, element: Dict[str, Any]) -> bool:
        """리프 요소인지 확인"""
        return element.get('type') in ['Text', 'SVG', 'Image', 'Chart', 'Table', 'Video', 'Component']
    
    def _draw_element(self, element: Dict[str, Any], alpha: float = 0.6):
        """개별 요소를 그리기 - 박스 그리기 기능 비활성화"""
        if 'position' not in element:
            return
        
        # 기본 요소 박스 그리기 기능 제거됨
        # 아래 코드들은 모두 주석처리되어 박스를 그리지 않음
        
        # pos = element['position']
        # x, y = pos['x'], pos['y']
        # w, h = pos.get('width', 100), pos.get('height', 30)
        # 
        # element_type = element.get('type', 'Unknown')
        # color = self.layout_colors.get(element_type, '#6b7280')
        # 
        # # 요소 박스 그리기 (알파 블렌딩 적용) - 비활성화
        # if element_type == 'Text':
        #     # 텍스트 요소는 흰색 배경에 테두리 (투명도 적용)
        #     rect = FancyBboxPatch(
        #         (x, y), w, h,
        #         boxstyle="round,pad=2",
        #         facecolor='white',
        #         edgecolor=color,
        #         linewidth=2,
        #         alpha=alpha
        #     )
        #     self.ax.add_patch(rect)
        #     
        #     # 텍스트 내용 표시
        #     content = element.get('content', '')
        #     if content:
        #         # 텍스트가 박스 안에 들어가도록 자르기
        #         max_chars = max(1, int(w / 10))  # 한글 고려하여 조정
        #         display_text = content[:max_chars] + '...' if len(content) > max_chars else content
        #         self.ax.text(x + w/2, y + h/2, display_text, 
        #                    ha='center', va='center', fontsize=9, color=color, fontweight='bold')
        # 
        # elif element_type == 'SVG':
        #     # SVG 요소는 연한 배경
        #     rect = FancyBboxPatch(
        #         (x, y), w, h,
        #         boxstyle="round,pad=2",
        #         facecolor=color,
        #         alpha=alpha * 0.5,
        #         edgecolor=color,
        #         linewidth=2
        #     )
        #     self.ax.add_patch(rect)
        #     self.ax.text(x + w/2, y + h/2, 'SVG', 
        #                 ha='center', va='center', fontsize=9, color=color, fontweight='bold')
        # 
        # else:
        #     # 기타 요소들
        #     rect = FancyBboxPatch(
        #         (x, y), w, h,
        #         boxstyle="round,pad=2",
        #         facecolor=color,
        #         alpha=alpha * 0.7,
        #         edgecolor=color,
        #         linewidth=2
        #     )
        #     self.ax.add_patch(rect)
        #     self.ax.text(x + w/2, y + h/2, element_type, 
        #                 ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    
    def _draw_layout_container(self, container: Dict[str, Any], alpha: float = 0.8):
        """레이아웃 컨테이너를 그리기"""
        if 'position' not in container:
            return
        
        pos = container['position']
        x, y = pos['x'], pos['y']
        w, h = pos.get('width', 200), pos.get('height', 100)
        
        container_type = container.get('type', 'Group')
        color = self.layout_colors.get(container_type, '#6b7280')
        
        # 컨테이너 외곽선 그리기 (더 굵고 투명도 적용)
        if container_type == 'Group':
            # Group은 점선으로
            rect = patches.Rectangle(
                (x-3, y-3), w+6, h+6,
                linewidth=3,
                edgecolor=color,
                facecolor='none',
                linestyle='--',
                alpha=alpha
            )
        else:
            # 기타 레이아웃은 실선으로
            rect = patches.Rectangle(
                (x-3, y-3), w+6, h+6,
                linewidth=3,
                edgecolor=color,
                facecolor='none',
                alpha=alpha
            )
        
        self.ax.add_patch(rect)
        
        # 레이아웃 타입 라벨 추가 (배경 투명도 조정)
        self.ax.text(x + 6, y - 10, container_type, 
                    fontsize=11, color='white', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=3", facecolor=color, alpha=0.9, edgecolor='none'))
    
    def _draw_graph_structure(self, graph_data: Dict[str, Any]):
        """그래프 구조 (노드와 엣지) 그리기"""
        if 'nodes' not in graph_data or 'edges' not in graph_data:
            return
        
        nodes = graph_data['nodes']
        edges = graph_data['edges']
        
        print(f"  Drawing graph: {len(nodes)} nodes, {len(edges)} edges")
        
        # 노드 위치 계산
        node_positions = {}
        for node in nodes:
            if 'x' in node and 'y' in node:
                node_positions[node['id']] = (node['x'], node['y'])
                print(f"    Node position: {node['id']} at ({node['x']}, {node['y']})")
        
        # 엣지 그리기 (더 명확한 화살표와 색상)
        for i, edge in enumerate(edges):
            from_id = edge['from']
            to_id = edge['to']
            
            if from_id in node_positions and to_id in node_positions:
                from_pos = node_positions[from_id]
                to_pos = node_positions[to_id]
                
                print(f"    Drawing edge {i+1}: {from_id} → {to_id}")
                
                # 화살표로 연결 (방향성 표시) - 가시성 향상
                arrow = patches.FancyArrowPatch(
                    from_pos, to_pos,
                    arrowstyle='->',
                    shrinkA=22, shrinkB=22,  # 노드에서 충분히 떨어뜨리기 (노드 반지름 20 + 여백 2)
                    color='#dc2626',  # 더 진한 빨간색
                    linewidth=4,  # 더 굵게
                    alpha=0.9,  # 더 선명하게
                    mutation_scale=25,  # 화살표 크기 증가
                    capstyle='round',
                    zorder=100  # 다른 요소들보다 위에 그리기
                )
                self.ax.add_patch(arrow)
        
        # 노드 그리기 (더 크고 명확하게)
        for i, node in enumerate(nodes):
            if node['id'] in node_positions:
                pos = node_positions[node['id']]
                
                # 노드 원 (더 크게) - 다른 요소들보다 위에
                circle = plt.Circle(pos, 20, facecolor='#3b82f6', edgecolor='#1e40af', linewidth=3, alpha=0.95, zorder=200)
                self.ax.add_patch(circle)
                
                # 노드 번호 (중앙에) - 최상위
                self.ax.text(pos[0], pos[1], str(i+1), 
                           ha='center', va='center', fontsize=12, color='white', fontweight='bold', zorder=300)
                
                # 노드 라벨 (아래쪽에) - 최상위 - 비활성화됨
                # display_id = node['id']
                # if len(display_id) > 10:  # ID가 너무 길면 줄이기
                #     display_id = display_id[:8] + '...'
                # 
                # self.ax.text(pos[0], pos[1] + 35, display_id, 
                #            ha='center', va='center', fontsize=9, fontweight='bold',
                #            bbox=dict(boxstyle="round,pad=2", facecolor='white', alpha=0.95, edgecolor='#3b82f6'),
                #            zorder=250)
                
                print(f"    Drew node {i+1}: {node['id']} at {pos}")
        
        print(f"  Graph structure drawn successfully")
    
    def _draw_element_from_metadata(self, elem_id: str, metadata: Dict[str, Any], alpha: float = 0.5):
        """메타데이터로부터 요소 그리기 - 박스 그리기 기능 비활성화"""
        x, y = metadata['x'], metadata['y']
        w, h = metadata['w'], metadata['h']
        
        # 너무 벗어나는 요소는 그리지 않음
        margin = 200  # 적당한 여백
        if (x < -margin or x > self.width + margin or 
            y < -margin or y > self.height + margin):
            return
        
        # 기본 요소 박스 그리기 기능 제거됨
        # 아래 코드들은 모두 주석처리되어 박스를 그리지 않음
        
        # tag = metadata.get('tag', 'Unknown')
        # 
        # # tag가 list인 경우 첫 번째 element 사용
        # if isinstance(tag, list) and len(tag) > 0:
        #     tag = tag[0]
        # elif isinstance(tag, list):
        #     tag = 'Unknown'
        #     
        # color = self.layout_colors.get(tag, '#6b7280')
        # 
        # # 요소 박스 그리기 (투명도 적용) - 비활성화
        # rect = FancyBboxPatch(
        #     (x, y), w, h,
        #     boxstyle="round,pad=2",
        #     facecolor=color,
        #     alpha=alpha,
        #     edgecolor=color,
        #     linewidth=2
        # )
        # self.ax.add_patch(rect)
        # 
        # # 레이아웃 관련 요소들만 텍스트 표시 (VStack, HStack, ZStack, Group, Graph)
        # layout_types = ['VStack', 'HStack', 'ZStack', 'Group', 'Graph']
        # if tag in layout_types:
        #     # 요소 ID와 태그 표시
        #     display_text = f"{elem_id}\n({tag})"
        #     self.ax.text(x + w/2, y + h/2, display_text, 
        #                 ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    
    def _draw_layout_function(self, func: Dict[str, Any], elements_metadata: Dict[str, Dict[str, Any]], index: int, all_layout_functions: List[Dict[str, Any]]):
        """레이아웃 함수를 시각화"""
        element_ids = func.get('elementIds', [])
        layout_type = func.get('layoutType', 'Group')
        
        if not element_ids:
            return
        
        # Graph 타입인 경우 특별 처리
        if layout_type == 'Graph':
            # graphInfo에서 노드와 엣지 정보 추출
            graph_info = func.get('graphInfo', {})
            nodes = graph_info.get('nodes', [])
            edges = graph_info.get('edges', [])
            
            print(f"  Graph visualization: {len(nodes)} nodes, {len(edges)} edges")
            for i, node in enumerate(nodes):
                print(f"    Node {i+1}: {node['id']} at ({node['x']}, {node['y']})")
            
            # 그래프 구조 그리기
            if nodes:
                graph_data = {
                    'nodes': nodes,
                    'edges': edges
                }
                self._draw_graph_structure(graph_data)
                
                # Graph 영역 주변에 작은 컨테이너 박스 그리기
                node_xs = [node['x'] for node in nodes]
                node_ys = [node['y'] for node in nodes]
                
                min_x = min(node_xs) - 35  # 노드 반지름(20) + 여백(15)
                max_x = max(node_xs) + 35
                min_y = min(node_ys) - 45  # 노드 반지름(20) + 라벨 여백(25)
                max_y = max(node_ys) + 45
                
                # 작은 컨테이너 박스 (점선으로 눈에 덜 띄게)
                color = self.layout_colors.get(layout_type, '#ef4444')
                
                rect = patches.Rectangle(
                    (min_x, min_y), max_x - min_x, max_y - min_y,
                    linewidth=1,
                    edgecolor=color,
                    facecolor='none',
                    linestyle='--',  # 점선
                    alpha=0.6
                )
                self.ax.add_patch(rect)
                
                # 함수 라벨 - 컨테이너 상단에 작게 배치
                avg_x = (min_x + max_x) / 2
                label_text = f"F{index+1}: {graph_info.get('pattern', 'unknown').upper()}"
                
                self.ax.text(avg_x, min_y - 10, label_text, 
                            fontsize=8, color='white', fontweight='bold',
                            bbox=dict(boxstyle="round,pad=2", facecolor=color, alpha=0.9),
                            ha='center', va='center', zorder=250)
                
                print(f"  Graph container: ({min_x:.1f}, {min_y:.1f}) to ({max_x:.1f}, {max_y:.1f})")
                print(f"  Graph label placed at ({avg_x:.1f}, {min_y - 10:.1f})")
            else:
                print(f"  No nodes found for Graph layout function")
            
            return
        
        # 너무 벗어나는 요소 제외를 위한 margin 설정
        margin = 200  # 적당한 여백
        
        # 관련된 요소들의 경계 계산 (group ID들 해결)
        positions = []
        expanded_element_ids = []
        
        # 먼저 실제 요소들과 그룹 ID들을 분리
        for elem_id in element_ids:
            if elem_id in elements_metadata:
                # 실제 요소인 경우 - 범위 체크
                metadata = elements_metadata[elem_id]
                x, y, w, h = metadata['x'], metadata['y'], metadata['w'], metadata['h']
                
                # 너무 벗어나는 요소는 제외
                if (x < -margin or x > self.width + margin or 
                    y < -margin or y > self.height + margin):
                    print(f"Excluding element {elem_id} - out of bounds: ({x}, {y})")
                    continue
                    
                positions.append((x, y, w, h))
                expanded_element_ids.append(elem_id)
            elif elem_id.startswith('group_'):
                # 그룹 ID인 경우 - 다른 layout function들에서 해당 그룹을 찾아 확장
                group_elements = self._find_group_elements(elem_id, all_layout_functions)
                for group_elem_id in group_elements:
                    if group_elem_id in elements_metadata:
                        metadata = elements_metadata[group_elem_id]
                        x, y, w, h = metadata['x'], metadata['y'], metadata['w'], metadata['h']
                        
                        # 너무 벗어나는 요소는 제외
                        if (x < -margin or x > self.width + margin or 
                            y < -margin or y > self.height + margin):
                            print(f"Excluding group element {group_elem_id} - out of bounds: ({x}, {y})")
                            continue
                            
                        positions.append((x, y, w, h))
                        expanded_element_ids.append(group_elem_id)
        
        if not positions:
            print(f"No valid elements for layout function F{index+1}: {layout_type}")
            return
        
        # 전체 경계 계산
        min_x = min(pos[0] for pos in positions)
        min_y = min(pos[1] for pos in positions)
        max_x = max(pos[0] + pos[2] for pos in positions)
        max_y = max(pos[1] + pos[3] for pos in positions)
        
        # Graph가 포함된 경우 컨테이너 박스를 더 조심스럽게 그리기
        has_graph_in_elements = self._check_for_graph_in_elements(expanded_element_ids, all_layout_functions)
        
        # 레이아웃 컨테이너 그리기 (Graph가 있으면 더 조심스럽게)
        color = self.layout_colors.get(layout_type, '#6b7280')
        
        if has_graph_in_elements:
            # Graph가 포함된 경우 점선으로 덜 눈에 띄게
            padding = 3
            rect = patches.Rectangle(
                (min_x - padding, min_y - padding), max_x - min_x + padding * 2, max_y - min_y + padding * 2,
                linewidth=1,
                edgecolor=color,
                facecolor='none',
                linestyle='--',  # 점선
                alpha=0.4  # 더 투명하게
            )
        else:
            # 일반적인 경우
            padding = 1
            rect = patches.Rectangle(
                (min_x - padding, min_y - padding), max_x - min_x + padding * 2, max_y - min_y + padding * 2,
                linewidth=1.5,
                edgecolor=color,
                facecolor='none',
                alpha=0.7
            )
        
        self.ax.add_patch(rect)
        
        # 함수 라벨 추가 - 상위 구조는 오른쪽에 배치
        label_text = f"F{index+1}: {layout_type}"
        
        # 상위 구조 판단 기준:
        # 1. 영역 크기가 일정 이상인 경우 (width * height > 500000)
        # 2. 또는 포함된 요소 개수가 많은 경우 (expanded_element_ids > 10)
        area = (max_x - min_x) * (max_y - min_y)
        is_high_level = area > 500000 or len(expanded_element_ids) > 10
        
        if is_high_level:
            # 상위 구조는 오른쪽에 라벨 배치
            label_x = max_x - 50  # 오른쪽 끝에서 50px 왼쪽
            label_y = min_y - 8
        else:
            # 하위 구조는 왼쪽에 라벨 배치 (기존 방식)
            label_x = min_x
            label_y = min_y - 8
        
        self.ax.text(label_x, label_y, label_text, 
                    fontsize=8, color='white', fontweight='bold',  # 폰트 크기 10에서 8로 더 줄임
                    bbox=dict(boxstyle="round,pad=1", facecolor=color, alpha=0.9))  # 패딩 2에서 1로 더 줄임
    
    def _find_group_elements(self, group_id: str, all_layout_functions: List[Dict[str, Any]]) -> List[str]:
        """그룹 ID에 해당하는 실제 요소들을 찾기"""
        # 해당 group_id를 가진 layout function 찾기
        for func in all_layout_functions:
            if func.get('groupId') == group_id:
                return self._expand_element_ids(func.get('elementIds', []), all_layout_functions)
        
        # 그룹을 찾지 못한 경우 빈 리스트 반환
        return []
    
    def _expand_element_ids(self, element_ids: List[str], all_layout_functions: List[Dict[str, Any]]) -> List[str]:
        """element_ids를 실제 요소들로 재귀적으로 확장"""
        expanded = []
        
        for elem_id in element_ids:
            if elem_id.startswith('group_'):
                # 그룹 ID인 경우 재귀적으로 확장
                group_elements = self._find_group_elements(elem_id, all_layout_functions)
                expanded.extend(group_elements)
            else:
                # 실제 요소 ID인 경우
                expanded.append(elem_id)
        
        return expanded
    
    def _check_for_graph_in_elements(self, element_ids: List[str], all_layout_functions: List[Dict[str, Any]]) -> bool:
        """요소들 중에 Graph 타입이 포함되어 있는지 확인"""
        for elem_id in element_ids:
            if elem_id.startswith('group_'):
                # 그룹 ID인 경우 해당 함수를 찾아서 확인
                for func in all_layout_functions:
                    if func.get('groupId') == elem_id:
                        if func.get('layoutType') == 'Graph':
                            return True
                        # 재귀적으로 하위 요소들도 확인
                        if self._check_for_graph_in_elements(func.get('elementIds', []), all_layout_functions):
                            return True
        return False
    
    def _add_legend(self, has_background=False):
        """범례 추가"""
        legend_elements = []
        for element_type, color in self.layout_colors.items():
            legend_elements.append(patches.Patch(color=color, label=element_type))
        
        # 배경 이미지가 있으면 범례 배경을 더 투명하게
        legend_alpha = 0.8 if has_background else 0.9
        
        # 범례를 그림 내부 우상단에 배치
        legend = self.ax.legend(handles=legend_elements, loc='upper right', 
                               framealpha=legend_alpha, fancybox=True, shadow=True,
                               fontsize=8, ncol=2)  # 폰트 크기 줄이고 2열로 배치
        
        # 범례 프레임 스타일 조정
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('gray')
        legend.get_frame().set_linewidth(0.5)

def main():
    """메인 함수 - 예제 데이터로 시각화 테스트"""
    # 데이터셋 경로
    dataset_base_path = "/data/shared/jjkim/dataset"
    
    # 테스트용 데이터셋 (몇 개만 선택)
    test_datasets = ["252097", "153465"]
    
    visualizer = StructuredContentVisualizer()
    
    for dataset_name in test_datasets:
        dataset_folder = os.path.join(dataset_base_path, dataset_name)
        
        # 필요한 파일들 확인
        struct_file = os.path.join(dataset_folder, "struct_label_group.json")
        metadata_file = os.path.join(dataset_folder, "elements_metadata.json")
        
        if not os.path.exists(struct_file) or not os.path.exists(metadata_file):
            print(f"Dataset {dataset_name}: Missing required files, skipping...")
            continue
        
        try:
            print(f"\n{'='*60}")
            print(f"Visualizing Dataset: {dataset_name}")
            print(f"{'='*60}")
            
            # 데이터 로드
            with open(struct_file, "r", encoding="utf-8") as f:
                semantic_group = json.load(f)
            
            with open(metadata_file, "r", encoding="utf-8") as f:
                elements_metadata = json.load(f)
            
            # 1. 구조화된 출력 생성 및 시각화
            print("Generating structured output...")
            structured_output = semanticGroup2StructuredOutput(semantic_group, elements_metadata)
            
            print("Visualizing structured output...")
            visualizer.visualize_structured_output(
                structured_output,
                save_path=f"structured_output_{dataset_name}.png",
                dataset_path=dataset_folder,
                dataset_name=dataset_name
            )
            
            # 2. 레이아웃 함수 생성 및 시각화
            print("Generating layout functions...")
            layout_functions = semanticGroup2LayoutFunction(semantic_group, elements_metadata)
            
            print("Visualizing layout functions...")
            visualizer.visualize_layout_functions(
                layout_functions,
                elements_metadata,
                save_path=f"layout_functions_{dataset_name}.png",
                dataset_path=dataset_folder,
                dataset_name=dataset_name
            )
            
            print(f"Visualization completed for dataset: {dataset_name}")
            
        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 