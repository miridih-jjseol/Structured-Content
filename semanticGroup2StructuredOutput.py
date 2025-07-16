import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import math
import os
import glob

class LayoutType(Enum):
    VSTACK = "VStack"
    HSTACK = "HStack"
    ZSTACK = "ZStack"
    GROUP = "Group"
    GRAPH = "Graph"

@dataclass
class Position:
    x: int
    y: int
    width: int
    height: int

@dataclass
class ElementMetadata:
    tag: str
    text_content: str
    tbpe_id: str
    x: int
    y: int
    w: int
    h: int
    priority: str

@dataclass
class BaseElement:
    id: int
    type: str
    position: Optional[Position] = None
    children: Optional[List['BaseElement']] = None

@dataclass
class TextElement(BaseElement):
    content: str = ""
    
    def __post_init__(self):
        if self.type == "":
            self.type = "Text"

@dataclass
class SVGElement(BaseElement):
    svgData: str = ""
    
    def __post_init__(self):
        if self.type == "":
            self.type = "SVG"

@dataclass
class LayoutElement(BaseElement):
    gap: Optional[int] = None

@dataclass
class ZStackElement(BaseElement):
    def __post_init__(self):
        if self.type == "":
            self.type = "ZStack"

def calculate_group_bounds(elements: List[ElementMetadata]) -> Tuple[int, int, int, int]:
    """그룹 내 요소들의 전체 경계를 계산"""
    if not elements:
        return 0, 0, 0, 0
    
    min_x = min(elem.x for elem in elements)
    min_y = min(elem.y for elem in elements)
    max_x = max(elem.x + elem.w for elem in elements)
    max_y = max(elem.y + elem.h for elem in elements)
    
    return min_x, min_y, max_x - min_x, max_y - min_y

def check_overlap(elem1: ElementMetadata, elem2: ElementMetadata, tolerance: int = 5) -> bool:
    """두 요소가 겹치는지 확인"""
    # 사각형 겹침 감지
    return not (elem1.x + elem1.w <= elem2.x + tolerance or 
                elem2.x + elem2.w <= elem1.x + tolerance or 
                elem1.y + elem1.h <= elem2.y + tolerance or 
                elem2.y + elem2.h <= elem1.y + tolerance)

def find_overlapping_groups(elements: List[ElementMetadata]) -> List[List[ElementMetadata]]:
    """겹치는 요소들을 그룹으로 찾기"""
    if len(elements) <= 1:
        return [[elem] for elem in elements]
    
    visited = set()
    groups = []
    
    for i, elem in enumerate(elements):
        if i in visited:
            continue
            
        # 현재 요소와 겹치는 모든 요소들을 찾기
        current_group = [elem]
        visited.add(i)
        
        for j, other_elem in enumerate(elements):
            if j != i and j not in visited and check_overlap(elem, other_elem):
                current_group.append(other_elem)
                visited.add(j)
        
        groups.append(current_group)
    
    return groups

def determine_layout_type(group_elements: List[ElementMetadata]) -> Tuple[LayoutType, int]:
    """그룹의 레이아웃 타입과 간격을 결정"""
    if len(group_elements) <= 1:
        return LayoutType.GROUP, 0
    
    # 먼저 겹치는 요소가 있는지 확인 - 겹치면 우선적으로 ZStack
    overlapping_groups = find_overlapping_groups(group_elements)
    has_overlaps = any(len(group) > 1 for group in overlapping_groups)
    
    if has_overlaps:
        return LayoutType.ZSTACK, 0
    
    # 그룹 전체 경계 계산
    x, y, width, height = calculate_group_bounds(group_elements)
    
    # 요소들을 y좌표로 그룹핑 (같은 행)
    y_groups = {}
    tolerance = 20  # 같은 행으로 간주할 y좌표 차이 허용값
    
    for elem in group_elements:
        found_group = False
        for y_center in y_groups.keys():
            if abs(elem.y - y_center) <= tolerance:
                y_groups[y_center].append(elem)
                found_group = True
                break
        if not found_group:
            y_groups[elem.y] = [elem]
    
    # 요소들을 x좌표로 그룹핑 (같은 열)
    x_groups = {}
    for elem in group_elements:
        found_group = False
        for x_center in x_groups.keys():
            if abs(elem.x - x_center) <= tolerance:
                x_groups[x_center].append(elem)
                found_group = True
                break
        if not found_group:
            x_groups[elem.x] = [elem]
    
    # 행과 열의 개수
    num_rows = len(y_groups)
    num_cols = len(x_groups)
    
    # 그리드 패턴 감지
    is_grid = len(group_elements) == num_rows * num_cols
    
    if is_grid and num_rows > 1 and num_cols > 1:
        # 그리드 패턴인 경우, 가로/세로 비율로 결정
        if num_rows >= num_cols:
            # 세로로 더 많은 행 -> VStack (행들을 세로로 배치)
            sorted_rows = sorted(y_groups.items())
            y_gaps = []
            for i in range(1, len(sorted_rows)):
                current_row_bottom = max(elem.y + elem.h for elem in sorted_rows[i-1][1])
                next_row_top = min(elem.y for elem in sorted_rows[i][1])
                gap = next_row_top - current_row_bottom
                if gap > 0:
                    y_gaps.append(gap)
            
            avg_gap = int(sum(y_gaps) / len(y_gaps)) if y_gaps else 20
            return LayoutType.VSTACK, avg_gap
        else:
            # 가로로 더 많은 열 -> HStack (열들을 가로로 배치)
            sorted_cols = sorted(x_groups.items())
            x_gaps = []
            for i in range(1, len(sorted_cols)):
                current_col_right = max(elem.x + elem.w for elem in sorted_cols[i-1][1])
                next_col_left = min(elem.x for elem in sorted_cols[i][1])
                gap = next_col_left - current_col_right
                if gap > 0:
                    x_gaps.append(gap)
            
            avg_gap = int(sum(x_gaps) / len(x_gaps)) if x_gaps else 20
            return LayoutType.HSTACK, avg_gap
    
    elif num_rows == 1 and num_cols > 1:
        # 한 행에 여러 열 -> HStack
        sorted_by_x = sorted(group_elements, key=lambda e: e.x)
        x_gaps = []
        for i in range(1, len(sorted_by_x)):
            gap = sorted_by_x[i].x - (sorted_by_x[i-1].x + sorted_by_x[i-1].w)
            if gap > 0:
                x_gaps.append(gap)
        
        avg_gap = int(sum(x_gaps) / len(x_gaps)) if x_gaps else 20
        return LayoutType.HSTACK, avg_gap
    
    elif num_cols == 1 and num_rows > 1:
        # 한 열에 여러 행 -> VStack
        sorted_by_y = sorted(group_elements, key=lambda e: e.y)
        y_gaps = []
        for i in range(1, len(sorted_by_y)):
            gap = sorted_by_y[i].y - (sorted_by_y[i-1].y + sorted_by_y[i-1].h)
            if gap > 0:
                y_gaps.append(gap)
        
        avg_gap = int(sum(y_gaps) / len(y_gaps)) if y_gaps else 20
        return LayoutType.VSTACK, avg_gap
    
    else:
        # 불규칙한 배치 -> Group
        return LayoutType.GROUP, 0

def create_element_from_metadata(metadata: ElementMetadata, element_id: int) -> BaseElement:
    """메타데이터로부터 요소 생성"""
    position = Position(
        x=metadata.x,
        y=metadata.y,
        width=metadata.w,
        height=metadata.h
    )
    
    if metadata.tag == "TEXT":
        return TextElement(
            id=element_id,
            type="Text",
            position=position,
            content=metadata.text_content if metadata.text_content != "None" else ""
        )
    elif metadata.tag == "SVG" or metadata.tag == "LineShapeItem":
        return SVGElement(
            id=element_id,
            type="SVG",
            position=position,
            svgData=""  # SVG 데이터는 별도로 제공되어야 함
        )
    else:
        return BaseElement(
            id=element_id,
            type="Group",
            position=position
        )

def create_layout_element(layout_type: LayoutType, element_id: int, gap: int, children: List[BaseElement]) -> BaseElement:
    """레이아웃 타입에 따른 요소 생성"""
    if layout_type == LayoutType.VSTACK:
        return LayoutElement(
            id=element_id,
            type="VStack",
            gap=gap,
            children=children
        )
    elif layout_type == LayoutType.HSTACK:
        return LayoutElement(
            id=element_id,
            type="HStack",
            gap=gap,
            children=children
        )
    elif layout_type == LayoutType.ZSTACK:
        return BaseElement(
            id=element_id,
            type="ZStack",
            children=children
        )
    else:
        return BaseElement(
            id=element_id,
            type="Group",
            children=children
        )

def semanticGroup2StructuredOutput(
    semantic_group: Dict[str, Any],
    elements_metadata: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    시맨틱 그룹과 요소 메타데이터를 schema 형태로 변환
    
    Args:
        semantic_group: struct_label_group.json의 내용
        elements_metadata: elements_metadata.json의 내용
    
    Returns:
        schema 형태의 구조화된 출력
    """
    
    # 메타데이터를 ElementMetadata 객체로 변환
    metadata_objects = {}
    for elem_id, metadata in elements_metadata.items():
        metadata_objects[elem_id] = ElementMetadata(
            tag=metadata["tag"],
            text_content=metadata["text_content"],
            tbpe_id=metadata["tbpe_id"],
            x=metadata["x"],
            y=metadata["y"],
            w=metadata["w"],
            h=metadata["h"],
            priority=metadata["priority"]
        )
    
    result_elements = []
    element_id_counter = 1
    
    # 최상위 요소들 처리 (Parent Group 외부)
    top_level_elements = []
    for key, value in semantic_group.items():
        if key != "Parent Group 1" and value is None:
            if key in metadata_objects:
                element = create_element_from_metadata(metadata_objects[key], element_id_counter)
                top_level_elements.append(element)
                element_id_counter += 1
    
    # Parent Group 처리
    if "Parent Group 1" in semantic_group:
        parent_group = semantic_group["Parent Group 1"]
        
        # Subgroup들의 요소 수집 및 위치 정보로 행별 그룹핑
        subgroup_data = {}
        
        for subgroup_name, subgroup_content in parent_group.items():
            if subgroup_name.startswith("Subgroup") and isinstance(subgroup_content, dict):
                # 각 서브그룹의 요소들
                subgroup_items = []
                subgroup_metadata = []
                for item_key, item_value in subgroup_content.items():
                    if item_value is None and item_key in metadata_objects:
                        element = create_element_from_metadata(metadata_objects[item_key], element_id_counter)
                        subgroup_items.append(element)
                        subgroup_metadata.append(metadata_objects[item_key])
                        element_id_counter += 1
                
                if subgroup_items:
                    # 서브그룹 레이아웃 결정
                    layout_type, gap = determine_layout_type(subgroup_metadata)
                    subgroup_element = create_layout_element(layout_type, element_id_counter, gap, subgroup_items)
                    
                    # 서브그룹의 대표 위치 (첫 번째 요소의 위치)
                    representative_pos = subgroup_metadata[0] if subgroup_metadata else None
                    
                    subgroup_data[subgroup_name] = {
                        'element': subgroup_element,
                        'position': representative_pos
                    }
                    element_id_counter += 1
        
        # 서브그룹들을 y좌표로 행별 그룹핑
        tolerance = 50  # 같은 행으로 간주할 y좌표 차이
        rows = []
        
        for subgroup_name, data in subgroup_data.items():
            pos = data['position']
            if not pos:
                continue
                
            # 기존 행에 속하는지 확인
            placed = False
            for row in rows:
                if any(abs(pos.y - existing_pos.y) <= tolerance for _, existing_pos in row):
                    row.append((data['element'], pos))
                    placed = True
                    break
            
            if not placed:
                rows.append([(data['element'], pos)])
        
        # 각 행을 y좌표 순으로 정렬
        rows.sort(key=lambda row: min(pos.y for _, pos in row))
        
        # 각 행 내에서 x좌표 순으로 정렬
        for row in rows:
            row.sort(key=lambda item: item[1].x)
        
        # 행별로 HStack 생성 (2개 이상의 요소가 있는 경우)
        row_elements = []
        for i, row in enumerate(rows):
            if len(row) > 1:
                # 같은 행에 여러 요소가 있으면 HStack으로 감싸기
                row_children = [item[0] for item in row]
                
                # 행 내 요소들의 x 간격 계산
                x_gaps = []
                for j in range(1, len(row)):
                    gap = row[j][1].x - (row[j-1][1].x + row[j-1][1].w)
                    if gap > 0:
                        x_gaps.append(gap)
                
                row_gap = int(sum(x_gaps) / len(x_gaps)) if x_gaps else 20
                
                row_element = LayoutElement(
                    id=element_id_counter,
                    type="HStack",
                    gap=row_gap,
                    children=row_children
                )
                element_id_counter += 1
            else:
                # 행에 요소가 하나만 있으면 그대로 사용
                row_element = row[0][0]
            
            row_elements.append(row_element)
        
        # 전체 Parent Group을 VStack으로 구성
        if len(row_elements) > 1:
            # 행 간 y 간격 계산
            y_gaps = []
            for i in range(1, len(rows)):
                prev_row_bottom = max(pos.y + pos.h for _, pos in rows[i-1])
                curr_row_top = min(pos.y for _, pos in rows[i])
                gap = curr_row_top - prev_row_bottom
                if gap > 0:
                    y_gaps.append(gap)
            
            parent_gap = int(sum(y_gaps) / len(y_gaps)) if y_gaps else 20
            
            parent_element = LayoutElement(
                id=element_id_counter,
                type="VStack",
                gap=parent_gap,
                children=row_elements
            )
        else:
            parent_element = row_elements[0] if row_elements else BaseElement(
                id=element_id_counter,
                type="Group"
            )
        
        result_elements.append(parent_element)
    
    # 전체 구조를 감싸는 최상위 컨테이너
    all_elements = top_level_elements + result_elements
    
    # 최상단 레이아웃 결정 - 실제 최상위 구성 요소들 간의 관계로 결정
    if len(all_elements) > 1:
        # 최상위 요소들의 위치 정보 추출
        top_level_positions = []
        
        # top_level_elements에서 위치 정보 추출
        for elem in top_level_elements:
            if elem.position:
                # 임시로 ElementMetadata 형식으로 변환
                temp_metadata = ElementMetadata(
                    tag=elem.type,
                    text_content="",
                    tbpe_id=str(elem.id),
                    x=elem.position.x,
                    y=elem.position.y,
                    w=elem.position.width,
                    h=elem.position.height,
                    priority="0"
                )
                top_level_positions.append(temp_metadata)
        
        # parent_element의 경계 정보 추출 (첫 번째 자식의 위치 기준)
        if result_elements:
            parent_elem = result_elements[0]
            if hasattr(parent_elem, 'children') and parent_elem.children:
                first_child = parent_elem.children[0]
                if hasattr(first_child, 'children') and first_child.children:
                    first_grandchild = first_child.children[0]
                    if first_grandchild.position:
                        temp_metadata = ElementMetadata(
                            tag="ParentGroup",
                            text_content="",
                            tbpe_id="parent",
                            x=first_grandchild.position.x,
                            y=first_grandchild.position.y,
                            w=800,  # 대략적인 너비
                            h=600,  # 대략적인 높이
                            priority="1"
                        )
                        top_level_positions.append(temp_metadata)
        
        # 최상위 레이아웃 결정
        if len(top_level_positions) > 1:
            root_layout_type, root_gap = determine_layout_type(top_level_positions)
            root_element = create_layout_element(root_layout_type, 0, root_gap, all_elements)
        else:
            root_element = BaseElement(
                id=0,
                type="Group",
                children=all_elements
            )
    else:
        root_element = all_elements[0] if all_elements else BaseElement(id=0, type="Group")
    
    # 딕셔너리 형태로 변환
    def element_to_dict(element: BaseElement) -> Dict[str, Any]:
        result = {
            "id": element.id,
            "type": element.type,
        }
        
        if element.position:
            result["position"] = {
                "x": element.position.x,
                "y": element.position.y,
                "width": element.position.width,
                "height": element.position.height
            }
        
        if isinstance(element, TextElement):
            result["content"] = element.content
        elif isinstance(element, SVGElement):
            result["svgData"] = element.svgData
        elif isinstance(element, LayoutElement) and element.gap is not None:
            result["gap"] = element.gap
        
        if element.children:
            result["children"] = [element_to_dict(child) for child in element.children]
        
        return result
    
    return element_to_dict(root_element)

def semanticGroup2LayoutFunction(
    semantic_group: Dict[str, Any],
    elements_metadata: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    시맨틱 그룹과 요소 메타데이터를 Function Array 형태로 변환
    struct_label_group.json의 계층 구조를 따라 처리:
    1. Subgroup 내 요소들 먼저 처리
    2. Parent Group 레벨 처리
    3. 최상위 레벨 처리
    
    Args:
        semantic_group: struct_label_group.json의 내용
        elements_metadata: elements_metadata.json의 내용
    
    Returns:
        Function Array 형태의 레이아웃 함수 호출 리스트
    """
    
    # 메타데이터를 ElementMetadata 객체로 변환
    metadata_objects = {}
    for elem_id, metadata in elements_metadata.items():
        metadata_objects[elem_id] = ElementMetadata(
            tag=metadata["tag"],
            text_content=metadata["text_content"],
            tbpe_id=metadata["tbpe_id"],
            x=metadata["x"],
            y=metadata["y"],
            w=metadata["w"],
            h=metadata["h"],
            priority=metadata["priority"]
        )
    
    layout_functions = []
    group_id_counter = 1000  # 그룹 ID는 1000부터 시작
    
    # 최상위 요소들 처리 (Parent Group 외부)
    top_level_element_ids = []
    for key, value in semantic_group.items():
        if not key.startswith("Parent Group") and value is None:
            if key in metadata_objects:
                top_level_element_ids.append(key)
    
    # Parent Group들 처리
    parent_group_ids = []
    
    # 모든 Parent Group을 찾아서 처리
    for parent_group_key in semantic_group.keys():
        if parent_group_key.startswith("Parent Group") and isinstance(semantic_group[parent_group_key], dict):
            parent_group = semantic_group[parent_group_key]
            
            # 1단계: Parent Group 전체를 재귀적으로 처리
            def process_group_recursively(group_dict, group_name_prefix=""):
                """그룹을 재귀적으로 처리하여 모든 하위 그룹들을 layout function으로 변환"""
                nonlocal group_id_counter, layout_functions
                
                current_group_elements = []
                current_group_metadata = []
                nested_group_ids = []
                
                for key, value in group_dict.items():
                    if isinstance(value, dict):
                        # 중첩된 그룹인 경우 재귀적으로 처리
                        nested_group_id = process_group_recursively(value, f"{group_name_prefix}_{key}")
                        if nested_group_id:
                            nested_group_ids.append(nested_group_id)
                    elif value is None and key in metadata_objects:
                        # 실제 요소인 경우
                        current_group_elements.append(key)
                        current_group_metadata.append(metadata_objects[key])
                
                # 현재 레벨의 요소들과 중첩된 그룹들이 있다면 layout function 생성
                if current_group_elements or nested_group_ids:
                    if current_group_metadata:
                        # 현재 레벨에 실제 요소들이 있는 경우 레이아웃 결정
                        layout_type, gap = determine_layout_type(current_group_metadata)
                        
                        # 현재 그룹의 모든 요소들 (실제 요소 + 중첩 그룹들)
                        all_elements = current_group_elements + nested_group_ids
                        
                        # 그룹 ID 생성
                        current_group_id = f"group_{group_id_counter}"
                        group_id_counter += 1
                        
                        # Layout function 생성
                        group_layout_function = {
                            "function": "applyLayout",
                            "layoutType": layout_type.value,
                            "elementIds": all_elements,
                            "spacing": gap if gap > 0 else 16,
                            "alignment": "center",
                            "groupId": current_group_id
                        }
                        layout_functions.append(group_layout_function)
                        
                        return current_group_id
                    elif len(nested_group_ids) == 1:
                        # 중첩 그룹이 하나만 있는 경우
                        return nested_group_ids[0]
                    elif len(nested_group_ids) > 1:
                        # 여러 중첩 그룹들을 묶기
                        current_group_id = f"group_{group_id_counter}"
                        group_id_counter += 1
                        
                        group_layout_function = {
                            "function": "applyLayout",
                            "layoutType": "Group",
                            "elementIds": nested_group_ids,
                            "spacing": 16,
                            "alignment": "center",
                            "groupId": current_group_id
                        }
                        layout_functions.append(group_layout_function)
                        
                        return current_group_id
                
                return None
            
            # Parent Group 전체를 처리
            current_parent_group_id = process_group_recursively(parent_group, parent_group_key)
            
            # 현재 Parent Group ID를 리스트에 추가
            if current_parent_group_id:
                parent_group_ids.append(current_parent_group_id)
    
    # 3단계: 최상위 레벨에서 전체 구조 결정
    if top_level_element_ids and parent_group_ids:
        # 최상위 요소들과 Parent Groups을 함께 배치
        all_top_level_ids = top_level_element_ids + parent_group_ids
        
        # 최상위 요소들의 위치 정보 수집
        top_level_positions = []
        for elem_id in top_level_element_ids:
            if elem_id in metadata_objects:
                top_level_positions.append(metadata_objects[elem_id])
        
        # Parent Groups의 대략적인 위치 추가
        for parent_group_key in semantic_group.keys():
            if parent_group_key.startswith("Parent Group") and isinstance(semantic_group[parent_group_key], dict):
                # Parent Group 내 모든 요소들의 경계 계산
                all_parent_metadata = []
                parent_group = semantic_group[parent_group_key]
                
                for key, value in parent_group.items():
                    if key.startswith("Subgroup") and isinstance(value, dict):
                        for elem_key, elem_value in value.items():
                            if elem_value is None and elem_key in metadata_objects:
                                all_parent_metadata.append(metadata_objects[elem_key])
                    elif value is None and key in metadata_objects:
                        all_parent_metadata.append(metadata_objects[key])
                
                if all_parent_metadata:
                    x, y, w, h = calculate_group_bounds(all_parent_metadata)
                    temp_metadata = ElementMetadata(
                        tag="ParentGroup",
                        text_content="",
                        tbpe_id=parent_group_key,
                        x=x, y=y, w=w, h=h,
                        priority="1"
                    )
                    top_level_positions.append(temp_metadata)
        
        # 최상위 레이아웃 결정
        if len(top_level_positions) > 1:
            root_layout_type, root_gap = determine_layout_type(top_level_positions)
            
            root_layout_function = {
                "function": "applyLayout",
                "layoutType": root_layout_type.value,
                "elementIds": all_top_level_ids,
                "spacing": root_gap if root_gap > 0 else 16,
                "alignment": "center",
                "groupId": "root_group"
            }
            layout_functions.append(root_layout_function)
            
    elif parent_group_ids and not top_level_element_ids:
        # Parent Groups만 있는 경우
        if len(parent_group_ids) > 1:
            # 여러 Parent Group들의 위치 정보 수집
            parent_group_positions = []
            for parent_group_key in semantic_group.keys():
                if parent_group_key.startswith("Parent Group") and isinstance(semantic_group[parent_group_key], dict):
                    # Parent Group 내 모든 요소들의 경계 계산
                    all_parent_metadata = []
                    parent_group = semantic_group[parent_group_key]
                    
                    for key, value in parent_group.items():
                        if key.startswith("Subgroup") and isinstance(value, dict):
                            for elem_key, elem_value in value.items():
                                if elem_value is None and elem_key in metadata_objects:
                                    all_parent_metadata.append(metadata_objects[elem_key])
                        elif value is None and key in metadata_objects:
                            all_parent_metadata.append(metadata_objects[key])
                    
                    if all_parent_metadata:
                        x, y, w, h = calculate_group_bounds(all_parent_metadata)
                        temp_metadata = ElementMetadata(
                            tag="ParentGroup",
                            text_content="",
                            tbpe_id=parent_group_key,
                            x=x, y=y, w=w, h=h,
                            priority="1"
                        )
                        parent_group_positions.append(temp_metadata)
            
            # Parent Groups 간의 레이아웃 결정
            if len(parent_group_positions) > 1:
                root_layout_type, root_gap = determine_layout_type(parent_group_positions)
                
                root_layout_function = {
                    "function": "applyLayout",
                    "layoutType": root_layout_type.value,
                    "elementIds": parent_group_ids,
                    "spacing": root_gap if root_gap > 0 else 16,
                    "alignment": "center",
                    "groupId": "root_group"
                }
                layout_functions.append(root_layout_function)
        # Parent Group이 하나만 있는 경우는 이미 처리됨
        
    elif top_level_element_ids and not parent_group_ids:
        # 최상위 요소들만 있는 경우
        if len(top_level_element_ids) > 1:
            top_level_positions = []
            for elem_id in top_level_element_ids:
                if elem_id in metadata_objects:
                    top_level_positions.append(metadata_objects[elem_id])
            
            if len(top_level_positions) > 1:
                root_layout_type, root_gap = determine_layout_type(top_level_positions)
                
                root_layout_function = {
                    "function": "applyLayout",
                    "layoutType": root_layout_type.value,
                    "elementIds": top_level_element_ids,
                    "spacing": root_gap if root_gap > 0 else 16,
                    "alignment": "center",
                    "groupId": "root_group"
                }
                layout_functions.append(root_layout_function)
    
    return layout_functions

def determine_subgroup_layout(
    subgroup_positions: List[Tuple[str, ElementMetadata]],
    elements_metadata: Dict[str, Dict[str, Any]]
) -> Tuple[str, List[List[str]], Dict[str, Any]]:
    """
    Subgroup들의 위치를 고려하여 최적의 레이아웃을 결정
    
    사용자의 요구사항:
    1. nxm 직사각형으로 표현 가능하면 최상단 그룹을 VStack/HStack으로 묶기
    2. 그렇지 않으면 Graph로 표현하고 각 subgroup을 노드로 표현
    
    Args:
        subgroup_positions: (group_id, position_metadata) 튜플들의 리스트
    
    Returns:
        (layout_type, groups, extra_info): 레이아웃 타입, 그룹핑된 subgroup ID들, 추가 정보
    """
    if len(subgroup_positions) <= 1:
        return "single", [[pos[0] for pos in subgroup_positions]], {}
    
    # 위치 정보 추출
    positions = [(group_id, pos.x, pos.y, pos.w, pos.h) for group_id, pos in subgroup_positions]
    
    # 1. nxm 직사각형 패턴 감지
    alignment_tolerance = 50  # 정렬 허용 오차를 늘림
    
    # 행별 그룹핑 (같은 y축 좌표 기준)
    horizontal_groups = []
    for group_id, x, y, w, h in positions:
        placed = False
        for h_group in horizontal_groups:
            if any(abs(y - other_y) <= alignment_tolerance for _, _, other_y, _, _ in h_group):
                h_group.append((group_id, x, y, w, h))
                placed = True
                break
        if not placed:
            horizontal_groups.append([(group_id, x, y, w, h)])
    
    # 열별 그룹핑 (같은 x축 좌표 기준)
    vertical_groups = []
    for group_id, x, y, w, h in positions:
        placed = False
        for v_group in vertical_groups:
            if any(abs(x - other_x) <= alignment_tolerance for _, other_x, _, _, _ in v_group):
                v_group.append((group_id, x, y, w, h))
                placed = True
                break
        if not placed:
            vertical_groups.append([(group_id, x, y, w, h)])
    
    num_rows = len(horizontal_groups)
    num_cols = len(vertical_groups)
    
    # 2. 직사각형 패턴 확인 (더 유연하게)
    is_rectangular_grid = False
    
    # 완전한 그리드 패턴 (nxm)
    if num_rows * num_cols == len(positions):
        row_sizes = [len(h_group) for h_group in horizontal_groups]
        if all(size == row_sizes[0] for size in row_sizes):
            is_rectangular_grid = True
    
    # 1차원 배열 패턴도 직사각형으로 간주
    elif num_rows == 1 or num_cols == 1:
        is_rectangular_grid = True
    
    # 3. 레이아웃 타입 결정
    if is_rectangular_grid:
        # 직사각형 패턴: 요소들의 실제 배치에 따라 레이아웃 결정
        
        # 1차원 배열의 경우 명확한 처리
        if num_rows == 1 and num_cols > 1:
            # 한 행에 여러 열 - 가로로 배치 -> HStack
            layout_type = "horizontal"
            sorted_positions = sorted(positions, key=lambda item: item[1])  # x좌표 순 정렬
            groups = [[item[0] for item in sorted_positions]]
            return layout_type, groups, {}
            
        elif num_cols == 1 and num_rows > 1:
            # 한 열에 여러 행 - 세로로 배치 -> VStack  
            layout_type = "vertical"
            sorted_positions = sorted(positions, key=lambda item: item[2])  # y좌표 순 정렬
            groups = [[item[0] for item in sorted_positions]]
            return layout_type, groups, {}
        
        # 2차원 그리드의 경우 행과 열의 개수로 결정
        elif num_rows <= num_cols:
            # 행보다 열이 많거나 같은 경우 - 가로 배치가 우세 -> HStack 기반
            if num_rows == 1:
                # 단일 행 - HStack
                layout_type = "horizontal"
                sorted_positions = sorted(positions, key=lambda item: item[1])  # x좌표 순 정렬
                groups = [[item[0] for item in sorted_positions]]
            else:
                # 여러 행이지만 가로 배치가 우세 - 열별 VStack 후 HStack
                layout_type = "grid_vertical"
                sorted_v_groups = sorted(vertical_groups, key=lambda g: min(x for _, x, _, _, _ in g))
                groups = []
                for v_group in sorted_v_groups:
                    # 각 열 내에서 y좌표 순으로 정렬
                    sorted_col = sorted(v_group, key=lambda item: item[2])
                    groups.append([item[0] for item in sorted_col])
            return layout_type, groups, {}
        
        else:  # num_rows > num_cols
            # 열보다 행이 많은 경우 - 세로 배치가 우세 -> VStack 기반
            if num_cols == 1:
                # 단일 열 - VStack
                layout_type = "vertical"
                sorted_positions = sorted(positions, key=lambda item: item[2])  # y좌표 순 정렬
                groups = [[item[0] for item in sorted_positions]]
            else:
                # 여러 열이지만 세로 배치가 우세 - 행별 HStack 후 VStack
                layout_type = "grid_horizontal"
                # 행별로 정렬된 그룹 생성
                sorted_h_groups = sorted(horizontal_groups, key=lambda g: min(y for _, _, y, _, _ in g))
                groups = []
                for h_group in sorted_h_groups:
                    # 각 행 내에서 x좌표 순으로 정렬
                    sorted_row = sorted(h_group, key=lambda item: item[1])
                    groups.append([item[0] for item in sorted_row])
            return layout_type, groups, {}
    
    else:
        # 불규칙한 배치: Graph로 표현
        # 각 subgroup을 노드로 생성
        nodes = []
        for group_id, x, y, w, h in positions:
            nodes.append({
                "id": group_id,
                "x": x + w/2,  # 중심점 x
                "y": y + h/2,  # 중심점 y
                "width": w,
                "height": h
            })
        
        # 간단한 연결 규칙: 가장 가까운 노드들을 연결
        edges = []
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i < j:  # 중복 방지
                    distance = ((node1["x"] - node2["x"])**2 + (node1["y"] - node2["y"])**2)**0.5
                    edges.append({
                        "from": node1["id"],
                        "to": node2["id"],
                        "distance": distance
                    })
        
        # 거리 기준으로 정렬하여 가장 가까운 몇 개만 선택
        edges.sort(key=lambda e: e["distance"])
        max_edges = min(len(edges), len(nodes))  # 최대 노드 수만큼만 연결
        selected_edges = edges[:max_edges]
        
        graph_info = {
            "nodes": nodes,
            "edges": selected_edges,
            "layout": "force_directed"
        }
        
        return "graph", [[node["id"] for node in nodes]], graph_info

# 사용 예시
if __name__ == "__main__":
    # 데이터셋 폴더 경로
    dataset_base_path = "/data/shared/jjkim/dataset"
    
    # 사용할 데이터셋들 (처음 5개만 테스트)
    # dataset_folders = sorted(glob.glob(f"{dataset_base_path}/*"))[:5]
    dataset_folders = [os.path.join(dataset_base_path, i) for i in ["252097", "153465"]]
    
    for i, dataset_folder in enumerate(dataset_folders):
        dataset_name = os.path.basename(dataset_folder)
        
        # 필요한 파일들이 존재하는지 확인
        struct_file = os.path.join(dataset_folder, "struct_label_group.json")
        metadata_file = os.path.join(dataset_folder, "elements_metadata.json")
        
        if not os.path.exists(struct_file) or not os.path.exists(metadata_file):
            print(f"Dataset {dataset_name}: Missing required files, skipping...")
            continue
        
        try:
            print(f"\n{'='*80}")
            print(f"DATASET {i+1}: {dataset_name}")
            print(f"{'='*80}")
            
            # 테스트 데이터 로드
            with open(struct_file, "r", encoding="utf-8") as f:
                semantic_group = json.load(f)
            
            with open(metadata_file, "r", encoding="utf-8") as f:
                elements_metadata = json.load(f)
            
            # Layout Function 배열 출력
            print("=== Layout Function Array ===")
            layout_functions = semanticGroup2LayoutFunction(semantic_group, elements_metadata)
            
            if layout_functions:
                # YAML 형태로 출력
                try:
                    import yaml
                    print(yaml.dump(layout_functions, default_flow_style=False, allow_unicode=True))
                except ImportError:
                    # yaml 모듈이 없으면 JSON 형태로 출력
                    print(json.dumps(layout_functions, indent=2, ensure_ascii=False))
            else:
                print("No layout functions generated.")
                
        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {str(e)}")
            continue