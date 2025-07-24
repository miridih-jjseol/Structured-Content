import json
from typing import Dict, List, Any, Optional, Tuple, Protocol
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

class LayoutElement(Protocol):
    """Layout 계산에 필요한 공통 속성을 정의하는 Protocol"""
    tbpe_id: str
    x: int
    y: int
    w: int
    h: int

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
class GroupMetadata:
    tag: str
    tbpe_id: str
    x: int
    y: int
    w: int
    h: int
    element_tags: List[str]  # 구성하는 element들의 tag 정보

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

def calculate_group_bounds(elements: List[LayoutElement]) -> Tuple[int, int, int, int]:
    """그룹 내 요소들의 전체 경계를 계산"""
    if not elements:
        return 0, 0, 0, 0
    
    # x, y, w, h가 모두 0인 요소들을 제외
    valid_elements = [elem for elem in elements if not (elem.x == 0 and elem.y == 0 and elem.w == 0 and elem.h == 0)]
    
    if not valid_elements:
        return 0, 0, 0, 0
    
    min_x = min(elem.x for elem in valid_elements)
    min_y = min(elem.y for elem in valid_elements)
    max_x = max(elem.x + elem.w for elem in valid_elements)
    max_y = max(elem.y + elem.h for elem in valid_elements)
    
    return min_x, min_y, max_x - min_x, max_y - min_y

def check_overlap(elem1: LayoutElement, elem2: LayoutElement, tolerance: int = 5) -> bool:
    """두 요소가 겹치는지 확인"""
    # 사각형 겹침 감지
    return not (elem1.x + elem1.w <= elem2.x + tolerance or 
                elem2.x + elem2.w <= elem1.x + tolerance or 
                elem1.y + elem1.h <= elem2.y + tolerance or 
                elem2.y + elem2.h <= elem1.y + tolerance)

def find_overlapping_groups(elements: List[LayoutElement]) -> List[List[LayoutElement]]:
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

def determine_layout_type(group_elements: List[LayoutElement], allow_graph_patterns: bool = True) -> Tuple[LayoutType, int, Optional[Dict[str, Any]]]:
    """그룹의 레이아웃 타입과 간격을 결정 (mxn grid 우선, 아니면 GRAPH, 마지막에 ZStack)"""
    if len(group_elements) <= 1:
        return LayoutType.GROUP, 0, None
    
    print(f"  LAYOUT: Determining layout for {len(group_elements)} elements (allow_graph_patterns={allow_graph_patterns})")
    
    # 겹치는 요소 확인 (나중에 사용)
    overlapping_groups = find_overlapping_groups(group_elements)
    has_overlaps = any(len(group) > 1 for group in overlapping_groups)
    print(f"  LAYOUT: Has overlaps: {has_overlaps}")
    
    # 1. 먼저 mxn linear grid 패턴 확인
    
    # 그룹 전체 경계 계산
    x, y, width, height = calculate_group_bounds(group_elements)
    
    # 요소들을 y좌표로 그룹핑 (같은 행)
    y_groups = {}
    # width, height 각각에 대한 tolerance 계산
    avg_width = sum(elem.w for elem in group_elements) / len(group_elements) if group_elements else 20
    avg_height = sum(elem.h for elem in group_elements) / len(group_elements) if group_elements else 20
    width_tolerance = int(avg_width) * 0.3  # 같은 열 판단용 (x좌표 차이)
    height_tolerance = int(avg_height) * 0.3  # 같은 행 판단용 (y좌표 차이)
    
    for elem in group_elements:
        found_group = False
        for y_center in y_groups.keys():
            if abs(elem.y - y_center) <= height_tolerance:
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
            if abs(elem.x - x_center) <= width_tolerance:
                x_groups[x_center].append(elem)
                found_group = True
                break
        if not found_group:
            x_groups[elem.x] = [elem]
    
    # 행과 열의 개수
    num_rows = len(y_groups)
    num_cols = len(x_groups)
    
    print(f"  LAYOUT: Grid analysis - {num_rows} rows × {num_cols} cols = {num_rows * num_cols} (total: {len(group_elements)})")
    
    # 그리드 패턴 감지
    is_grid = len(group_elements) == num_rows * num_cols
    
    print(f"  LAYOUT: Is perfect grid: {is_grid}")
    
    if is_grid and num_rows > 1 and num_cols > 1:
        # 그리드 패턴인 경우, 가로/세로 비율로 결정
        print(f"  LAYOUT: ✅ Perfect mxn grid detected")
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
            print(f"  LAYOUT: Returning VStack with gap {avg_gap}")
            return LayoutType.VSTACK, avg_gap, None
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
            print(f"  LAYOUT: Returning HStack with gap {avg_gap}")
            return LayoutType.HSTACK, avg_gap, None
    
    elif num_rows == 1 and num_cols > 1:
        # 한 행에 여러 열 -> HStack
        print(f"  LAYOUT: ✅ Single row linear layout detected")
        sorted_by_x = sorted(group_elements, key=lambda e: e.x)
        x_gaps = []
        for i in range(1, len(sorted_by_x)):
            gap = sorted_by_x[i].x - (sorted_by_x[i-1].x + sorted_by_x[i-1].w)
            if gap > 0:
                x_gaps.append(gap)
        
        avg_gap = int(sum(x_gaps) / len(x_gaps)) if x_gaps else 20
        print(f"  LAYOUT: Returning HStack with gap {avg_gap}")
        return LayoutType.HSTACK, avg_gap, None
    
    elif num_cols == 1 and num_rows > 1:
        # 한 열에 여러 행 -> VStack
        print(f"  LAYOUT: ✅ Single column linear layout detected")
        sorted_by_y = sorted(group_elements, key=lambda e: e.y)
        y_gaps = []
        for i in range(1, len(sorted_by_y)):
            gap = sorted_by_y[i].y - (sorted_by_y[i-1].y + sorted_by_y[i-1].h)
            if gap > 0:
                y_gaps.append(gap)
        
        avg_gap = int(sum(y_gaps) / len(y_gaps)) if y_gaps else 20
        print(f"  LAYOUT: Returning VStack with gap {avg_gap}")
        return LayoutType.VSTACK, avg_gap, None
    
    else:
        # 2. mxn linear grid가 아니면 GRAPH 패턴 감지 시도 (허용된 경우에만)
        if allow_graph_patterns:
            print(f"  LAYOUT: Not a linear grid, trying GRAPH patterns...")
            graph_info = detect_graph_patterns(group_elements)
            if graph_info:
                print(f"  LAYOUT: ✅ GRAPH pattern detected: {graph_info.get('pattern', 'unknown')}")
                return LayoutType.GRAPH, 0, graph_info
        else:
            print(f"  LAYOUT: Skipping GRAPH pattern detection (individual elements)")
        
        # 3. GRAPH 패턴도 아니고 겹침이 있으면 ZStack
        if has_overlaps:
            print(f"  LAYOUT: ✅ Has overlaps, returning ZStack")
            return LayoutType.ZSTACK, 0, None
        
        # 4. 마지막으로 Group
        print(f"  LAYOUT: ✅ Irregular layout, returning Group")
        return LayoutType.GROUP, 0, None

def detect_graph_patterns(group_elements: List[LayoutElement]) -> Optional[Dict[str, Any]]:
    """6가지 그래프 패턴을 감지하고 적절한 edge 정보 반환"""
    if len(group_elements) < 2:
        return None
    
    print(f"  GRAPH: Starting pattern detection with {len(group_elements)} elements")
    
    # 요소들의 중심점 계산
    nodes = []
    for i, elem in enumerate(group_elements):
        center_x = elem.x + elem.w / 2
        center_y = elem.y + elem.h / 2
        nodes.append({
            'id': elem.tbpe_id,
            'index': i,
            'center_x': center_x,
            'center_y': center_y,
            'width': elem.w,
            'height': elem.h,
            'elem': elem
        })
        print(f"    Node {i}: {elem.tbpe_id} at center ({center_x}, {center_y})")
    
    # 1. 지그재그 (명확한 기하학적 특성) - 우선순위 높임
    print(f"  GRAPH: Trying zigzag pattern...")
    zigzag_info = detect_zigzag_pattern(nodes)
    if zigzag_info:
        print(f"  GRAPH: ✅ ZIGZAG pattern detected!")
        return zigzag_info
    else:
        print(f"  GRAPH: ❌ ZIGZAG pattern failed")
    
    # 2. 다각형 (순환형) - 지그재그 이후 검사
    print(f"  GRAPH: Trying polygon pattern...")
    polygon_info = detect_polygon_pattern(nodes)
    if polygon_info:
        print(f"  GRAPH: ✅ POLYGON pattern detected!")
        return polygon_info
    else:
        print(f"  GRAPH: ❌ POLYGON pattern failed")
    
    # 3. 둘러 쌓는 형태 (중심-방사형)
    print(f"  GRAPH: Trying surrounding pattern...")
    surrounding_info = detect_surrounding_pattern(nodes)
    if surrounding_info:
        print(f"  GRAPH: ✅ SURROUNDING pattern detected!")
        return surrounding_info
    else:
        print(f"  GRAPH: ❌ SURROUNDING pattern failed")

    
    # 4. 조직도 (계층형 분기) - 이후 시도
    print(f"  GRAPH: Trying hierarchy pattern...")
    hierarchy_info = detect_hierarchy_pattern(nodes)
    if hierarchy_info:
        print(f"  GRAPH: ✅ HIERARCHY pattern detected!")
        return hierarchy_info
    else:
        print(f"  GRAPH: ❌ HIERARCHY pattern failed")
    
    # 5. 워크플로우 (순차 연결) - 먼저 시도
    print(f"  GRAPH: Trying workflow pattern...")
    workflow_info = detect_workflow_pattern(nodes)
    if workflow_info:
        print(f"  GRAPH: ✅ WORKFLOW pattern detected!")
        return workflow_info
    else:
        print(f"  GRAPH: ❌ WORKFLOW pattern failed")
    
    # 6. 흐름 (대각선)
    print(f"  GRAPH: Trying flow pattern...")
    flow_info = detect_flow_pattern(nodes)
    if flow_info:
        print(f"  GRAPH: ✅ FLOW pattern detected!")
        return flow_info
    else:
        print(f"  GRAPH: ❌ FLOW pattern failed")
    
    print(f"  GRAPH: 🚫 No patterns detected!")
    return None

def detect_surrounding_pattern(nodes: List[Dict]) -> Optional[Dict[str, Any]]:
    """중심-방사형 패턴 감지"""
    if len(nodes) < 3:
        return None
    
    print(f"  SURROUNDING: Checking {len(nodes)} nodes...")
    
    # 중심점 후보들을 찾기 (다른 요소들로부터의 평균 거리가 가장 가까운 요소)
    center_candidate = None
    min_average_distance = float('inf')
    
    for potential_center in nodes:
        distances = []
        for other in nodes:
            if other['id'] != potential_center['id']:
                dist = math.sqrt(
                    (potential_center['center_x'] - other['center_x'])**2 + 
                    (potential_center['center_y'] - other['center_y'])**2
                )
                distances.append(dist)
        
        if distances:
            average_distance = sum(distances) / len(distances)
            if average_distance < min_average_distance:
                min_average_distance = average_distance
                center_candidate = potential_center
    
    if not center_candidate:
        print(f"  SURROUNDING: No center candidate found")
        return None
    
    print(f"  SURROUNDING: Center candidate: {center_candidate['id']} at ({center_candidate['center_x']}, {center_candidate['center_y']})")
    
    # 둘러싸는 노드들
    surrounding_nodes = [n for n in nodes if n['id'] != center_candidate['id']]
    if len(surrounding_nodes) < 2:
        print(f"  SURROUNDING: Not enough surrounding nodes ({len(surrounding_nodes)} < 2)")
        return None
    
    # 1. 중심에서 둘러싸는 노드들까지의 거리 유사성 검사
    distances_from_center = []
    for node in surrounding_nodes:
        dist = math.sqrt(
            (center_candidate['center_x'] - node['center_x'])**2 + 
            (center_candidate['center_y'] - node['center_y'])**2
        )
        distances_from_center.append(dist)
    
    avg_distance = sum(distances_from_center) / len(distances_from_center)
    distance_variance = sum((d - avg_distance)**2 for d in distances_from_center) / len(distances_from_center)
    distance_std = math.sqrt(distance_variance)
    
    # 거리 편차가 평균의 10% 이내인지 확인
    is_similar_distance = distance_std < avg_distance * 0.1
    print(f"  SURROUNDING: Distance similarity - avg={avg_distance:.1f}, std={distance_std:.1f}, similar={is_similar_distance}")
    
    # 2. 360도 각도 분포 검사
    angles = []
    for node in surrounding_nodes:
        angle = math.atan2(
            node['center_y'] - center_candidate['center_y'],
            node['center_x'] - center_candidate['center_x']
        )
        # 각도를 0~2π 범위로 정규화
        if angle < 0:
            angle += 2 * math.pi
        angles.append(angle)
    
    # 각도를 정렬
    angles.sort()
    
    # 각도 간격 계산 (surrounding pattern은 n-1개의 연속 간격만 계산)
    angle_gaps = []
    for i in range(len(angles) - 1):  # n-1개의 연속 간격만 계산
        current_angle = angles[i]
        next_angle = angles[i + 1]
        gap = next_angle - current_angle
        if gap < 0:
            gap += 2 * math.pi
        angle_gaps.append(gap)
    
    print(f"  SURROUNDING: Calculated {len(angle_gaps)} angle gaps for {len(angles)} nodes")
    
    # 각도 간격의 균일성 확인
    # expected_angle_gap = 2 * math.pi / len(surrounding_nodes)
    # avg_angle_gap = sum(angle_gaps) / len(angle_gaps)
    # angle_gap_variance = sum((gap - avg_angle_gap)**2 for gap in angle_gaps) / len(angle_gaps)
    # angle_gap_std = math.sqrt(angle_gap_variance)
    
    # 각도 간격이 예상값의 40% 이내로 균일한지 확인 (더 관대하게)
    # angle_uniformity = angle_gap_std / expected_angle_gap if expected_angle_gap > 0 else float('inf')
    # is_uniform_angles = angle_uniformity < 0.4
    
    # print(f"  SURROUNDING: Angle uniformity - expected_gap={math.degrees(expected_angle_gap):.1f}°, uniformity={angle_uniformity:.3f}, uniform={is_uniform_angles}")
    
    # 크기 유사성 검사 (기존 코드 유지하되 더 관대하게)
    areas = [n['width'] * n['height'] for n in surrounding_nodes]
    avg_area = sum(areas) / len(areas)
    size_variance = sum((area - avg_area)**2 for area in areas) / len(areas)
    
    # 크기 차이가 50% 이상 차이나면 제외 
    is_similar_size = size_variance <= avg_area * 0.5
    print(f"  SURROUNDING: Size similarity - avg_area={avg_area:.0f}, variance={size_variance:.0f}, similar={is_similar_size}")
    
    # 구성 유사성 검사 - surrounding node들이 같은 구성을 가지는지 확인
    surrounding_compositions = []
    for node in surrounding_nodes:
        elem = node['elem']
        if hasattr(elem, 'element_tags'):
            # GroupMetadata인 경우 - element_tags를 정렬하여 비교
            composition = tuple(sorted(elem.element_tags))
        elif hasattr(elem, 'tag'):
            # ElementMetadata인 경우 - tag를 사용
            composition = (elem.tag,)
        else:
            # 기타 경우 - tbpe_id에서 tag 추출
            tag = elem.tbpe_id.split('_')[0] if '_' in elem.tbpe_id else elem.tbpe_id
            composition = (tag,)
        surrounding_compositions.append(composition)
    
    # 모든 surrounding node들이 같은 구성을 가지는지 확인
    unique_compositions = set(surrounding_compositions)
    is_similar_composition = len(unique_compositions) == 1
    
    print(f"  SURROUNDING: Composition similarity - compositions={surrounding_compositions}, similar={is_similar_composition}")
    
    # 중심 노드가 주변 노드들과 구별되는 특성을 가지는지 검사
    center_area = center_candidate['width'] * center_candidate['height']
    center_is_different_size = abs(center_area - avg_area) > avg_area * 0.1  # 중심 노드가 평균보다 10% 이상 다름
    
    # 중심 노드의 크기 비율 (가로세로비) 검사  
    center_aspect_ratio = center_candidate['width'] / center_candidate['height'] if center_candidate['height'] > 0 else 1
    surrounding_aspect_ratios = [n['width'] / n['height'] if n['height'] > 0 else 1 for n in surrounding_nodes]
    avg_surrounding_aspect = sum(surrounding_aspect_ratios) / len(surrounding_aspect_ratios)
    center_has_different_aspect = abs(center_aspect_ratio - avg_surrounding_aspect) > 0.1  # 가로세로비가 0.1 이상 차이
    
    # ID prefix 검사 - 중심 노드와 주변 노드들의 ID prefix가 다른지 확인
    def extract_id_prefix(node_id):
        # 숫자가 나오기 전까지의 문자열을 prefix로 추출
        import re
        match = re.match(r'^([^\d]*)', str(node_id).strip())
        return match.group(1).strip() if match else ""
    
    center_prefix = extract_id_prefix(center_candidate.get('id', ''))
    surrounding_prefixes = [extract_id_prefix(node.get('id', '')) for node in surrounding_nodes]
    center_has_different_id_prefix = center_prefix and all(center_prefix != prefix for prefix in surrounding_prefixes if prefix)
    
    # 중심 노드가 구별되는 특성을 가지는지 종합 판단
    center_is_distinctive = center_is_different_size or center_has_different_aspect or center_has_different_id_prefix
    
    print(f"  SURROUNDING: Center distinctiveness check:")
    print(f"    - Center area: {center_area:.0f} vs avg surrounding: {avg_area:.0f}")
    print(f"    - Center different size: {center_is_different_size}")
    print(f"    - Center aspect ratio: {center_aspect_ratio:.2f} vs avg surrounding: {avg_surrounding_aspect:.2f}")
    print(f"    - Center different aspect: {center_has_different_aspect}")
    print(f"    - Center ID prefix: '{center_prefix}' vs surrounding: {surrounding_prefixes}")
    print(f"    - Center different ID prefix: {center_has_different_id_prefix}")
    print(f"    - Center is distinctive: {center_is_distinctive}")
    
    # 각도 커버리지 검사 - 주변 노드들이 적절히 분산되어 있는지 확인
    # 전체 360도 중 실제 분포된 각도 범위 계산
    if len(angles) >= 3:
        angle_range = max(angles) - min(angles)
        # 만약 가장 큰 각도와 가장 작은 각도 사이의 간격이 더 작다면 (원형 분포)
        circular_gap = (2 * math.pi) - angle_range
        actual_coverage = max(angle_range, circular_gap)
        angle_coverage_ratio = actual_coverage / (2 * math.pi)
        has_good_angle_coverage = angle_coverage_ratio >= 0.5  # 최소 180도 이상 분포
    else:
        angle_coverage_ratio = 1.0  # 2개 노드는 항상 최대 커버리지로 간주
        has_good_angle_coverage = True  # 2개 노드는 항상 충분한 분포로 간주
    
    print(f"  SURROUNDING: Angle coverage ratio: {angle_coverage_ratio:.3f}, good coverage: {has_good_angle_coverage}")
    
    # 최종 판단: (거리 유사성 OR 크기 유사성 OR 구성 유사성) AND (적절한 각도 분포 OR 중심 노드 구별성)
    # 거리, 크기, 구성 중 하나라도 유사하면서 (각도 분포가 좋거나 중심이 구별되면) surrounding 패턴으로 인정
    is_surrounding = (is_similar_distance or is_similar_composition) and (is_similar_size or is_similar_composition) and has_good_angle_coverage  and center_is_distinctive
    
    print(f"  SURROUNDING: Final decision: {is_surrounding}")
    print(f"    - Similar distance: {is_similar_distance}")
    # print(f"    - Uniform angles: {is_uniform_angles}")
    print(f"    - Similar size: {is_similar_size}")
    print(f"    - Similar composition: {is_similar_composition}")
    print(f"    - Good angle coverage: {has_good_angle_coverage}")
    print(f"    - Center is distinctive: {center_is_distinctive}")
    print(f"    - Decision logic: (distance OR composition) and (size OR composition) AND (coverage OR distinctive)")
    
    if not is_surrounding:
        return None
    
    print(f"  SURROUNDING: ✅ Pattern detected!")
    
    # 중심에서 방사형으로 edge 연결
    edges = []
    for node in surrounding_nodes:
        edges.append({
            'from': center_candidate['id'],
            'to': node['id']
        })
        print(f"    Edge: {center_candidate['id']} → {node['id']}")
    
    return {
        'pattern': 'surrounding',
        'center_node': center_candidate['id'],
        'nodes': [{'id': n['id'], 'x': n['center_x'], 'y': n['center_y']} for n in nodes],
        'edges': edges
    }

def detect_hierarchy_pattern(nodes: List[Dict]) -> Optional[Dict[str, Any]]:
    """조직도 (계층형) 패턴 감지 - 분기 구조 기준"""
    print(f"  HIERARCHY: Checking {len(nodes)} nodes...")
    if len(nodes) < 3:  # 최소 3개 (부모-자식 분기를 위해)
        print(f"  HIERARCHY: Too few nodes ({len(nodes)} < 3)")
        return None
    
    # y좌표로 레벨 분류
    y_levels = {}
    tolerance = 40
    
    for node in nodes:
        placed = False
        for level_y in y_levels.keys():
            if abs(node['center_y'] - level_y) <= tolerance:
                y_levels[level_y].append(node)
                placed = True
                break
        if not placed:
            y_levels[node['center_y']] = [node]
    
    num_levels = len(y_levels)
    print(f"  HIERARCHY: Found {num_levels} y-levels")
    
    # 최소 2개 레벨 필요
    if num_levels < 2:
        print(f"  HIERARCHY: Not enough levels")
        return None
    
    # 레벨을 y좌표 순으로 정렬
    sorted_levels = sorted(y_levels.items())
    
    print(f"  HIERARCHY: Level structure:")
    for i, (y_pos, level_nodes) in enumerate(sorted_levels):
        print(f"    Level {i+1}: {len(level_nodes)} nodes at y={y_pos:.1f}")
    
    # 분기 구조 확인 - 핵심 로직
    level_sizes = [len(level[1]) for level in sorted_levels]
    
    # 1. 일대다 분기 구조 (1→N)
    has_one_to_many = any(
        level_sizes[i] == 1 and level_sizes[i+1] >= 2 
        for i in range(len(level_sizes)-1)
    )
    
    # 2. 다대일 수렴 구조 (N→1)  
    has_many_to_one = any(
        level_sizes[i] >= 2 and level_sizes[i+1] == 1
        for i in range(len(level_sizes)-1)
    )
    
    # 3. 피라미드 구조 (점진적 확장/수축)
    is_expanding = all(level_sizes[i] <= level_sizes[i+1] for i in range(len(level_sizes)-1))
    is_contracting = all(level_sizes[i] >= level_sizes[i+1] for i in range(len(level_sizes)-1))
    
    print(f"  HIERARCHY: Branch structure analysis:")
    print(f"    One-to-many branch: {has_one_to_many}")
    print(f"    Many-to-one convergence: {has_many_to_one}")
    print(f"    Expanding pyramid: {is_expanding}")
    print(f"    Contracting pyramid: {is_contracting}")
    
    # 분기 구조가 있어야 hierarchy로 인정
    has_branching = has_one_to_many or has_many_to_one or is_expanding or is_contracting
    
    if not has_branching:
        print(f"  HIERARCHY: No clear branching structure - might be sequential flow")
        return None
    
    # 순차 연결 구조인지 추가 확인 (workflow와 구분)
    if num_levels == len(nodes):  # 모든 노드가 다른 레벨 = 순차적
        print(f"  HIERARCHY: All nodes on different levels - sequential, not hierarchical")
        return None
    
    # 원형 배치인지 확인 (polygon과 구분)
    print(f"  HIERARCHY: Checking for circular arrangement (polygon conflict)...")
    
    # 중심점 계산
    center_x = sum(n['center_x'] for n in nodes) / len(nodes)
    center_y = sum(n['center_y'] for n in nodes) / len(nodes)
    
    # 각 노드의 중심으로부터의 각도 계산
    angles = []
    for node in nodes:
        angle = math.atan2(node['center_y'] - center_y, node['center_x'] - center_x)
        if angle < 0:
            angle += 2 * math.pi
        angles.append(angle)
    
    # 각도 정렬
    angles.sort()
    
    # 각도 간격의 균일성 확인
    if len(angles) >= 3:
        angle_gaps = []
        for i in range(len(angles)):
            current_angle = angles[i]
            next_angle = angles[(i + 1) % len(angles)]
            gap = next_angle - current_angle
            if gap < 0:
                gap += 2 * math.pi
            angle_gaps.append(gap)
        
        avg_angle_gap = sum(angle_gaps) / len(angle_gaps)
        angle_variance = sum((gap - avg_angle_gap)**2 for gap in angle_gaps) / len(angle_gaps)
        angle_std = math.sqrt(angle_variance)
        angle_uniformity = angle_std / avg_angle_gap if avg_angle_gap > 0 else float('inf')
        
        # 각도가 너무 균일하면 polygon일 가능성 높음 (16.7%도 부정확하므로 15% 기준 사용)
        is_circular_arrangement = angle_uniformity < 0.15
        
        print(f"    Circular check - angle uniformity: {angle_uniformity:.3f}, circular: {is_circular_arrangement}")
        
        if is_circular_arrangement:
            print(f"  HIERARCHY: Detected circular arrangement - likely polygon, not hierarchy")
            return None
    
    # 추가 조건 체크: composition과 ID prefix 유사성 검사
    print(f"  HIERARCHY: Checking additional conditions (composition & ID similarity)...")
    
    # 1. Composition 유사성 검사
    def extract_composition(node):
        """노드에서 composition 정보 추출"""
        node_data = node.get('elem', node)
        if hasattr(node_data, 'element_tags'):
            # GroupMetadata인 경우
            return tuple(sorted(node_data.element_tags))
        elif hasattr(node_data, 'tag'):
            # ElementMetadata인 경우
            return (node_data.tag,)
        else:
            # 기타 경우 - tbpe_id에서 tag 추출
            tbpe_id = node_data.get('tbpe_id', '') if isinstance(node_data, dict) else getattr(node_data, 'tbpe_id', '')
            tag = tbpe_id.split('_')[0] if '_' in tbpe_id else tbpe_id
            return (tag,)
    
    # 2. ID prefix 유사성 검사
    def extract_id_prefix(node_id):
        """ID에서 prefix 추출 (숫자가 나오기 전까지)"""
        import re
        match = re.match(r'^([^\d]*)', str(node_id).strip())
        return match.group(1).strip() if match else ""
    
    # 최상단 레벨(level 0)과 그 자식들(level 1) 간의 composition과 ID prefix 유사성 검사
    composition_similarity_high = False
    id_prefix_similarity_high = False
    
    if len(sorted_levels) >= 2:  # 최소 2개 레벨이 있어야 검사 가능
        parent_level = sorted_levels[0][1]  # 최상단 레벨
        child_level = sorted_levels[1][1]   # 그 다음 레벨
        
        # Parent level의 composition들
        parent_compositions = set()
        parent_prefixes = set()
        for parent_node in parent_level:
            parent_compositions.add(extract_composition(parent_node))
            parent_prefixes.add(extract_id_prefix(parent_node.get('id', '')))
        
        # Child level의 composition들
        child_compositions = set()
        child_prefixes = set()
        for child_node in child_level:
            child_compositions.add(extract_composition(child_node))
            child_prefixes.add(extract_id_prefix(child_node.get('id', '')))
        
        # Composition 유사성 계산
        common_compositions = parent_compositions.intersection(child_compositions)
        composition_overlap_ratio = len(common_compositions) / max(len(parent_compositions), len(child_compositions))
        
        # ID prefix 유사성 계산  
        common_prefixes = parent_prefixes.intersection(child_prefixes)
        prefix_overlap_ratio = len(common_prefixes) / max(len(parent_prefixes), len(child_prefixes))
        
        # 크기와 aspect ratio 유사성 계산
        def extract_size_info(node):
            """노드에서 크기 정보 추출"""
            node_data = node.get('elem', node)
            if isinstance(node_data, dict):
                w = node_data.get('width', node_data.get('w', 0))
                h = node_data.get('height', node_data.get('h', 0))
            else:
                w = getattr(node_data, 'w', getattr(node_data, 'width', 0))
                h = getattr(node_data, 'h', getattr(node_data, 'height', 0))
            
            # aspect ratio 계산 (0으로 나누기 방지)
            aspect_ratio = w / h if h > 0 else 1.0
            return w, h, aspect_ratio
        
        # Parent level의 크기 정보들
        parent_sizes = []
        parent_aspects = []
        for parent_node in parent_level:
            w, h, aspect = extract_size_info(parent_node)
            parent_sizes.append((w, h))
            parent_aspects.append(aspect)
        
        # Child level의 크기 정보들
        child_sizes = []
        child_aspects = []
        for child_node in child_level:
            w, h, aspect = extract_size_info(child_node)
            child_sizes.append((w, h))
            child_aspects.append(aspect)
        
        # 크기 유사성 검사 (30% 이내 차이면 유사한 것으로 판단)
        def are_sizes_similar(sizes1, sizes2, tolerance=0.3):
            """두 크기 집합이 유사한지 확인"""
            if not sizes1 or not sizes2:
                return False
            
            avg_w1 = sum(w for w, h in sizes1) / len(sizes1)
            avg_h1 = sum(h for w, h in sizes1) / len(sizes1)
            avg_w2 = sum(w for w, h in sizes2) / len(sizes2)
            avg_h2 = sum(h for w, h in sizes2) / len(sizes2)
            
            w_diff = abs(avg_w1 - avg_w2) / max(avg_w1, avg_w2) if max(avg_w1, avg_w2) > 0 else 0
            h_diff = abs(avg_h1 - avg_h2) / max(avg_h1, avg_h2) if max(avg_h1, avg_h2) > 0 else 0
            
            return w_diff <= tolerance and h_diff <= tolerance
        
        # Aspect ratio 유사성 검사 (20% 이내 차이면 유사한 것으로 판단)
        def are_aspects_similar(aspects1, aspects2, tolerance=0.2):
            """두 aspect ratio 집합이 유사한지 확인"""
            if not aspects1 or not aspects2:
                return False
            
            avg_aspect1 = sum(aspects1) / len(aspects1)
            avg_aspect2 = sum(aspects2) / len(aspects2)
            
            aspect_diff = abs(avg_aspect1 - avg_aspect2) / max(avg_aspect1, avg_aspect2) if max(avg_aspect1, avg_aspect2) > 0 else 0
            
            return aspect_diff <= tolerance
        
        size_similarity = are_sizes_similar(parent_sizes, child_sizes)
        aspect_similarity = are_aspects_similar(parent_aspects, child_aspects)
        
        print(f"    Top Level (1 → 2):")
        print(f"      Parent compositions: {parent_compositions}")
        print(f"      Child compositions: {child_compositions}")
        print(f"      Composition overlap: {composition_overlap_ratio:.2f}")
        print(f"      Parent prefixes: {parent_prefixes}")
        print(f"      Child prefixes: {child_prefixes}")
        print(f"      Prefix overlap: {prefix_overlap_ratio:.2f}")
        print(f"      Parent sizes: {parent_sizes}")
        print(f"      Child sizes: {child_sizes}")
        print(f"      Size similarity: {size_similarity}")
        print(f"      Parent aspects: {[f'{a:.2f}' for a in parent_aspects]}")
        print(f"      Child aspects: {[f'{a:.2f}' for a in child_aspects]}")
        print(f"      Aspect similarity: {aspect_similarity}")
        
        # 높은 유사성 검출 및 크기/비율 고려
        composition_high = composition_overlap_ratio >= 0.7
        prefix_high = prefix_overlap_ratio >= 0.7
        
        # 최종 판단: (composition 또는 prefix가 유사) AND (크기와 비율도 유사)면 hierarchy 제외
        if (composition_high or prefix_high) and (size_similarity and aspect_similarity):
            composition_similarity_high = True
            id_prefix_similarity_high = True
            print(f"      → Similar content AND similar size/aspect → Not hierarchy")
        elif composition_high or prefix_high:
            print(f"      → Similar content BUT different size/aspect → Still hierarchy")
        else:
            print(f"      → Different content → Hierarchy allowed")
    
    print(f"  HIERARCHY: Composition similarity high: {composition_similarity_high}")
    print(f"  HIERARCHY: ID prefix similarity high: {id_prefix_similarity_high}")
    
    # 조건 체크: composition이나 ID prefix가 너무 유사하면 hierarchy가 아님
    if composition_similarity_high or id_prefix_similarity_high:
        print(f"  HIERARCHY: ❌ Too similar composition/ID - likely workflow, not hierarchy")
        return None
    
    print(f"  HIERARCHY: ✅ Valid branching hierarchy detected")
    
    # 분기 구조 기반 연결 (레벨별 분기/수렴)
    edges = []
    print(f"  HIERARCHY: Creating branching connections:")
    
    for i in range(len(sorted_levels) - 1):
        current_level = sorted_levels[i][1]  # 상위 레벨
        next_level = sorted_levels[i+1][1]   # 하위 레벨
        
        current_size = len(current_level)
        next_size = len(next_level)
        
        print(f"    Level {i+1} → Level {i+2}: {current_size} → {next_size}")
        
        if current_size == 1 and next_size > 1:
            # 일대다 분기 (1→N)
            parent = current_level[0]
            for child in next_level:
                edges.append({
                    'from': parent['id'],
                    'to': child['id']
                })
                print(f"      Branch: {parent['id']} → {child['id']}")
                
        elif current_size > 1 and next_size == 1:
            # 다대일 수렴 (N→1)
            child = next_level[0]
            for parent in current_level:
                edges.append({
                    'from': parent['id'],
                    'to': child['id']
                })
                print(f"      Converge: {parent['id']} → {child['id']}")
                
        elif current_size <= next_size:
            # 확장형 (각 부모가 여러 자식에게)
            current_sorted = sorted(current_level, key=lambda n: n['center_x'])
            next_sorted = sorted(next_level, key=lambda n: n['center_x'])
            
            # 각 부모가 담당할 자식 수 계산
            children_per_parent = next_size // current_size
            extra_children = next_size % current_size
            
            child_index = 0
            for j, parent in enumerate(current_sorted):
                # 기본 자식 수 + 추가 자식 (남은 것 분배)
                num_children = children_per_parent + (1 if j < extra_children else 0)
                
                for k in range(num_children):
                    if child_index < len(next_sorted):
                        child = next_sorted[child_index]
                        edges.append({
                            'from': parent['id'],
                            'to': child['id']
                        })
                        print(f"      Expand: {parent['id']} → {child['id']}")
                        child_index += 1
                        
        else:
            # 수축형 (여러 부모가 각자 자식에게)
            current_sorted = sorted(current_level, key=lambda n: n['center_x'])
            next_sorted = sorted(next_level, key=lambda n: n['center_x'])
            
            # 각 자식이 받을 부모 수 계산
            parents_per_child = current_size // next_size
            extra_parents = current_size % next_size
            
            parent_index = 0
            for j, child in enumerate(next_sorted):
                # 기본 부모 수 + 추가 부모 (남은 것 분배)
                num_parents = parents_per_child + (1 if j < extra_parents else 0)
                
                for k in range(num_parents):
                    if parent_index < len(current_sorted):
                        parent = current_sorted[parent_index]
                        edges.append({
                            'from': parent['id'],
                            'to': child['id']
                        })
                        print(f"      Contract: {parent['id']} → {child['id']}")
                        parent_index += 1
    
    print(f"  HIERARCHY: ✅ Created {len(edges)} branching connections")
    
    return {
        'pattern': 'hierarchy',
        'levels': len(sorted_levels),
        'branch_type': 'expanding' if is_expanding else 'contracting' if is_contracting else 'mixed',
        'nodes': [{'id': n['id'], 'x': n['center_x'], 'y': n['center_y']} for n in nodes],
        'edges': edges
    }

def detect_zigzag_pattern(nodes: List[Dict]) -> Optional[Dict[str, Any]]:
    """지그재그 패턴 감지 - 기하학적 특성 기반"""
    print(f"  ZIGZAG: Checking {len(nodes)} nodes...")
    if len(nodes) < 4:
        print(f"  ZIGZAG: Too few nodes ({len(nodes)} < 4)")
        return None
    
    # 1. y좌표가 정확히 2개 레벨인지 확인
    y_tolerance = 30
    y_levels = {}
    for node in nodes:
        placed = False
        for y_center in y_levels.keys():
            if abs(node['center_y'] - y_center) <= y_tolerance:
                y_levels[y_center].append(node)
                placed = True
                break
        if not placed:
            y_levels[node['center_y']] = [node]
    
    num_y_levels = len(y_levels)
    print(f"  ZIGZAG: Found {num_y_levels} y-levels")
    
    # 지그재그는 정확히 2개 y레벨이어야 함
    if num_y_levels != 2:
        print(f"  ZIGZAG: Not exactly 2 y-levels, cannot be zigzag")
        return None
    
    # 2. 각 레벨에 최소 2개 이상의 노드가 있어야 함
    level_sizes = [len(level) for level in y_levels.values()]
    if any(size < 2 for size in level_sizes):
        print(f"  ZIGZAG: Level sizes {level_sizes}, need at least 2 nodes per level")
        return None
    
    # 3. x좌표 순으로 정렬하여 지그재그 패턴 확인
    sorted_nodes = sorted(nodes, key=lambda n: n['center_x'])
    y_positions = [n['center_y'] for n in sorted_nodes]
    
    print(f"  ZIGZAG: Checking x-sorted nodes:")
    for i, node in enumerate(sorted_nodes):
        print(f"    {i}: {node['id']} at ({node['center_x']}, {node['center_y']})")
    
    # 4. x값의 차이가 일정한지 확인
    x_positions = [n['center_x'] for n in sorted_nodes]
    x_gaps = []
    for i in range(1, len(x_positions)):
        gap = x_positions[i] - x_positions[i-1]
        x_gaps.append(gap)
    
    if x_gaps:
        avg_x_gap = sum(x_gaps) / len(x_gaps)
        x_gap_variance = sum((gap - avg_x_gap)**2 for gap in x_gaps) / len(x_gaps)
        x_gap_std = math.sqrt(x_gap_variance)
        is_regular_spacing = x_gap_std < avg_x_gap * 0.3  # 30% 이내 편차
        
        print(f"  ZIGZAG: X-spacing - avg={avg_x_gap:.1f}, std={x_gap_std:.1f}, regular={is_regular_spacing}")
        
        if not is_regular_spacing:
            print(f"  ZIGZAG: X-spacing not regular enough")
            return None
    
    # 5. y좌표 지그재그 전환 확인
    zigzag_count = 0
    for i in range(2, len(y_positions)):
        prev_diff = y_positions[i-1] - y_positions[i-2]
        curr_diff = y_positions[i] - y_positions[i-1]
        
        # 방향이 바뀌면 지그재그
        if prev_diff * curr_diff < 0 and abs(prev_diff) > 20 and abs(curr_diff) > 20:
            zigzag_count += 1
            print(f"    Zigzag transition at position {i}: prev_diff={prev_diff}, curr_diff={curr_diff}")
    
    # 지그재그 전환점이 충분한지 확인 (전체 구간의 70% 이상)
    max_possible_transitions = max(1, len(nodes) - 2)
    min_required_transitions = max(1, int(max_possible_transitions * 0.7))
    
    print(f"  ZIGZAG: Transitions {zigzag_count}/{max_possible_transitions}, required: {min_required_transitions}")
    
    if zigzag_count < min_required_transitions:
        print(f"  ZIGZAG: Not enough transitions")
        return None
    
    # 6. 최종 검증: 2개 y레벨 + 규칙적 x간격 + 충분한 전환
    print(f"  ZIGZAG: All criteria met - 2 y-levels, regular x-spacing, sufficient transitions")
    
    # 왼쪽부터 오른쪽으로 순차 연결 (x좌표 기준)
    edges = []
    print(f"  ZIGZAG: Creating edges in x-coordinate order:")
    for i in range(len(sorted_nodes) - 1):
        from_node = sorted_nodes[i]
        to_node = sorted_nodes[i+1]
        edge = {
            'from': from_node['id'],
            'to': to_node['id']
        }
        edges.append(edge)
        print(f"    Edge {i+1}: {from_node['id']} (x={from_node['center_x']}) → {to_node['id']} (x={to_node['center_x']})")
    
    print(f"  ZIGZAG: ✅ Pattern confirmed with {len(edges)} edges")
    return {
        'pattern': 'zigzag',
        'nodes': [{'id': n['id'], 'x': n['center_x'], 'y': n['center_y']} for n in nodes],
        'edges': edges
    }

def detect_flow_pattern(nodes: List[Dict]) -> Optional[Dict[str, Any]]:
    """흐름 (대각선) 패턴 감지"""
    if len(nodes) < 3:
        return None
    
    # 좌상단에서 우하단으로의 대각선 흐름 확인
    # x, y 좌표 모두 증가하는 패턴
    sorted_nodes = sorted(nodes, key=lambda n: n['center_x'])
    
    # x가 증가할 때 y도 대체로 증가하는지 확인
    x_positions = [n['center_x'] for n in sorted_nodes]
    y_positions = [n['center_y'] for n in sorted_nodes]
    
    # 피어슨 상관계수 계산
    n = len(nodes)
    sum_x = sum(x_positions)
    sum_y = sum(y_positions)
    sum_xy = sum(x_positions[i] * y_positions[i] for i in range(n))
    sum_x2 = sum(x * x for x in x_positions)
    sum_y2 = sum(y * y for y in y_positions)
    
    # 분모가 0이 되는 경우 처리
    denominator = (n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2)
    if denominator <= 0:
        return None
    
    correlation = (n * sum_xy - sum_x * sum_y) / math.sqrt(denominator)
    
    # 강한 양의 상관관계 (우하향) 또는 강한 음의 상관관계 (우상향)
    if abs(correlation) < 0.6:
        return None
    
    # 왼쪽부터 오른쪽으로 순차 연결
    edges = []
    for i in range(len(sorted_nodes) - 1):
        edges.append({
            'from': sorted_nodes[i]['id'],
            'to': sorted_nodes[i+1]['id']
        })
    
    return {
        'pattern': 'flow',
        'direction': 'downward' if correlation > 0 else 'upward',
        'nodes': [{'id': n['id'], 'x': n['center_x'], 'y': n['center_y']} for n in nodes],
        'edges': edges
    }

def detect_polygon_pattern(nodes: List[Dict]) -> Optional[Dict[str, Any]]:
    """다각형 (순환형) 패턴 감지 - 360°/n 각도 기준"""
    print(f"  POLYGON: Checking {len(nodes)} nodes...")
    if len(nodes) < 3:
        print(f"  POLYGON: Too few nodes ({len(nodes)} < 3)")
        return None
    
    # 중심점 계산
    center_x = sum(n['center_x'] for n in nodes) / len(nodes)
    center_y = sum(n['center_y'] for n in nodes) / len(nodes)
    print(f"  POLYGON: Center at ({center_x:.1f}, {center_y:.1f})")
    
    # 각 노드의 중심으로부터의 각도와 거리 계산
    nodes_with_angle = []
    for i, node in enumerate(nodes):
        angle = math.atan2(node['center_y'] - center_y, node['center_x'] - center_x)
        # 각도를 0~2π 범위로 정규화
        if angle < 0:
            angle += 2 * math.pi
        distance = math.sqrt((node['center_x'] - center_x)**2 + (node['center_y'] - center_y)**2)
        nodes_with_angle.append({
            'node': node,
            'angle': angle,
            'distance': distance
        })
        print(f"    Node {i}: {node['id']} at angle={math.degrees(angle):.1f}°, distance={distance:.1f}")
    
    # 시계방향으로 polygon 순서 찾기 (이웃하는 노드들을 순차적으로 연결)
    print(f"  POLYGON: Finding clockwise polygon order using neighbor traversal...")
    
    def find_clockwise_polygon_order(nodes_data):
        """시계방향으로 polygon의 노드 순서를 찾는 함수"""
        if len(nodes_data) < 3:
            return nodes_data
        
        # 1. 시작점 찾기: 가장 오른쪽 상단 점 (시계방향 시작점으로 명확)
        start_node = max(nodes_data, key=lambda n: (n['node']['center_x'], -n['node']['center_y']))
        print(f"    Start node: {start_node['node']['id']} at ({start_node['node']['center_x']}, {start_node['node']['center_y']})")
        
        ordered_nodes = [start_node]
        remaining_nodes = [n for n in nodes_data if n['node']['id'] != start_node['node']['id']]
        
        current_node = start_node
        
        while remaining_nodes:
            best_next = None
            best_angle_diff = float('inf')
            
            current_angle = current_node['angle']  # polygon 중점을 기준으로 한 현재 노드의 각도
            
            for candidate in remaining_nodes:
                candidate_angle = candidate['angle']  # polygon 중점을 기준으로 한 후보 노드의 각도
                
                # 시계방향 각도 차이 계산
                angle_diff = candidate_angle - current_angle
                
                # 시계방향으로 정규화 (0 ~ 2π)
                if angle_diff < 0:
                    angle_diff += 2 * math.pi
                
                # 가장 작은 시계방향 각도 차이를 가진 노드 선택
                if angle_diff < best_angle_diff:
                    best_angle_diff = angle_diff
                    best_next = candidate
            
            if best_next is None:
                print(f"    Warning: Cannot find next node from {current_node['node']['id']}")
                break
            
            # 다음 노드로 이동
            ordered_nodes.append(best_next)
            remaining_nodes.remove(best_next)
            current_node = best_next
            
            print(f"    Next node: {best_next['node']['id']} (center angle diff: {math.degrees(best_angle_diff):.1f}°)")
        
        return ordered_nodes
    
    # 시계방향 순서로 정렬
    nodes_with_angle = find_clockwise_polygon_order(nodes_with_angle)
    
    print(f"  POLYGON: Final clockwise order:")
    for i, node_data in enumerate(nodes_with_angle):
        print(f"    {i+1}: {node_data['node']['id']} at ({node_data['node']['center_x']}, {node_data['node']['center_y']})")
    
    # 1. 360°/n 각도 분포 확인
    n = len(nodes_with_angle)
    expected_angle_gap = 2 * math.pi / n  # 360°/n in radians
    print(f"  POLYGON: Expected angle gap: {math.degrees(expected_angle_gap):.1f}°")
    
    # 실제 각도 간격 계산
    actual_angle_gaps = []
    for i in range(n):
        current_angle = nodes_with_angle[i]['angle']
        next_angle = nodes_with_angle[(i + 1) % n]['angle']
        
        # 각도 차이 계산 (순환 고려)
        gap = next_angle - current_angle
        if gap < 0:
            gap += 2 * math.pi
        actual_angle_gaps.append(gap)
        print(f"    Angle gap {i+1}: {math.degrees(gap):.1f}°")
    
    # 각도 간격의 평균 정확도 확인
    avg_angle_gap = sum(actual_angle_gaps) / len(actual_angle_gaps)
    angle_gap_variance = sum((gap - avg_angle_gap)**2 for gap in actual_angle_gaps) / len(actual_angle_gaps)
    angle_gap_std = math.sqrt(angle_gap_variance)
    
    # 평균 각도가 예상값(360°/n)과 얼마나 가까운지 확인
    angle_accuracy = abs(avg_angle_gap - expected_angle_gap) / expected_angle_gap if expected_angle_gap > 0 else float('inf')
    is_accurate_angles = angle_accuracy < 0.2  # 예상값의 20% 이내
    
    print(f"  POLYGON: Angle gap stats - avg={math.degrees(avg_angle_gap):.1f}°, expected={math.degrees(expected_angle_gap):.1f}°, std={math.degrees(angle_gap_std):.1f}°")
    print(f"  POLYGON: Angle accuracy: {angle_accuracy:.3f} (< 0.2 = {is_accurate_angles})")
    
    # 2. 거리 분산 확인 (중심에서 각 노드까지의 거리)
    distances = [n['distance'] for n in nodes_with_angle]
    avg_distance = sum(distances) / len(distances)
    distance_variance = sum((d - avg_distance)**2 for d in distances) / len(distances)
    distance_std = math.sqrt(distance_variance)
    
    # 거리 기준 (더 엄격하게)
    max_distance = max(distances)
    min_distance = min(distances)
    distance_ratio = max_distance / min_distance if min_distance > 0 else float('inf')
    is_reasonable_distance = distance_ratio <= 2.5  # 2.5:1 이내 (더 엄격)
    

    # 2.5. 크기 유사성 확인 (subgroup들의 크기가 비슷한지 체크)
    areas = [n["width"] * n["height"] for n in nodes]
    avg_area = sum(areas) / len(areas)
    area_variance = sum((area - avg_area)**2 for area in areas) / len(areas)
    area_std = math.sqrt(area_variance)
    
    # 크기 기준: 최대 크기와 최소 크기의 비율
    max_area = max(areas)
    min_area = min(areas)
    size_ratio = max_area / min_area if min_area > 0 else float("inf")
    is_similar_size = size_ratio <= 3.0  # 3:1 이내면 비슷한 크기로 판단
    
    print(f"  POLYGON: Size stats - avg={avg_area:.0f}, std={area_std:.0f}")
    print(f"  POLYGON: Size ratio: {size_ratio:.2f} (< 3.0 = {is_similar_size})")

    print(f"  POLYGON: Distance stats - avg={avg_distance:.1f}, std={distance_std:.1f}")
    print(f"  POLYGON: Distance ratio: {distance_ratio:.2f} (< 2.5 = {is_reasonable_distance})")
    
    # 3. 지그재그 패턴 강화 검사
    y_tolerance = 30
    y_levels = {}
    for node in nodes:
        placed = False
        for y_center in y_levels.keys():
            if abs(node['center_y'] - y_center) <= y_tolerance:
                y_levels[y_center].append(node)
                placed = True
                break
        if not placed:
            y_levels[node['center_y']] = [node]
    
    num_y_levels = len(y_levels)
    # 수정: 삼각형(3개 노드)이고 2개 y-level인 경우는 정상적인 polygon
    if len(nodes) == 3 and num_y_levels == 2:
        is_not_zigzag_basic = True  # 삼각형은 2개 y-level이 정상
    else:
        is_not_zigzag_basic = num_y_levels != 2  # 기타 경우: 지그재그는 정확히 2개 y레벨
    
    # 추가 지그재그 특성 검사 (2개 y레벨인 경우)
    is_not_zigzag_advanced = True
    if num_y_levels == 2 and len(nodes) >= 4:
        print(f"  POLYGON: Detected 2 y-levels, checking for zigzag characteristics...")
        
        # x좌표 순으로 정렬
        x_sorted = sorted(nodes, key=lambda n: n['center_x'])
        y_positions = [n['center_y'] for n in x_sorted]
        
        # 지그재그 전환점 계산
        zigzag_transitions = 0
        for i in range(2, len(y_positions)):
            prev_diff = y_positions[i-1] - y_positions[i-2]
            curr_diff = y_positions[i] - y_positions[i-1]
            if prev_diff * curr_diff < 0 and abs(prev_diff) > 20 and abs(curr_diff) > 20:
                zigzag_transitions += 1
        
        # x간격 규칙성 검사
        x_positions = [n['center_x'] for n in x_sorted]
        x_gaps = [x_positions[i] - x_positions[i-1] for i in range(1, len(x_positions))]
        avg_x_gap = sum(x_gaps) / len(x_gaps) if x_gaps else 0
        x_gap_variance = sum((gap - avg_x_gap)**2 for gap in x_gaps) / len(x_gaps) if x_gaps else 0
        x_gap_std = math.sqrt(x_gap_variance)
        is_regular_x_spacing = x_gap_std < avg_x_gap * 0.3 if avg_x_gap > 0 else False
        
        # 지그재그 특성이 강하면 polygon이 아님 (더 엄격한 기준)
        max_transitions = max(1, len(nodes) - 2)
        zigzag_ratio = zigzag_transitions / max_transitions
        
        has_zigzag_characteristics = (zigzag_ratio >= 0.8) and is_regular_x_spacing  # 80%로 상향
        
        print(f"    Zigzag transitions: {zigzag_transitions}/{max_transitions} (ratio: {zigzag_ratio:.2f})")
        print(f"    Regular x-spacing: {is_regular_x_spacing} (std: {x_gap_std:.1f}, avg: {avg_x_gap:.1f})")
        print(f"    Has zigzag characteristics: {has_zigzag_characteristics}")
        
        is_not_zigzag_advanced = not has_zigzag_characteristics
    
    is_not_zigzag = is_not_zigzag_basic and is_not_zigzag_advanced
    
    print(f"  POLYGON: Y-levels: {num_y_levels}")
    print(f"  POLYGON: Not zigzag (basic): {is_not_zigzag_basic}")
    print(f"  POLYGON: Not zigzag (advanced): {is_not_zigzag_advanced}")
    print(f"  POLYGON: Not zigzag (final): {is_not_zigzag}")
    
    # 4. 최종 판단: 각도 정확도 AND 합리적 거리 AND 지그재그 아님 (더 엄격)
    is_polygon = is_accurate_angles and is_reasonable_distance and is_not_zigzag and is_similar_size
    
    print(f"  POLYGON: Final decision: {is_polygon}")
    print(f"    - Accurate angles: {is_accurate_angles}")
    print(f"    - Reasonable distance: {is_reasonable_distance}")
    print(f"    - Not zigzag: {is_not_zigzag}")
    print(f"    - Similar size: {is_similar_size}")
    
    if not is_polygon:
        return None
    
    # 순환형으로 이웃하는 노드들 연결 (각도 순)
    edges = []
    print(f"  POLYGON: Creating edges in angle order:")
    for i in range(len(nodes_with_angle)):
        current = nodes_with_angle[i]
        next_node = nodes_with_angle[(i + 1) % len(nodes_with_angle)]
        
        edge = {
            'from': current['node']['id'],
            'to': next_node['node']['id']
        }
        edges.append(edge)
        print(f"    Edge {i+1}: {current['node']['id']} → {next_node['node']['id']}")
    
    shape_name = 'triangle' if len(nodes) == 3 else f'{len(nodes)}-gon'
    print(f"  POLYGON: ✅ Pattern confirmed as {shape_name} with {len(edges)} edges")
    
    return {
        'pattern': 'polygon',
        'shape': shape_name,
        'nodes': [{'id': n['id'], 'x': n['center_x'], 'y': n['center_y']} for n in nodes],
        'edges': edges
    }

def detect_workflow_pattern(nodes: List[Dict]) -> Optional[Dict[str, Any]]:
    """workflow (순차 연결) 패턴 감지 - 가로/세로 흐름"""
    print(f"  WORKFLOW: Checking {len(nodes)} nodes...")
    if len(nodes) < 3:
        print(f"  WORKFLOW: Too few nodes ({len(nodes)} < 3)")
        return None
    
    # 1. 가로 workflow 패턴 체크 (좌→우 흐름)
    print(f"  WORKFLOW: Checking horizontal flow...")
    x_sorted = sorted(nodes, key=lambda n: n['center_x'])
    
    # 가로 배치인지 확인 (y좌표 차이가 작음)
    y_variance = sum((n['center_y'] - sum(node['center_y'] for node in nodes) / len(nodes))**2 for n in nodes) / len(nodes)
    y_std = math.sqrt(y_variance)
    avg_width = sum(abs(x_sorted[i+1]['center_x'] - x_sorted[i]['center_x']) for i in range(len(x_sorted)-1)) / (len(x_sorted)-1)
    
    # 기본 조건 확인
    is_horizontal_flow_basic = y_std < avg_width * 0.3  # y축 편차가 x축 간격의 30% 이내
    
    # 2레벨 구조에서 다수 노드가 가로 배치된 경우도 horizontal workflow로 인정
    is_horizontal_flow_multilevel = False
    if not is_horizontal_flow_basic:
        # y좌표로 레벨 분류
        y_tolerance = avg_width * 0.5  # x간격의 50%를 tolerance로 사용
        y_levels = {}
        for node in nodes:
            placed = False
            for level_y in y_levels.keys():
                if abs(node['center_y'] - level_y) <= y_tolerance:
                    y_levels[level_y].append(node)
                    placed = True
                    break
            if not placed:
                y_levels[node['center_y']] = [node]
        
        # 다중 레벨이고 한 레벨에 2개 이상 노드가 있으면 horizontal workflow 후보
        if len(y_levels) >= 2:
            level_sizes = [len(level) for level in y_levels.values()]
            max_level_size = max(level_sizes)
            if max_level_size >= 2:  # 한 레벨에 2개 이상
                # 가장 큰 레벨의 y편차 확인
                largest_level = max(y_levels.values(), key=len)
                if len(largest_level) >= 2:
                    level_y_variance = sum((n['center_y'] - sum(node['center_y'] for node in largest_level) / len(largest_level))**2 for n in largest_level) / len(largest_level)
                    level_y_std = math.sqrt(level_y_variance)
                    # 해당 레벨 내에서의 y편차가 작으면 horizontal
                    is_horizontal_flow_multilevel = level_y_std < avg_width * 0.3
    
    is_horizontal_flow = is_horizontal_flow_basic or is_horizontal_flow_multilevel
    print(f"    Y-std: {y_std:.1f}, avg x-gap: {avg_width:.1f}, horizontal: {is_horizontal_flow}")
    
    if is_horizontal_flow:
        print(f"  WORKFLOW: ✅ Horizontal flow detected")
        edges = []
        for i in range(len(x_sorted) - 1):
            edges.append({
                'from': x_sorted[i]['id'],
                'to': x_sorted[i+1]['id']
            })
            print(f"    {x_sorted[i]['id']} → {x_sorted[i+1]['id']}")
        
        return {
            'pattern': 'workflow',
            'flow_type': 'horizontal',
            'nodes': [{'id': n['id'], 'x': n['center_x'], 'y': n['center_y']} for n in nodes],
            'edges': edges
        }
    
    # 2. 세로 workflow 패턴 체크 (위→아래 흐름)
    print(f"  WORKFLOW: Checking vertical flow...")
    y_sorted = sorted(nodes, key=lambda n: n['center_y'])
    
    # 세로 배치인지 확인 (x좌표 차이가 작음)
    x_variance = sum((n['center_x'] - sum(node['center_x'] for node in nodes) / len(nodes))**2 for n in nodes) / len(nodes)
    x_std = math.sqrt(x_variance)
    avg_height = sum(abs(y_sorted[i+1]['center_y'] - y_sorted[i]['center_y']) for i in range(len(y_sorted)-1)) / (len(y_sorted)-1)
    
    # 기본 조건 확인
    is_vertical_flow_basic = x_std < avg_height * 0.3  # x축 편차가 y축 간격의 30% 이내
    
    # 2레벨 구조에서 다수 노드가 세로 배치된 경우도 vertical workflow로 인정
    is_vertical_flow_multilevel = False
    if not is_vertical_flow_basic:
        # x좌표로 레벨 분류
        x_tolerance = avg_height * 0.5  # y간격의 50%를 tolerance로 사용
        x_levels = {}
        for node in nodes:
            placed = False
            for level_x in x_levels.keys():
                if abs(node['center_x'] - level_x) <= x_tolerance:
                    x_levels[level_x].append(node)
                    placed = True
                    break
            if not placed:
                x_levels[node['center_x']] = [node]
        
        # 다중 레벨이고 한 레벨에 2개 이상 노드가 있으면 vertical workflow 후보
        if len(x_levels) >= 2:
            level_sizes = [len(level) for level in x_levels.values()]
            max_level_size = max(level_sizes)
            if max_level_size >= 2:  # 한 레벨에 2개 이상
                # 가장 큰 레벨의 x편차 확인
                largest_level = max(x_levels.values(), key=len)
                if len(largest_level) >= 2:
                    level_x_variance = sum((n['center_x'] - sum(node['center_x'] for node in largest_level) / len(largest_level))**2 for n in largest_level) / len(largest_level)
                    level_x_std = math.sqrt(level_x_variance)
                    # 해당 레벨 내에서의 x편차가 작으면 vertical
                    is_vertical_flow_multilevel = level_x_std < avg_height * 0.3
    
    is_vertical_flow = is_vertical_flow_basic or is_vertical_flow_multilevel
    print(f"    X-std: {x_std:.1f}, avg y-gap: {avg_height:.1f}, vertical: {is_vertical_flow}")
    
    if is_vertical_flow:
        print(f"  WORKFLOW: ✅ Vertical flow detected")
        edges = []
        for i in range(len(y_sorted) - 1):
            edges.append({
                'from': y_sorted[i]['id'],
                'to': y_sorted[i+1]['id']
            })
            print(f"    {y_sorted[i]['id']} → {y_sorted[i+1]['id']}")
        
        return {
            'pattern': 'workflow',
            'flow_type': 'vertical',
            'nodes': [{'id': n['id'], 'x': n['center_x'], 'y': n['center_y']} for n in nodes],
            'edges': edges
        }
    
    # 3. 대각선 workflow 패턴 체크
    print(f"  WORKFLOW: Checking diagonal flow...")
    
    # x,y 상관관계 계산
    n = len(nodes)
    sum_x = sum(n['center_x'] for n in nodes)
    sum_y = sum(n['center_y'] for n in nodes)
    sum_xy = sum(nodes[i]['center_x'] * nodes[i]['center_y'] for i in range(n))
    sum_x2 = sum(n['center_x']**2 for n in nodes)
    sum_y2 = sum(n['center_y']**2 for n in nodes)
    
    denominator = (n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2)
    if denominator > 0:
        correlation = (n * sum_xy - sum_x * sum_y) / math.sqrt(denominator)
        print(f"    X-Y correlation: {correlation:.3f}")
        
        # 강한 상관관계면 대각선 flow
        if abs(correlation) > 0.7:
            print(f"  WORKFLOW: ✅ Diagonal flow detected ({'down-right' if correlation > 0 else 'up-right'})")
            
            # x좌표 순으로 정렬해서 연결
            diagonal_sorted = sorted(nodes, key=lambda n: n['center_x'])
            edges = []
            for i in range(len(diagonal_sorted) - 1):
                edges.append({
                    'from': diagonal_sorted[i]['id'],
                    'to': diagonal_sorted[i+1]['id']
                })
                print(f"    {diagonal_sorted[i]['id']} → {diagonal_sorted[i+1]['id']}")
            
            return {
                'pattern': 'workflow',
                'flow_type': 'diagonal',
                'direction': 'down-right' if correlation > 0 else 'up-right',
                'nodes': [{'id': n['id'], 'x': n['center_x'], 'y': n['center_y']} for n in nodes],
                'edges': edges
            }
    
    print(f"  WORKFLOW: ❌ No clear sequential flow detected")
    return None

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

def create_layout_element(layout_type: LayoutType, element_id: int, gap: int, children: List[BaseElement], graph_info: Optional[Dict[str, Any]] = None) -> BaseElement:
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
    elif layout_type == LayoutType.GRAPH:
        # GRAPH 타입일 때는 그래프 정보를 포함한 특별한 element 생성
        graph_element = BaseElement(
            id=element_id,
            type="Graph",
            children=children
        )
        # 그래프 정보를 별도 속성으로 저장 (element_to_dict에서 처리)
        if graph_info:
            graph_element.graph_info = graph_info
        return graph_element
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
        if not key.startswith("Parent Group") and value is None:
            if key in metadata_objects:
                element = create_element_from_metadata(metadata_objects[key], element_id_counter)
                top_level_elements.append(element)
                element_id_counter += 1
    
    # Parent Group들 처리
    parent_groups = {k: v for k, v in semantic_group.items() if k.startswith("Parent Group") and isinstance(v, dict)}
    
    if parent_groups:
        # 첫 번째 Parent Group을 처리 (기존 로직 유지)
        first_parent_group_key = list(parent_groups.keys())[0]
        parent_group = parent_groups[first_parent_group_key]
        
        # Subgroup들의 요소 수집 및 위치 정보로 행별 그룹핑
        subgroup_data = {}
        
        for subgroup_name, subgroup_content in parent_group.items():
            if (subgroup_name.startswith("Subgroup") or subgroup_name.startswith("b group")) and isinstance(subgroup_content, dict):
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
                    # 서브그룹 레이아웃 결정 (개별 요소들이므로 graph 패턴 비허용)
                    layout_type, gap, graph_info = determine_layout_type(subgroup_metadata, allow_graph_patterns=False)
                    subgroup_element = create_layout_element(layout_type, element_id_counter, gap, subgroup_items, graph_info)
                    
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
        
        # 최상위 레이아웃 결정 (top level이므로 graph 패턴 허용)
        if len(top_level_positions) > 1:
            root_layout_type, root_gap, root_graph_info = determine_layout_type(top_level_positions)
            root_element = create_layout_element(root_layout_type, 0, root_gap, all_elements, root_graph_info)
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
        
        # Graph 정보 추가
        if hasattr(element, 'graph_info') and element.graph_info:
            result["graphInfo"] = element.graph_info
        
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
    
    # Parent Group과 a group을 찾아서 처리
    for parent_group_key in semantic_group.keys():
        if (parent_group_key.startswith("Parent Group") or parent_group_key.startswith("a group")) and isinstance(semantic_group[parent_group_key], dict):
            parent_group = semantic_group[parent_group_key]
            
            # 1단계: 먼저 subgroup들의 정보를 수집
            subgroup_info = {}  # {group_id: metadata}
            
            def collect_subgroup_info(group_dict, prefix=""):
                """subgroup들의 대표 위치 정보를 미리 수집"""
                for key, value in group_dict.items():
                    if isinstance(value, dict) and (key.startswith("Subgroup") or key.startswith("b group")):
                        # subgroup의 모든 요소들 수집
                        subgroup_elements = []
                        for sub_key, sub_value in value.items():
                            if sub_value is None and sub_key in metadata_objects:
                                subgroup_elements.append(metadata_objects[sub_key])
                        
                        if subgroup_elements:
                            # 대표 위치 계산
                            x, y, w, h = calculate_group_bounds(subgroup_elements)
                            
                            # 구성하는 element들의 tag 정보 수집
                            element_tags = [elem.tag[0] if isinstance(elem.tag, list) else elem.tag for elem in subgroup_elements]
                            
                            representative_metadata = GroupMetadata(
                                tag="SubGroup",
                                tbpe_id=f"{prefix}_{key}" if prefix else key,
                                x=x, y=y, w=w, h=h,
                                element_tags=element_tags
                            )
                            subgroup_info[f"{prefix}_{key}" if prefix else key] = representative_metadata
                    elif isinstance(value, dict):
                        # 재귀적으로 하위 그룹들도 확인
                        collect_subgroup_info(value, f"{prefix}_{key}" if prefix else key)
            
            # subgroup 정보 수집
            collect_subgroup_info(parent_group)
            
            # 디버깅: subgroup 정보 출력
            print(f"Parent Group: {parent_group_key}")
            print(f"Collected subgroups: {list(subgroup_info.keys())}")
            for sg_key, sg_meta in subgroup_info.items():
                element_tag_counts = {}
                for tag in sg_meta.element_tags:
                    element_tag_counts[tag] = element_tag_counts.get(tag, 0) + 1
                print(f"  {sg_key}: position=({sg_meta.x}, {sg_meta.y}, {sg_meta.w}, {sg_meta.h}), elements={element_tag_counts}")
            
            # 2단계: Parent Group 전체를 재귀적으로 처리
            def process_group_recursively(group_dict, group_name_prefix=""):
                """그룹을 재귀적으로 처리하여 모든 하위 그룹들을 layout function으로 변환"""
                nonlocal group_id_counter, layout_functions
                
                print(f"\n=== PROCESSING GROUP: {group_name_prefix} ===")
                print(f"Group dict keys: {list(group_dict.keys())}")
                
                current_group_elements = []
                current_group_metadata = []
                nested_group_ids = []
                nested_group_positions = []
                
                for key, value in group_dict.items():
                    if isinstance(value, dict):
                        # 중첩된 그룹인 경우 재귀적으로 처리
                        nested_group_id = process_group_recursively(value, f"{group_name_prefix}_{key}")
                        if nested_group_id:
                            nested_group_ids.append(nested_group_id)
                            # subgroup 정보에서 대표 위치 가져오기
                            subgroup_key = f"{group_name_prefix}_{key}" if group_name_prefix else key
                            if subgroup_key in subgroup_info:
                                nested_group_positions.append(subgroup_info[subgroup_key])
                                print(f"  Added subgroup position: {subgroup_key} -> {subgroup_info[subgroup_key].tbpe_id}")
                    elif value is None and key in metadata_objects:
                        # 실제 요소인 경우
                        current_group_elements.append(key)
                        current_group_metadata.append(metadata_objects[key])
                        print(f"  Added element: {key}")
                
                # 현재 레벨의 요소들과 중첩된 그룹들이 있다면 layout function 생성
                if current_group_elements or nested_group_ids:
                    print(f"Processing group {group_name_prefix}: current_elements={len(current_group_elements)}, nested_groups={len(nested_group_ids)}")
                    
                    # 레이아웃 결정을 위한 메타데이터 준비
                    layout_metadata = []
                    
                    # Parent Group 레벨에서는 subgroup들의 위치만 사용
                    if group_name_prefix.startswith("Parent Group") and subgroup_info:
                        # Parent Group 레벨에서는 모든 subgroup들의 위치 사용
                        layout_metadata = list(subgroup_info.values())
                        print(f"  Using all subgroups for Parent Group: {[sg.tbpe_id for sg in layout_metadata]}")
                    elif nested_group_positions:
                        layout_metadata = nested_group_positions
                        print(f"  Using nested group positions: {len(nested_group_positions)}")
                    else:
                        # subgroup이 없는 경우에만 개별 요소들 사용
                        layout_metadata = current_group_metadata
                        print(f"  Using current group metadata: {len(current_group_metadata)}")
                    
                    print(f"  Final layout_metadata length: {len(layout_metadata)}")
                    
                    if layout_metadata:
                        # 디버깅: 레이아웃 결정에 사용되는 메타데이터 출력
                        print(f"Layout decision for {group_name_prefix}:")
                        print(f"  Using {len(layout_metadata)} elements for layout decision")
                        for i, meta in enumerate(layout_metadata):
                            print(f"    {i}: {meta.tbpe_id} at ({meta.x}, {meta.y}, {meta.w}, {meta.h})")
                        
                        # 전체 레이아웃 결정 (Parent Group의 경우 subgroup들에 대해 graph 패턴 허용)
                        layout_type, gap, graph_info = determine_layout_type(layout_metadata, allow_graph_patterns=group_name_prefix.startswith("Parent Group"))
                        print(f"  Determined layout: {layout_type.value}")
                        if graph_info:
                            print(f"  Graph pattern: {graph_info.get('pattern', 'none')}")
                        
                        # 현재 그룹의 모든 요소들 (실제 요소 + 중첩 그룹들)
                        all_elements = current_group_elements + nested_group_ids
                        
                        # 그룹 ID 생성
                        current_group_id = f"group_{group_id_counter}"
                        group_id_counter += 1
                        
                        # Layout function 생성
                        if layout_type == LayoutType.GRAPH and graph_info:
                            # GRAPH 타입인 경우 특별한 함수 생성
                            group_layout_function = {
                                "function": "applyGraphLayout",
                                "layoutType": layout_type.value,
                                "elementIds": all_elements,
                                "graphInfo": graph_info,
                                "groupId": current_group_id
                            }
                        else:
                            # 기존 레이아웃 타입들
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
        
        # Parent Groups과 a group의 대략적인 위치 추가
        for parent_group_key in semantic_group.keys():
            if (parent_group_key.startswith("Parent Group") or parent_group_key.startswith("a group")) and isinstance(semantic_group[parent_group_key], dict):
                # Parent Group 내 모든 요소들의 경계 계산
                all_parent_metadata = []
                parent_group = semantic_group[parent_group_key]
                
                for key, value in parent_group.items():
                    if (key.startswith("Subgroup") or key.startswith("b group")) and isinstance(value, dict):
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
            root_layout_type, root_gap, _ = determine_layout_type(top_level_positions, allow_graph_patterns=True)
            
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
            # 여러 Parent Group들과 a group의 위치 정보 수집
            parent_group_positions = []
            for parent_group_key in semantic_group.keys():
                if (parent_group_key.startswith("Parent Group") or parent_group_key.startswith("a group")) and isinstance(semantic_group[parent_group_key], dict):
                    # Parent Group 내 모든 요소들의 경계 계산
                    all_parent_metadata = []
                    parent_group = semantic_group[parent_group_key]
                    
                    for key, value in parent_group.items():
                        if (key.startswith("Subgroup") or key.startswith("b group")) and isinstance(value, dict):
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
            
            # Parent Groups 간의 레이아웃 결정 (subgroup 레벨이므로 graph 패턴 허용)
                root_layout_type, root_gap, root_graph_info = determine_layout_type(parent_group_positions, allow_graph_patterns=True)
                if root_layout_type == LayoutType.GRAPH and root_graph_info:
                    root_layout_function = {
                        "function": "applyGraphLayout",
                        "layoutType": root_layout_type.value,
                        "elementIds": parent_group_ids,
                        "graphInfo": root_graph_info,
                        "groupId": "root_group"
                    }
                else:
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
                root_layout_type, root_gap, root_graph_info = determine_layout_type(top_level_positions)
                
                if root_layout_type == LayoutType.GRAPH and root_graph_info:
                    root_layout_function = {
                        "function": "applyGraphLayout",
                        "layoutType": root_layout_type.value,
                        "elementIds": top_level_element_ids,
                        "graphInfo": root_graph_info,
                        "groupId": "root_group"
                    }
                else:
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
            continue        # 현재 노드에서 가장 가까운 방문하지 않은 노드 찾기
        next_idx = None
        for dist, neighbor_idx in distances[current_idx]:
            if neighbor_idx not in visited:
                next_idx = neighbor_idx
                break
        
        if next_idx is None:
            print("  POLYGON: ❌ Cannot form complete polygon chain")
            break
            
        polygon_order.append(next_idx)
        visited.add(next_idx)
        current_idx = next_idx

