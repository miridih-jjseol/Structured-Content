import os
import xml.etree.ElementTree as ET
from tqdm import tqdm

from pathlib import Path
import re
import html
import json
import math
import glob

from PIL import Image, ImageDraw, ImageFont
import base64
import wandb

def generate_color_palette(num_colors=12):
    # Base colors with high saturation and varying hue
    base_colors = [
        (230, 25, 75),   # Red
        (60, 180, 75),   # Green
        (0, 130, 200),   # Blue
        (145, 30, 180),  # Purple
        (70, 240, 240),  # Cyan
        (240, 50, 230),  # Magenta
        (210, 245, 60),  # Lime
        (250, 190, 190), # Pink
        (0, 128, 128),   # Teal
        (230, 190, 255), # Lavender
        (170, 110, 40),  # Brown
        (245, 130, 48),  # Orange
    ]
    return base_colors[:num_colors]

colors = generate_color_palette()

def clean_and_parse_json(json_str):
    # Find the last closing brace of the JSON object
    last_brace_index = json_str.rindex('}')
    
    # Extract only the JSON part
    clean_json = json_str[:last_brace_index + 1]
    
    # Parse the cleaned JSON string
    try:
        data = json.loads(clean_json)
        return data
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None

def draw_group_boxes(img, group_dict, element_dict):
    def get_group_bounds(group_data):
        x_coords = []
        y_coords = []
        
        # Check if FrameItem exists in this group (recursively)
        def has_frameitem_in_group(data):
            for element_id, value in data.items():
                if isinstance(value, dict):
                    if has_frameitem_in_group(value):
                        return True
                elif element_id in element_dict:
                    element = element_dict[element_id]
                    if element.get('tag') == 'FrameItem':
                        return True
            return False
        
        frameitem_exists = has_frameitem_in_group(group_data)
        
        for element_id, value in group_data.items():
            if isinstance(value, dict):
                # Recursive call for nested groups
                sub_x, sub_y = get_group_bounds(value)
                x_coords.extend(sub_x)
                y_coords.extend(sub_y)
            elif element_id in element_dict:
                element = element_dict[element_id]
                
                # Skip elements with priority 0 if they are not TEXT/SIMPLE_TEXT/FrameItem and FrameItem exists
                if frameitem_exists:
                    element_tag = element.get('tag', '')
                    element_priority = element.get('priority', 'Not_Exists')
                    
                    # Convert priority to int for comparison, default to 1 if not valid
                    try:
                        priority_int = int(element_priority) if element_priority != 'Not_Exists' else 1
                    except (ValueError, TypeError):
                        priority_int = 1
                    
                    # Skip if not TEXT/SIMPLE_TEXT/FrameItem and priority is 0
                    if element_tag not in ['TEXT', 'SIMPLE_TEXT', 'FrameItem'] and priority_int == 0:
                        continue
                
                x_coords.extend([element['x'], element['x'] + element['w']])
                y_coords.extend([element['y'], element['y'] + element['h']])
                
        return x_coords, y_coords

    def draw_group(draw_overlay, group_data, group_name, depth=0, parent_color=None, parent_number=None):
        if not isinstance(group_data, dict):
            return

        x_coords, y_coords = get_group_bounds(group_data)
        
        if not x_coords or not y_coords:
            return
            
        # Calculate bbox coordinates
        min_x = min(x_coords)
        min_y = min(y_coords)
        max_x = max(x_coords)
        max_y = max(y_coords)
        
        # Determine color and opacity
        if parent_color is None:
            # This is a parent group, assign a new color
            color = colors[len(drawn_parent_groups) % len(colors)]
            drawn_parent_groups.add(group_name)
            # Parent groups have 90% transparency (opacity = 25)
            opacity = 25
            display_name = f"Parent group {parent_number}"
        else:
            # Use parent's color for subgroups
            color = parent_color
            # Subgroups get progressively more opaque
            opacity = min(255, 50 + (depth * 40))#10#min(128,20+(depth * 10))#min(255, 50 + (depth * 40))
            display_name = group_name
            if depth>=1:
                display_name = "Sub_"*(depth-1)+display_name
        # Create color with opacity
        fill_color = (*color, 100)

        # Draw filled rectangle
        #if depth==0:
        #    draw_overlay.rectangle(
        #    [min_x, min_y, max_x, max_y],
        #    fill=fill_color
        #)
        
        # Draw border with semi-transparency
        
        border_opacity = min(100, opacity + 50)  # Border slightly more visible than fill
        draw_overlay.rectangle(
            [min_x, min_y, max_x, max_y],
            #fill=fill_color
            outline= fill_color,#(0, 0, 0, 255),  # Border color matches group color
            width=7
        )
        
        # Add group name
        font_size = 30
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()

        text_bbox = draw_overlay.textbbox((0, 0), display_name, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        if depth == 0:
            text_x = max_x - text_width-5
            text_y = min_y - text_height - 5  # Position above the box
        else:
            text_x = min_x
            text_y = min_y - text_height - 5  # Position below the box

        # Draw semi-transparent text background
        padding = 5
        bg_opacity = min(200, opacity + 100)  # Background more visible than the box
        draw_overlay.rectangle(
            [text_x - padding, text_y - padding,
             text_x + text_width + padding, text_y + text_height + padding],
            fill=(*color, bg_opacity)  # Use group color for background with higher opacity
        )

        # Draw text with semi-transparency
        text_opacity = min(255, opacity + 150)  # Text more visible than background
        draw_overlay.text(
            (text_x, text_y),
            display_name,
            fill=(0, 0, 0, text_opacity),  # Semi-transparent black text
            font=font
        )
        
        # Recursively process subgroups
        for subgroup_name, subgroup_data in group_data.items():
            if isinstance(subgroup_data, dict):
                draw_group(draw_overlay, subgroup_data, subgroup_name, depth + 1, color, parent_number)

    # Create a new RGBA overlay
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    
    # Keep track of parent groups we've drawn to assign consistent colors
    drawn_parent_groups = set()
    
    # Process each parent group
    parent_groups = [i for i in list(group_dict.keys()) if i.startswith("Parent")]
    # parent group visualization
    idx =0
    for idx, parent_group in enumerate(parent_groups, 1):
        parent_color = colors[idx]
        draw_group(draw_overlay, group_dict[parent_group], parent_group, parent_number=idx,parent_color=parent_color)
    
    p_idx = idx
    a_groups = [i for i in list(group_dict.keys()) if i.startswith("a group")]
    # a group visualization
    for idx, a_group in enumerate(a_groups, 1):
        a_color = colors[idx+p_idx]
        draw_group(draw_overlay, group_dict[a_group], a_group, parent_number=idx,parent_color=a_color)

    
    # Composite the overlay onto the main image
    img = Image.alpha_composite(img.convert('RGBA'), overlay)
    return img


def draw_element_boxes(img, element_dict):
    # Open the image
    draw = ImageDraw.Draw(img)
    
    # Create a transparent overlay
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    
    # Different colors for different element types
    color_map = {
        'TEXT': (255, 0, 0,100),    # Red for text
        'LineShapeItem': (0, 0, 255, 50),  # Blue for lines
        'GENERALSVG': (70, 240, 240, 50),  # Cyan
    }
    
    # Try to load a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    except:
        font = ImageFont.load_default()
    
    # Draw boxes for each element
    for element_id, element in element_dict.items():
        x = element['x']
        y = element['y']
        w = element['w']
        h = element['h']
        tag = element['tag']
        
        # Get color based on element type
        if type(tag) == list:
            tag = tag[0]
        color = color_map.get(tag, (0, 255, 0, 50))  # Default to green if type unknown
        
        # Draw filled rectangle
        #draw_overlay.rectangle(
        #    [x, y, x + w, y + h],
        #    fill=color
        #)
        
        # Draw border
        draw_overlay.rectangle(
            [x, y, x + w, y + h],
            outline=(0, 0, 0, 255),  # Black border
            width=5
        )
        
        # Prepare label text
        #if isinstance(element['text_content'], list):
        #    label = element['text_content'][0] if element['text_content'] else element_id
        #else:
        #    label = str(element['text_content']) if element['text_content'] != 'None' else element_id
        label = element_id
            
        # Calculate text size
        text_bbox = draw_overlay.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Position for label (above the box)
        text_x = x
        text_y = y - text_height - 2
        
        # Draw text background
        padding = 2
        draw_overlay.rectangle(
            [text_x - padding, text_y - padding,
             text_x + text_width + padding, text_y + text_height + padding],
            fill=(255, 255, 255, 200)  # White background with some transparency
        )
        
        # Draw text
        draw_overlay.text(
            (text_x, text_y),
            label,
            fill=(0, 0, 0, 255),  # Black text
            font=font
        )  
        
    
    # Composite the overlay onto the main image
    img = Image.alpha_composite(img.convert('RGBA'), overlay)

    return img
    
def get_single_or_list(values):
    return values[0] if len(values) == 1 else values

def rgb_to_hex(rgb_value):
    if isinstance(rgb_value, list):
        return [rgb_to_hex(item) for item in rgb_value]
    try:
        if isinstance(rgb_value, str) and rgb_value.startswith('rgb('):
            r, g, b = map(int, re.findall(r'\d+', rgb_value))
            return f'#{r:02x}{g:02x}{b:02x}'
    except Exception:
        pass
    return rgb_value

def convert_colors_to_hex(colors):
    return [rgb_to_hex(color) for color in colors] if isinstance(colors, list) else rgb_to_hex(colors)

def extract_text_content(xml_string):
    tag_matches = re.findall(r'<(\w+)[^>]*\bTbpeId="([^"]+)"', xml_string)
    results = {}
    for tag, tbpe_id in tag_matches:
        pattern = rf'<{tag}[^>]*\bTbpeId="{tbpe_id}"[^>]*>.*?<TextBody>(.*?)</TextBody>'
        text_body_matches = re.findall(pattern, xml_string, re.DOTALL)
        text_contents = []
        if tag == 'SIMPLE_TEXT':
            for content in text_body_matches:
                try:
                    json_data = json.loads(content)
                    runs = []
                    for para in json_data.get('c', []):
                        runs.extend(flatten_runs(para))
                    paragraph_contents = [html.unescape(r) for r in runs if isinstance(r, str)]
                    text_contents.append(get_single_or_list(paragraph_contents) or 'Not_Exists')
                except Exception:
                    text_contents.append('Not_Exists')
        elif tag == 'TEXT':
            text_matches = re.findall(rf'<{tag}[^>]*\bTbpeId="{tbpe_id}"[^>]*>.*?<Text>(.*?)</Text>', xml_string, re.DOTALL)
            text_contents = [html.unescape(tc) for tc in text_matches] if text_matches else ['Not_Exists']
        else:
            text_contents = ['Not_Exists'] * (len(text_body_matches) or 1)
        results[tbpe_id] = text_contents
    return results

def flatten_runs(paragraph):
    if isinstance(paragraph, dict):
        runs = []
        for item in paragraph.get('c', []):
            if isinstance(item, dict):
                runs.extend(flatten_runs(item))
            elif isinstance(item, str):
                runs.append(item)
        return runs
    return []

def extract_text_attributes(xml_string):
    text_body_matches = re.findall(r'<TextBody>(.*?)</TextBody>', xml_string, re.DOTALL)
    if '<SIMPLE_TEXT' not in xml_string:
        return [{'text_align': 'Not_Exists', 'font_size': 'Not_Exists', 'font_type': 'Not_Exists', 'font_color': 'Not_Exists', 'vertical_align': 'Not_Exists'}]

    attributes_list = []
    for content in text_body_matches:
        try:
            text_body = json.loads(content)
            text_align = text_body.get('bp', {}).get('txal', 'Not_Exists')
            vertical_align = text_body.get('bp', {}).get('vtln', 'Not_Exists')
            font_sizes, font_types, font_colors = [], [], []
            for para in text_body.get('c', []):
                extract_font_attributes(para, font_sizes, font_types, font_colors)
            attributes_list.append({
                'text_align': text_align,
                'font_size': get_single_or_list(font_sizes),
                'font_type': get_single_or_list(font_types),
                'font_color': get_single_or_list(font_colors),
                'vertical_align': vertical_align
            })
        except Exception:
            attributes_list.append({'text_align': 'Not_Exists', 'font_size': 'Not_Exists', 'font_type': 'Not_Exists', 'font_color': 'Not_Exists', 'vertical_align': 'Not_Exists'})
    return attributes_list

def extract_font_attributes(paragraph, font_sizes, font_types, font_colors):
    if isinstance(paragraph, dict):
        for run in paragraph.get('c', []):
            if isinstance(run, dict):
                if 'rp' in run:
                    rp = run['rp']
                    font_sizes.append(str(rp.get('size', 'Not_Exists')))
                    font_types.append(rp.get('fmly', 'Not_Exists'))
                    font_colors.append(rp.get('fill', 'Not_Exists'))
                else:
                    extract_font_attributes(run, font_sizes, font_types, font_colors)

def merge_text_attributes(attributes_list):
    merged = {'text_align': [], 'font_size': [], 'font_type': [], 'font_color': [], 'vertical_align': []}
    for attrs in attributes_list:
        for key in merged.keys():
            merged[key].append(attrs.get(key, 'Not_Exists'))
    return {k: get_single_or_list(v) for k, v in merged.items()}

def parse_positions(positions):
    lefts, rights, tops, bottoms = [], [], [], []
    for pos in positions:
        attributes = dict(re.findall(r'(\w+)="([^"]*)"', pos))
        lefts.append(float(attributes.get('Left', 0)))
        rights.append(float(attributes.get('Right', 0)))
        tops.append(float(attributes.get('Top', 0)))
        bottoms.append(float(attributes.get('Bottom', 0)))
    return lefts, rights, tops, bottoms

def get_element_info_to_xml(xml_string):
    tag_matches = re.findall(r'<(\w+)[^>]*\bTbpeId="([^"]+)"', xml_string)
    if not tag_matches:
        return None
    tags = ["PHOTO" if tag == "STICKER" else "FrameItem" if tag == "SVGIMAGEFRAME" else tag for tag, _ in tag_matches]
    tbpe_ids = [tbpe_id for _, tbpe_id in tag_matches]
    is_texts = ['TRUE' if tag in ('SIMPLE_TEXT', 'TEXT') else 'FALSE' for tag in tags]

    text_contents = extract_text_content(xml_string)
    text_values = [text_contents.get(tbpe_id, ['Not_Exists'])[0] for tbpe_id in tbpe_ids]
    merged_text_attributes = merge_text_attributes(extract_text_attributes(xml_string))

    rotate = re.findall(r'\bRotate="([^"]+)"', xml_string)
    opacity = re.findall(r'\bOpacity="([^"]+)"', xml_string)
    priority_match = re.search(r'\bPriority="([^"]+)"', xml_string)

    positions = re.findall(r'<Position\s+([^>]+)/>', xml_string)
    if positions:
        left_values, right_values, top_values, bottom_values = parse_positions(positions)
    else:
        left_values, right_values, top_values, bottom_values = [0], [0], [0], [0]

    return {
        'tag': get_single_or_list(tags),
        'is_text': get_single_or_list(is_texts),
        'text_content': get_single_or_list(text_values),
        'tbpe_id': get_single_or_list(tbpe_ids),
        'resource_key': 'Not_Exists',
        'left': get_single_or_list(left_values),
        'top': get_single_or_list(top_values),
        'element_width': get_single_or_list([r - l for l, r in zip(left_values, right_values)]),
        'element_height': get_single_or_list([b - t for t, b in zip(top_values, bottom_values)]),
        'rotation': get_single_or_list(rotate),
        'opacity': get_single_or_list(opacity),
        'priority': priority_match.group(1) if priority_match else 'Not_Exists',
        **merged_text_attributes
    }

def split_elements(xml_string):
    # TbpeId 별로 요소 분리
    tag_matches = re.findall(r'<(\w+)[^>]*\bTbpeId="([^"]+)"', xml_string)
    elements = []
    for tag, tbpe_id in tag_matches:
        pattern = rf'<{tag}[^>]*\bTbpeId="{tbpe_id}"[^>]*>.*?</{tag}>'
        match = re.search(pattern, xml_string, re.DOTALL)
        if match:
            elements.append(match.group(0))
    return elements

def replace_tbpe_with_label_in_xml(input_path, output_path, tbpe2label):
    # 1. XML 파일 전체를 문자열로 읽기
    with open(input_path, 'r', encoding='utf-8') as f:
        xml = f.read()
    
    # 2. tbpe → label 치환
    for tbpe, label in tbpe2label.items():
        xml = xml.replace(f'TbpeId="{tbpe}"', f'TbpeId="{label}"')
        xml = xml.replace(f"TbpeId='{tbpe}'", f"TbpeId='{label}'")  # 홑따옴표도 커버

    # 3. 결과 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(xml)

def xml_to_mllm_converter(xml_path,img_path=None):
    xml_path = Path(xml_path)
    xml_string = xml_path.read_text(encoding='utf-8')
    element_fragments = split_elements(xml_string)
    elements_metadata = {}
    # extract metadata from each element fragment
    for idx, elem_xml in enumerate(element_fragments, 1):
        elem_data = get_element_info_to_xml(elem_xml)
        tbpe_id = elem_data['tbpe_id']
        if type(tbpe_id) == list:
            tbpe_id = tbpe_id[0]
        #keys = ['tag', 'is_text', 'text_content', 'tbpe_id', 'resource_key', 'left', 'top', 'element_width', 'element_height', 'rotation', 'opacity', 'priority', 'text_align', 'font_size', 'font_type', 'font_color', 'vertical_align']
        keys = ['tag', 'text_content', 'tbpe_id','left', 'top', 'element_width', 'element_height', 'priority']
        metadata = {}
        for key in keys:
            value = elem_data[key]
            if value == 'SIMPLE_TEXT':
                value = 'TEXT'
            if key in ['left', 'top', 'element_width', 'element_height']:
                try:
                    value = float(value)
                    if math.isinf(value) or math.isnan(value):
                        value = 0  # 무한대나 NaN이면 0으로 처리
                    else:

                        value = int(round(value))
                except (ValueError, TypeError):
                    value = 0
            #key = key.replace('SIMPLE_TEXT', 'TEXT')
            key = key.replace('left', 'x').replace('top', 'y').replace('element_width', 'w').replace('element_height', 'h')
            metadata[key] = value
            
        elements_metadata[tbpe_id] = metadata
    
    tbpeid2label = {}
    label_count = {}
    # extract counting label from tbpe_id
    for tbpe_id, metadata in elements_metadata.items():

        if type(metadata['tag']) == list:
            name = '_'.join(metadata['tag'])
        elif type(metadata['tag']) == str:
            name = metadata['tag']
        else:
            raise ValueError(f"Invalid tag type: {type(metadata['tag'])}")
        
        # 이름 중복 처리
        if name in label_count:
            label_count[name] += 1
        else:
            label_count[name] = 1
        final_name = f"{name}_{label_count[name]}"
        tbpeid2label[tbpe_id] = final_name        
    # convert tbpe_id to label
    output_metadata = {}
    for tbpe_id, metadata in tbpeid2label.items():
        elem_meta = elements_metadata[tbpe_id]
        elem_meta['tbpe_id'] = metadata
        if elem_meta['text_content'] == 'Not_Exists':
            elem_meta['text_content'] = 'None'
        output_metadata[metadata] = elem_meta\
    
    json_string = json.dumps(output_metadata, ensure_ascii=False)
    mllm_data = {
        "messages": [
            {
                "content": f"<image> {json_string}",
                "role": "user"
            },
            {
                "content": "None",
                "role": "assistant"
            }
        ],
        "images": [img_path if img_path else None]
    }
    return mllm_data