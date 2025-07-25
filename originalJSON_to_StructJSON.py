import os
import json
from collections import defaultdict
from tqdm import tqdm

# 경로 설정
JSON_DIR = "/data/shared/jjkim/original_json"
OUTPUT_DIR = "/data/shared/jjkim/dataset"

# ===== Utility Functions =====

def safe_load_json(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[Error] Failed to load {filepath}: {e}")
        return None

def write_file(path, content):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

# ===== ID Mapping Builders =====

def build_id2labels(output_data, use_suffix=False):
    tag_list = [
        "STICKER", "SVG_PRIVATE", "PHOTO", "GIF", "YOUTUBE", "QRCode", "SHAPESVG",
        "SIMPLE_TEXT", "TEXT", "GROUP", "GRID", "SVGIMAGEFRAME", "LineShapeItem",
        "CHILDTEXT", "FrameItem", "GENERALSVG", "VIDEO", "Chart", "SVG"
    ]
    id2labels = {}
    label_counts = {}

    for item in output_data:
        node_id = item['id']
        labels = item.get('value', {}).get('rectanglelabels', [])
        selected_label = next((label for label in labels if label in tag_list), None)

        if use_suffix and selected_label:
            label_counts[selected_label] = label_counts.get(selected_label, 0) + 1
            selected_label = f"{selected_label}_{label_counts[selected_label]}"

        id2labels[node_id] = selected_label
    return id2labels

def build_id2tbpe(output_data):
    return {
        item['id']: item['value']['TbpeId']
        for item in output_data if 'TbpeId' in item['value']
    }

def build_id2group(output_data):
    group_tags = ['Parent Group', 'Subgroup', 'a group', 'b group']
    return {
        item['id']: next((label for label in item['value']['rectanglelabels'] if label in group_tags), 'None')
        for item in output_data
    }

def build_tbpe2label(id2tbpe, id2label):
    return {
        tbpe: id2label[node_id]
        for node_id, tbpe in id2tbpe.items()
        if tbpe and id2label.get(node_id)
    }

# ===== Structure Builders =====

def build_nested_dict(outputs, root_id):
    id_to_node = {}
    children_map = defaultdict(list)

    for item in outputs:
        node_id = item['id']
        parent_id = item.get('parent_id')
        if not parent_id:
            raise ValueError(f"[Error] parent_id missing for {node_id}")
        id_to_node[node_id] = {}
        children_map[parent_id].append(node_id)

    for parent_id, child_ids in children_map.items():
        id_to_node.setdefault(parent_id, {})
        for child_id in child_ids:
            id_to_node[parent_id][child_id] = id_to_node[child_id]

    return {
        item['id']: id_to_node[item['id']]
        for item in outputs if item.get("parent_id") == root_id
    }

def replace_empty_dicts_with_none(data):
    if isinstance(data, dict):
        return {
            k: replace_empty_dicts_with_none(v) if v != {} else None
            for k, v in data.items()
        }
    return data

def map_ids_to_labels(struct, id2label, id2group, group_counts=None):
    if group_counts is None:
        group_counts = {}

    new_struct = {}
    for k, v in struct.items():
        new_key = id2label.get(k)
        if not new_key:
            group = id2group.get(k)
            if group:
                group_counts[group] = group_counts.get(group, 0) + 1
                new_key = f"{group} {group_counts[group]}"
            else:
                new_key = k

        new_struct[new_key] = (
            map_ids_to_labels(v, id2label, id2group, group_counts) if isinstance(v, dict) else v
        )
    return new_struct

def replace_tbpe_in_xml(xml_path, save_path, tbpe2label):
    with open(xml_path, 'r', encoding='utf-8') as f:
        xml = f.read()

    for tbpe, label in tbpe2label.items():
        xml = xml.replace(f'TbpeId="{tbpe}"', f'TbpeId="{label}"')
        xml = xml.replace(f"TbpeId='{tbpe}'", f"TbpeId='{label}'")

    write_file(save_path, xml)

# ===== Main Pipeline =====

def process_json_files(json_dir):
    for fname in tqdm(os.listdir(json_dir)):
        if not fname.endswith(".json"):
            continue

        file_path = os.path.join(json_dir, fname)
        data = safe_load_json(file_path)
        if not data:
            continue

        try:
            root_id = fname.replace(".json", "")
            output_data = data['annotations'][0]['result']

            id2tbpe = build_id2tbpe(output_data)
            id2group = build_id2group(output_data)
            id2label_suffix = build_id2labels(output_data, use_suffix=True)
            tbpe2label = build_tbpe2label(id2tbpe, id2label_suffix)

            struct = build_nested_dict(output_data, root_id)
            struct = replace_empty_dicts_with_none(struct)
            mapped_struct = map_ids_to_labels(struct, id2label_suffix, id2group)

            # output_file = os.path.join(OUTPUT_DIR, f"{root_id}.json")
            # write_file(output_file, json.dumps(mapped_struct, indent=2, ensure_ascii=False))

        except Exception as e:
            print(f"[Error] Failed to process {fname}: {e}")

# 실행
if __name__ == "__main__":
    process_json_files(JSON_DIR)
