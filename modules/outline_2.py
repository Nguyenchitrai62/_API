import cv2
import numpy as np
import json
import os
from pathlib import Path
from typing import List, Union, Tuple, Dict, Any
from collections import defaultdict
from shapely.geometry import Polygon
from shapely import make_valid

# ================================================================= #
# CÁC HÀM TIỆN ÍCH CƠ BẢN
# ================================================================= #

def is_contour_inside_another(inner_contour, outer_contour):
    """
    Kiểm tra xem một contour có nằm hoàn toàn bên trong contour khác không.
    """
    test_points = inner_contour[::max(1, len(inner_contour)//5)]
    
    for point in test_points:
        pt = (float(point[0][0]), float(point[0][1]))
        if cv2.pointPolygonTest(outer_contour, pt, False) < 0:
            return False
    
    return True

def are_contours_duplicate_optimized(contour1: np.ndarray, contour2: np.ndarray, tolerance: int = 10) -> bool:
    """Kiểm tra hai contour có trùng lặp không bằng cách so sánh các đỉnh xấp xỉ."""
    epsilon1 = 0.02 * cv2.arcLength(contour1, True)
    epsilon2 = 0.02 * cv2.arcLength(contour2, True)
    approx1 = cv2.approxPolyDP(contour1, epsilon1, True)
    approx2 = cv2.approxPolyDP(contour2, epsilon2, True)

    if len(approx1) != len(approx2): 
        return False
    
    num_vertices = len(approx1)
    if num_vertices < 3: 
        return False
        
    vertices1 = approx1.reshape(-1, 2).astype(np.float32)
    vertices2 = approx2.reshape(-1, 2).astype(np.float32)

    for start_idx in range(num_vertices):
        rotated_vertices2 = np.roll(vertices2, start_idx, axis=0)
        if np.all(np.linalg.norm(vertices1 - rotated_vertices2, axis=1) <= tolerance): 
            return True
        reversed_vertices2 = np.flip(rotated_vertices2, axis=0)
        if np.all(np.linalg.norm(vertices1 - reversed_vertices2, axis=1) <= tolerance): 
            return True
    return False

def remove_duplicate_contours_optimized(contours: List[np.ndarray], tolerance: int = 10) -> List[np.ndarray]:
    """
    Loại bỏ các contour trùng lặp.
    Refactor: Đơn giản hóa để chỉ trả về list contour, vì giá trị thứ hai (indices) không được sử dụng.
    """
    if len(contours) <= 1: 
        return contours

    areas = np.array([cv2.contourArea(c) for c in contours])
    sorted_indices = np.argsort(areas)[::-1]
    
    unique_contours, unique_areas, unique_bboxes = [], [], []
    original_indices_of_uniques = []

    for idx in sorted_indices:
        contour, current_area, current_bbox = contours[idx], areas[idx], cv2.boundingRect(contours[idx])
        is_duplicate = False
        for j, kept_contour in enumerate(unique_contours):
            kept_area, kept_bbox = unique_areas[j], unique_bboxes[j]
            # Bỏ qua nếu diện tích hoặc kích thước quá khác biệt
            if abs(current_area - kept_area) / max(current_area, kept_area, 1) > 0.2: 
                continue
            if (abs(current_bbox[2] - kept_bbox[2]) > tolerance or abs(current_bbox[3] - kept_bbox[3]) > tolerance): 
                continue
            if are_contours_duplicate_optimized(contour, kept_contour, tolerance):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_contours.append(contour)
            unique_areas.append(current_area)
            unique_bboxes.append(current_bbox)
            original_indices_of_uniques.append(idx)
    
    # Sắp xếp lại các contour duy nhất theo thứ tự ban đầu của chúng
    final_contours = [contours[i] for i in sorted(original_indices_of_uniques)]
    return final_contours


def _check_rectangle_angles(approx_contour: np.ndarray) -> int:
    """Hàm trợ giúp: Đếm số góc gần vuông (85-95 độ) trong một contour xấp xỉ 4 đỉnh."""
    if len(approx_contour) != 4:
        return 0
    
    angles = []
    for i in range(4):
        p1 = approx_contour[i][0]
        p2 = approx_contour[(i + 1) % 4][0]
        p3 = approx_contour[(i + 2) % 4][0]
        
        v1 = p1 - p2
        v2 = p3 - p2
        
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm_product == 0: continue

        cos_angle = np.dot(v1, v2) / norm_product
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
        angles.append(angle)
    
    return sum(1 for angle in angles if 80 <= angle <= 100)


def classify_shape(contour: np.ndarray) -> Tuple[str, List[List[int]]]:
    """
    Phân loại hình dạng của contour dựa trên đặc trưng hình học.
    KHÔI PHỤC HOÀN TOÀN LOGIC GỐC.
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    if area == 0 or perimeter == 0:
        return "unknown", []
    
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 0
    
    epsilon1 = 0.005 * perimeter
    epsilon2 = 0.015 * perimeter
    epsilon3 = 0.04 * perimeter
    
    approx1 = cv2.approxPolyDP(contour, epsilon1, True)
    approx2 = cv2.approxPolyDP(contour, epsilon2, True)
    approx3 = cv2.approxPolyDP(contour, epsilon3, True)
    
    vertices_list = [[int(point[0][0]), int(point[0][1])] for point in approx1]
    
    vertices1, vertices2, vertices3 = len(approx1), len(approx2), len(approx3)
    
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    hull_perimeter = cv2.arcLength(hull, True)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    contour_noise = perimeter / hull_perimeter if hull_perimeter > 0 else 0
    
    ellipse_ratio = 0
    if len(contour) >= 5:
        try:
            ellipse = cv2.fitEllipse(contour)
            ellipse_ratio = min(ellipse[1]) / max(ellipse[1]) if max(ellipse[1]) > 0 else 0
        except:
            pass # Bỏ qua lỗi fitEllipse
    
    if (vertices3 == 4 and _check_rectangle_angles(approx3) >= 3 and solidity > 0.7):
        return "Pipe", vertices_list
    
    elif (contour_noise > 1.1 and vertices3 < 5 and vertices3 > 2):
        return "Air Filter", vertices_list
    
    elif (contour_noise > 1.1):
        return "Noise", vertices_list
    
    elif (vertices3 == 4 and _check_rectangle_angles(approx3) == 2 and 
        solidity > 0.8 and circularity < 0.8):
        return "Eccentric Reducer", vertices_list
    
    elif (vertices3 == 4 and _check_rectangle_angles(approx3) < 2 and 
        solidity > 0.8 and circularity < 0.7):
        return "Concentric Reducer", vertices_list

    elif vertices3 == 3 and solidity > 0.7:
        return "Supply Air Fan", vertices_list
    
    elif vertices3 == 5 and solidity > 0.8 and circularity > 0.6:
        return "Elbow 90", vertices_list
    
    elif aspect_ratio > 5 or aspect_ratio < 0.2:
        return "Pipe", vertices_list
    
    elif vertices3 == 4:
        return "Elbow 45", vertices_list
    
    else:
        return "Noise", vertices_list

def analyze_shape_details(contour: np.ndarray, shape_name: str, vertices_list: List[List[int]], source_image_name: str = "", pipe_id: Union[int, None] = None) -> Dict[str, Any]:
    """Phân tích và trích xuất thông tin chi tiết của một hình dạng."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)
    M = cv2.moments(contour)
    
    cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else x + w // 2
    cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else y + h // 2
    
    return {
        "name": shape_name, "area": round(area, 2), "perimeter": round(perimeter, 2),
        "centroid": (cx, cy), "bounding_box": (x, y, w, h),
        "aspect_ratio": round(float(w) / h, 2) if h > 0 else 0,
        "source_image": source_image_name, "pipe_id": pipe_id, "vertices": vertices_list
    }

def get_color_for_shape(shape_name: str, shape_colors: Dict[str, Tuple[int, int, int]], lock) -> Tuple[int, int, int]:
    """Lấy hoặc tạo một màu mới cho một loại hình dạng."""
    with lock:
        if shape_name not in shape_colors:
            color_palette = [
                (255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100),
                (255, 100, 255), (100, 255, 255), (255, 150, 50), (150, 100, 255),
            ]
            shape_colors[shape_name] = color_palette[len(shape_colors) % len(color_palette)]
        return shape_colors[shape_name]

# ================================================================= #
# CÁC HÀM PHÁT HIỆN KẾT NỐI
# ================================================================= #

def find_connected_shapes_by_centerline(
    detected_shapes: List[Dict], all_contours: List[np.ndarray],
    max_distance_ratio: float = 3.0
) -> List[Dict]:
    """
    Tìm các cặp hình dạng kết nối và lưu thông tin kết nối.
    Pipe-Pipe luôn dùng centroid mặc định, không dùng tâm động.
    """
    num_shapes = len(detected_shapes)
    if num_shapes < 2: 
        return detected_shapes

    for shape in detected_shapes:
        shape['connected_to'] = []
        shape['connection_lines'] = []
    
    for i in range(num_shapes):
        for j in range(i + 1, num_shapes):
            shape1, shape2 = detected_shapes[i], detected_shapes[j]
            contour1, contour2 = all_contours[i], all_contours[j]
            
            center1 = np.array(shape1['centroid'])
            center2 = np.array(shape2['centroid'])
            distance = np.linalg.norm(center2 - center1)
            
            avg_size1 = (shape1['bounding_box'][2] + shape1['bounding_box'][3]) / 2
            avg_size2 = (shape2['bounding_box'][2] + shape2['bounding_box'][3]) / 2
            max_allowed_distance = max_distance_ratio * max(avg_size1, avg_size2, 1) + 6
            
            if distance > max_allowed_distance: 
                continue
            
            poly1 = Polygon(contour1.reshape(-1, 2)).simplify(3)
            poly2 = Polygon(contour2.reshape(-1, 2)).simplify(3)
            
            if not poly1.is_valid:
                poly1 = make_valid(poly1)
            if not poly2.is_valid:
                poly2 = make_valid(poly2)
            
            buffered1 = poly1.buffer(3)
            buffered2 = poly2.buffer(3)
            
            if not buffered1.intersects(buffered2):
                continue
            
            is_connected = True
            
            if shape1['name'] == "Pipe" and shape2['name'] == "Pipe":
                dx = center2[0] - center1[0]
                dy = center2[1] - center1[1]
                angle = np.degrees(np.arctan2(dy, dx))
                abs_angle = abs(angle) % 180
                is_horizontal = abs(abs_angle - 0) <= 2 or abs(abs_angle - 180) <= 2
                is_vertical = abs(abs_angle - 90) <= 2
                if not (is_horizontal or is_vertical):
                    is_connected = False
            
            if is_connected:
                connection_type = f"{shape1['name']}-{shape2['name']}"
                
                shape1['connected_to'].append({
                    'shape_index': j, 
                    'shape_name': shape2['name'], 
                    'distance': round(distance, 2),
                    'connection_type': connection_type
                })
                shape2['connected_to'].append({
                    'shape_index': i, 
                    'shape_name': shape1['name'], 
                    'distance': round(distance, 2),
                    'connection_type': connection_type
                })
                
                connection_line = [[center1[0], center1[1]], [center2[0], center2[1]]]
                shape1['connection_lines'].append(connection_line)
                shape2['connection_lines'].append(connection_line)
    
    return detected_shapes

def draw_connections_on_image(image: np.ndarray, detected_shapes: List[Dict]) -> np.ndarray:
    """Vẽ các đường kết nối và tâm lên ảnh."""
    result_image = image.copy()
    drawn_lines = set()

    for shape in detected_shapes:
        # Vẽ tâm
        center = tuple(map(int, shape['centroid']))
        cv2.circle(result_image, center, 8, (255, 0, 0), -1)
        
        # Vẽ đường kết nối từ thông tin đã lưu
        if 'connection_lines' in shape:
            for line in shape['connection_lines']:
                pt1 = tuple(map(int, line[0]))
                pt2 = tuple(map(int, line[1]))
                line_key = tuple(sorted([pt1, pt2]))
                if line_key not in drawn_lines:
                    cv2.line(result_image, pt1, pt2, (0, 0, 255), 3)
                    drawn_lines.add(line_key)
    
    return result_image

# ================================================================= #
# HÀM XỬ LÝ CHÍNH
# ================================================================= #

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            self.parent[px] = py
        elif self.rank[px] > self.rank[py]:
            self.parent[py] = px
        else:
            self.parent[py] = px
            self.rank[px] += 1

def process_image(image, min_area_outline=0.0005):
    """
    Xử lý ảnh để tạo cả ảnh outline và masked dựa trên các vùng có diện tích >= % tổng diện tích
    
    Args:
        image (np.ndarray): Ảnh đầu vào dạng numpy array (BGR)
        min_area_outline (float): Tỉ lệ diện tích tối thiểu (0–1) của vùng cần giữ lại
                          Ví dụ: 0.01 nghĩa là vùng >= 1% tổng diện tích ảnh
    
    Returns:
        tuple: (outline_image, masked_image) - Ảnh outline và masked đã xử lý
        
    Raises:
        ValueError: Nếu ảnh đầu vào không hợp lệ
    """
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Ảnh đầu vào phải là numpy array hợp lệ")
    
    # Chuyển sang ảnh grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Áp dụng ngưỡng để tạo ảnh nhị phân
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Tìm các đường viền (contours)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Tổng diện tích ảnh
    total_area = image.shape[0] * image.shape[1]
    absolute_min_area_outline = min_area_outline * total_area   # đổi % sang giá trị tuyệt đối
    
    # Tạo ảnh outline (nền trắng với viền xanh)
    outline_image = np.ones_like(image) * 255
    
    # Tạo mask cho ảnh masked
    mask = np.zeros(gray.shape, dtype=np.uint8)
    
    kept_large_contours = []
    # Vẽ outline và tạo mask cho các vùng có diện tích >= absolute_min_area_outline
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= absolute_min_area_outline:
            cv2.drawContours(outline_image, [contour], -1, (255, 0, 0), 2)
            cv2.fillPoly(mask, [contour], 255)
            kept_large_contours.append(contour)
    
    # Tạo ảnh masked
    masked_image = np.ones_like(image) * 255
    for i in range(3):
        masked_image[:, :, i] = np.where(mask == 255, image[:, :, i], 255)
    
    return outline_image, masked_image, kept_large_contours

def detect_contours(image: np.ndarray, min_area_outline: float, min_area_object: float):
    outline_result, masked_result, large_contours = process_image(image, min_area_outline=min_area_outline)
    
    gray = cv2.cvtColor(masked_result, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Tổng diện tích ảnh
    total_area = image.shape[0] * image.shape[1]
    absolute_min_area_object = min_area_object * total_area   # đổi % sang giá trị tuyệt đối
    
    # Lọc contours hợp lệ (có cha và đủ lớn)
    valid_contours_with_idx = []
    if hierarchy is not None:
        for i, c in enumerate(contours):
            if hierarchy[0][i][3] != -1 and cv2.contourArea(c) >= absolute_min_area_object:
                valid_contours_with_idx.append((i, c))
    
    return masked_result, large_contours, valid_contours_with_idx

def group_contours_by_outline(large_contours: List[np.ndarray], valid_contours_with_idx: List[Tuple[int, np.ndarray]]):
    # Group by outline
    outline_groups = {idx: [] for idx in range(len(large_contours))}
    for orig_idx, small_c in valid_contours_with_idx:
        for outline_idx, large_c in enumerate(large_contours):
            if is_contour_inside_another(small_c, large_c):
                outline_groups[outline_idx].append(small_c)
                break
    return outline_groups

def process_group(outline_idx: int, group_contours: List[np.ndarray], duplicate_tolerance: int, image_name: str, max_distance_ratio: float):
    if not group_contours:
        return [], []
    
    unique_contours_group = remove_duplicate_contours_optimized(group_contours, duplicate_tolerance)
    
    group_shapes = []
    
    # Xử lý từng contour để phân loại và kiểm tra Vent
    for contour in unique_contours_group:
        shape_name, vertices_list = classify_shape(contour)
        if shape_name == "unknown": continue

        # THÊM LOGIC KIỂM TRA VENT (từ code 2)
        if shape_name == "Pipe":
            is_nested = False
            for other_contour in unique_contours_group:
                if cv2.contourArea(other_contour) > cv2.contourArea(contour):
                    if is_contour_inside_another(contour, other_contour):
                        shape_name = "Volume Control Damper"
                        is_nested = True
                        break
        
        shape_details = analyze_shape_details(contour, shape_name, vertices_list, image_name)
        shape_details['outline_id'] = outline_idx
        group_shapes.append(shape_details)
    
    if group_shapes:
        group_shapes = find_connected_shapes_by_centerline(
            group_shapes, unique_contours_group, 
            max_distance_ratio
        )
    
    return group_shapes, unique_contours_group

def detect_specials(detected_shapes: List[Dict]):
    n = len(detected_shapes)
    specials = []
    pair_dict = {}
    for i in range(n):
        shape = detected_shapes[i]
        if shape['name'] != 'Pipe': continue
        pipe_connections = [c for c in shape['connected_to'] if c['shape_name'] == 'Pipe']
        if len(pipe_connections) != 4: continue
        neighbors = [c['shape_index'] for c in pipe_connections]
        centroids = [detected_shapes[k]['centroid'] for k in neighbors]
        center = shape['centroid']
        vecs = [np.array(cent) - np.array(center) for cent in centroids]
        angles = [np.arctan2(v[1], v[0]) for v in vecs]
        used = [False] * 4
        pairs = []
        for _ in range(2):
            best_pair = None
            best_diff = float('inf')
            for kk in range(4):
                if used[kk]: continue
                for mm in range(kk+1, 4):
                    if used[mm]: continue
                    angle_diff = abs(angles[kk] - angles[mm])
                    angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
                    diff_180 = abs(angle_diff - np.pi)
                    if diff_180 < best_diff:
                        best_diff = diff_180
                        best_pair = (kk, mm)
            if best_pair is None or best_diff > np.radians(30):
                break
            k, m = best_pair
            used[k] = used[m] = True
            pairs.append((neighbors[k], neighbors[m]))
        if len(pairs) == 2:
            pair_dict[i] = pairs
            specials.append(i)
    return specials, pair_dict

def assign_pipe_ids(detected_shapes: List[Dict], global_shape_colors: Dict, lock):
    specials, pair_dict = detect_specials(detected_shapes)

    n = len(detected_shapes)
    virtual_start = n
    special_to_virtuals = {}
    for s in specials:
        special_to_virtuals[s] = (virtual_start, virtual_start + 1)
        virtual_start += 2
    total_nodes = virtual_start

    uf = UnionFind(total_nodes)

    # Union normal connections not involving specials
    special_set = set(specials)
    for i in range(n):
        for conn in detected_shapes[i]['connected_to']:
            j = conn['shape_index']
            if i < j:
                if i not in special_set and j not in special_set:
                    uf.union(i, j)

    # Union pairs to virtuals
    for s, pairs in pair_dict.items():
        v1, v2 = special_to_virtuals[s]
        p1, p2 = pairs
        uf.union(p1[0], v1)
        uf.union(p1[1], v1)
        uf.union(p2[0], v2)
        uf.union(p2[1], v2)

    # Assign root_to_id with global next_id
    with lock:
        next_id = global_shape_colors.get('_global_pipe_id', 1)
        root_to_id = {}
        for k in range(total_nodes):
            root = uf.find(k)
            if root not in root_to_id:
                root_to_id[root] = next_id
                next_id += 1
        global_shape_colors['_global_pipe_id'] = next_id

    for i in range(n):
        if i in special_set:
            v1, v2 = special_to_virtuals[i]
            r1 = uf.find(v1)
            r2 = uf.find(v2)
            ids = [root_to_id[r1], root_to_id[r2]]
            ids.sort()
            detected_shapes[i]['pipe_id'] = ids
        else:
            r = uf.find(i)
            detected_shapes[i]['pipe_id'] = root_to_id[r]

def create_polys(all_unique_contours: List[np.ndarray]):
    polys = []
    for cont in all_unique_contours:
        poly = Polygon(cont.reshape(-1, 2)).simplify(3)
        if not poly.is_valid:
            poly = make_valid(poly)
        polys.append(poly)
    return polys

def merge_small_pipe_groups(detected_shapes: List[Dict], polys: List[Polygon], small_threshold: int = 3, min_dist_threshold: float = 20):
    while True:
        id_to_shapes = defaultdict(list)
        for i, shape in enumerate(detected_shapes):
            pid = shape['pipe_id']
            if isinstance(pid, int):
                id_to_shapes[pid].append(i)
            else:
                for p in pid:
                    id_to_shapes[p].append(i)

        id_to_outline = {}
        for pid, shapes_idx in id_to_shapes.items():
            outline_ids = set(detected_shapes[i]['outline_id'] for i in shapes_idx)
            id_to_outline[pid] = next(iter(outline_ids))

        small_ids = [pid for pid in id_to_shapes if len(id_to_shapes[pid]) <= small_threshold]
        if not small_ids:
            break

        merged = False
        for small_id in small_ids:
            min_dist = float('inf')
            nearest_id = None
            small_outline = id_to_outline[small_id]
            for other_id in id_to_shapes:
                if other_id == small_id or id_to_outline[other_id] != small_outline:
                    continue
                current_min = float('inf')
                for si in id_to_shapes[small_id]:
                    poly_s = polys[si]
                    for oi in id_to_shapes[other_id]:
                        poly_o = polys[oi]
                        d = poly_s.distance(poly_o)
                        if d < current_min:
                            current_min = d
                if current_min < min_dist:
                    min_dist = current_min
                    nearest_id = other_id
            if nearest_id is not None and min_dist < min_dist_threshold:
                # Merge small_id into nearest_id
                for shape in detected_shapes:
                    pid = shape['pipe_id']
                    if isinstance(pid, int):
                        if pid == small_id:
                            shape['pipe_id'] = nearest_id
                    else:
                        new_pid = [p if p != small_id else nearest_id for p in pid]
                        new_pid = sorted(set(new_pid))
                        if len(new_pid) == 1:
                            shape['pipe_id'] = new_pid[0]
                        else:
                            shape['pipe_id'] = new_pid
                merged = True
                break  # Merge one, then recompute

        if not merged:
            break

def draw_results(image: np.ndarray, detected_shapes: List[Dict], all_unique_contours: List[np.ndarray], global_shape_colors: Dict, lock):
    outline_image = np.ones_like(image) * 255
    masked_image = np.ones_like(image) * 255
    for i, shape_details in enumerate(detected_shapes):
        contour_to_draw = all_unique_contours[i]
        shape_color = get_color_for_shape(shape_details['name'], global_shape_colors, lock)
        
        cv2.drawContours(masked_image, [contour_to_draw], -1, shape_color, -1)
        cv2.drawContours(masked_image, [contour_to_draw], -1, (0, 0, 0), 1)
        cv2.drawContours(outline_image, [contour_to_draw], -1, (0, 0, 0), 2)
    
    masked_image = draw_connections_on_image(masked_image, detected_shapes)
    outline_image = draw_connections_on_image(outline_image, detected_shapes)

    return outline_image, masked_image

def process_one_image(source, i, global_shape_colors, lock, 
                      min_area_object, min_area_outline, 
                      duplicate_tolerance, max_distance_ratio, 
                      small_threshold, min_dist_threshold):
    image, image_name = None, ""
    if isinstance(source, str):
        image, image_name = cv2.imread(source), Path(source).stem
        if image is None: 
            return None, None, None
    elif isinstance(source, np.ndarray):
        image, image_name = source.copy(), f"numpy_array_{i}"
    else: 
        return None, None, None
    
    # Lấy kích thước ảnh
    h_img, w_img = image.shape[:2]
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if np.all(gray == gray[0, 0]):
        return None, None, None
    
    masked_result, large_contours, valid_contours_with_idx = detect_contours(image, min_area_outline, min_area_object)
    outline_groups = group_contours_by_outline(large_contours, valid_contours_with_idx)
    
    detected_shapes = []
    all_unique_contours = []
    
    for outline_idx, group_contours in outline_groups.items():
        group_shapes, unique_contours_group = process_group(outline_idx, group_contours, duplicate_tolerance, image_name, max_distance_ratio)
        
        # Adjust shape_index to global
        offset = len(detected_shapes)
        for shape in group_shapes:
            for conn in shape.get('connected_to', []):
                conn['shape_index'] += offset
        
        # Gắn thêm kích thước ảnh gốc cho từng shape
        for shape in group_shapes:
            shape["w_image"] = w_img
            shape["h_image"] = h_img
        
        detected_shapes.extend(group_shapes)
        all_unique_contours.extend(unique_contours_group)
    
    assign_pipe_ids(detected_shapes, global_shape_colors, lock)
    
    polys = create_polys(all_unique_contours)
    merge_small_pipe_groups(detected_shapes, polys, small_threshold=small_threshold, min_dist_threshold=min_dist_threshold)
    
    outline_img, masked_img = draw_results(image, detected_shapes, all_unique_contours, global_shape_colors, lock)
    
    return detected_shapes, outline_img, masked_img

def create_json_data(all_detected_shapes: List[List[Dict]], scale: float = 72 / 400):
    """Tạo dữ liệu JSON từ kết quả phát hiện shapes"""
    combined_data = []

    for detected_shapes in all_detected_shapes:
        for shape in detected_shapes:
            pipe_ids = shape.get('pipe_id', [])
            # Ép pipe_ids luôn là list
            if isinstance(pipe_ids, int):
                pipe_ids = [pipe_ids]
            
            for pid in pipe_ids:  # tách ra thành từng object riêng
                x, y, w, h = shape['bounding_box']
                shape_dict = {
                    "x1": float(round(x * scale, 2)),
                    "y1": float(round(y * scale, 2)),
                    "x2": float(round((x + w) * scale, 2)),
                    "y2": float(round((y + h) * scale, 2)),
                    "w": float(round(w * scale, 2)),
                    "h": float(round(h * scale, 2)),
                    "shape_name": str(shape['name']),
                    "area": float(shape['area']),
                    "centroid": [float(round(c * scale, 2)) for c in shape['centroid']],
                    "pipe_id": pid,  # mỗi object 1 pipe_id duy nhất
                    "vertices": [[float(round(p * scale, 2)) for p in pt] for pt in shape['vertices']],
                    "w_image": float(round(shape.get("w_image", 0) * scale, 2)),
                    "h_image": float(round(shape.get("h_image", 0) * scale, 2))
                }

                if shape.get("connected_to"):
                    shape_dict["connected_to"] = shape['connected_to']
                if shape.get("connection_lines"):
                    shape_dict["connection_lines"] = [
                        [[float(round(p * scale, 2)) for p in pt] for pt in line]
                        for line in shape['connection_lines']
                    ]
                
                combined_data.append(shape_dict)

    return combined_data

def process_multiple_images_with_connection_detection(
    input_sources: List[Union[str, np.ndarray]], **kwargs
) -> Tuple[List[Dict], List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Xử lý một danh sách ảnh, phát hiện kết nối và trả về kết quả.
    
    Returns:
        tuple: (json_data, result_images)
            - json_data: List các dictionary chứa thông tin shapes đã được xử lý
            - result_images: List các tuple (outline_image, masked_image)
    """
    # Dùng threading.Lock thay vì Manager (nhẹ hơn cho sequential processing)
    from threading import Lock
    global_shape_colors = {'_global_pipe_id': 1}
    lock = Lock()
    
    # Xử lý tuần tự thay vì dùng Pool
    all_detected_shapes = []
    result_images = []
    
    for i, source in enumerate(input_sources):
        shapes, outline_img, masked_img = process_one_image(
            source, i, global_shape_colors, lock,
            kwargs.get('min_area_object', 0.000001),
            kwargs.get('min_area_outline', 0.0005),
            kwargs.get('duplicate_tolerance', 10),
            kwargs.get('max_distance_ratio', 3.0),
            kwargs.get('small_threshold', 3),
            kwargs.get('min_dist_threshold', 20)
        )
        if shapes is not None:
            all_detected_shapes.append(shapes)
            result_images.append((outline_img, masked_img))
    
    # Tạo dữ liệu JSON
    json_data = []
    if all_detected_shapes:
        json_data = create_json_data(
            all_detected_shapes, 
            scale=kwargs.get('scale', 72 / 400)
        )
    
    return json_data, result_images

def save_results(json_data: List[Dict], result_images: List[Tuple[np.ndarray, np.ndarray]], 
                input_sources: List[Union[str, np.ndarray]], output_dir: str, 
                json_output_filename: str):
    """
    Lưu kết quả JSON và ảnh ra file.
    
    Args:
        json_data: Dữ liệu JSON để lưu
        result_images: List các tuple (outline_image, masked_image)
        input_sources: Danh sách nguồn ảnh đầu vào
        output_dir: Thư mục đích
        json_output_filename: Tên file JSON
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Lưu JSON
    json_path = os.path.join(output_dir, json_output_filename)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
    
    # Lưu ảnh
    for i, (outline_img, masked_img) in enumerate(result_images):
        if i < len(input_sources):
            source = input_sources[i]
            if isinstance(source, str):
                image_name = Path(source).stem
            else:
                image_name = f"numpy_array_{i}"
            
            outline_path = os.path.join(output_dir, f"{image_name}_outline_connected.png")
            masked_path = os.path.join(output_dir, f"{image_name}_masked_connected.png")
            
            cv2.imwrite(outline_path, outline_img)
            cv2.imwrite(masked_path, masked_img)

# ================================================================= #
# MAIN EXECUTION - Sửa đổi để sử dụng API mới
# ================================================================= #

if __name__ == "__main__":
    input_images = [
        "./quick_output/extracted_layers/INNO-ACMV-B.EA.png",
        "./quick_output/extracted_layers/INNO-ACMV-B.SA.png", 
        "./quick_output/extracted_layers/INNO-ACMV-PA.png",
    ]

    try:
        import time
        t = time.time()
        
        # Xử lý ảnh và lấy kết quả
        json_data, result_images = process_multiple_images_with_connection_detection(
            input_sources=input_images,
            min_area_object=0.000001,
            min_area_outline=0.0005,
            duplicate_tolerance=10,
            max_distance_ratio=2.5,
            scale=72 / 400,
            small_threshold=3,
            min_dist_threshold=20
        )
        
        # Lưu kết quả ra file (tùy chọn)
        save_results(
            json_data=json_data,
            result_images=result_images,
            input_sources=input_images,
            output_dir="output_centerline_connection_dynamic_threshold",
            json_output_filename="results_centerline_connection_dynamic_threshold.json"
        )
        
        print(f"Xử lý xong trong {time.time() - t:.2f} giây")
        print(f"Tìm thấy {len(json_data)} shapes")
        print(f"Tạo được {len(result_images)} cặp ảnh kết quả")
        
    except Exception as e:
        import traceback
        traceback.print_exc()