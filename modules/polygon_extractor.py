import cv2
import numpy as np
import math


class PolygonExtractor:
    """
    Module trích xuất polygon từ ảnh với thuật toán cải tiến
    - Phát hiện vùng khép kín 2 phase
    - Filter concave vertices
    - Content filtering
    - Dashed line detection
    """

    def __init__(self, 
                 angle_thresh=80.0,
                 white_pixel_threshold=230,
                 white_ratio_threshold=0.95,
                 dash_frac_thresh=0.35,
                 max_allowed_dashed_edges=2,
                 angle_tolerance_deg=15,
                 offset_px=3,
                 margin_threshold=0):
        """
        Args:
            angle_thresh: chỉ loại đỉnh lõm có góc < angle_thresh
            white_pixel_threshold: ngưỡng để coi là pixel trắng
            white_ratio_threshold: nếu >= threshold pixel trắng thì loại bỏ
            dash_frac_thresh: ngưỡng để coi là cạnh nét đứt
            max_allowed_dashed_edges: số cạnh nét đứt tối đa cho phép
            angle_tolerance_deg: tolerance góc cho horizontal/vertical lines
            offset_px: khoảng cách offset khi sample edge
            margin_threshold: khoảng cách tối thiểu từ tâm đến lề (pixel)
        """
        self.angle_thresh = angle_thresh
        self.white_pixel_threshold = white_pixel_threshold
        self.white_ratio_threshold = white_ratio_threshold
        self.dash_frac_thresh = dash_frac_thresh
        self.max_allowed_dashed_edges = max_allowed_dashed_edges
        self.angle_tolerance_deg = angle_tolerance_deg
        self.offset_px = offset_px
        self.margin_threshold = margin_threshold

    # ---- UTILITY FUNCTIONS ----
    def _signed_area(self, poly):
        """Tính diện tích có dấu của polygon"""
        a = 0.0
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i+1) % n]
            a += (x1*y2 - x2*y1)
        return a/2.0

    def _cross_z(self, p, q, r):
        """Tính cross product z-component"""
        return (q[0]-p[0])*(r[1]-q[1]) - (q[1]-p[1])*(r[0]-q[0])

    def _angle_deg(self, p, q, r):
        """Tính góc tại điểm q tạo bởi p-q-r"""
        v1 = (p[0]-q[0], p[1]-q[1])
        v2 = (r[0]-q[0], r[1]-q[1])
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        n1 = math.hypot(v1[0], v1[1])
        n2 = math.hypot(v2[0], v2[1])
        if n1*n2 == 0: 
            return 180.0
        ang = math.degrees(math.acos(max(-1.0, min(1.0, dot/(n1*n2)))))
        return ang

    def _filter_concave(self, verts, max_iters=50):
        """Filter các đỉnh lõm có góc nhỏ hơn angle_thresh"""
        poly = verts.copy()
        for it in range(max_iters):
            n = len(poly)
            if n < 4: 
                break
            area = self._signed_area(poly)
            orient = 1 if area > 0 else -1
            concave_info = []
            for i in range(n):
                p = poly[(i-1) % n]
                q = poly[i]
                r = poly[(i+1) % n]
                cz = self._cross_z(p, q, r)
                if cz*orient < 0:  
                    ang = self._angle_deg(p, q, r)
                    if ang < self.angle_thresh:
                        concave_info.append(i)
            if not concave_info: 
                break
            poly = [pt for i, pt in enumerate(poly) if i not in concave_info]
        return poly

    def _has_sufficient_content(self, cropped_img):
        """
        Kiểm tra xem ảnh crop có đủ nội dung hay không
        Trả về False nếu >= white_ratio_threshold pixel > white_pixel_threshold
        """
        if cropped_img is None or cropped_img.size == 0:
            return False
        
        # Chuyển sang grayscale để phân tích
        if len(cropped_img.shape) == 3:
            gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = cropped_img.copy()
        
        total_pixels = gray.shape[0] * gray.shape[1]
        
        # Đếm số pixel trắng
        white_pixels = np.sum(gray > self.white_pixel_threshold)
        white_ratio = white_pixels / total_pixels
        
        # Nếu >= threshold pixel trắng thì loại bỏ
        return white_ratio < self.white_ratio_threshold

    def _bbox_iou(self, boxA, boxB):
        """
        Tính tỷ lệ: diện tích giao / diện tích khối nhỏ hơn
        """
        # boxes in [x_min, y_min, x_max, y_max]
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        
        if interArea == 0:
            return 0.0
        
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        # Tỷ lệ trùng = diện tích giao / diện tích khối nhỏ
        overlap_ratio = interArea / boxAArea
        return overlap_ratio

    def _bridge_with_hough(self, binary_img, min_line_len=50, max_gap=20, draw_thickness=3):
        """Bridge gaps using Hough line detection"""
        edges = cv2.Canny(binary_img, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50,
                                minLineLength=min_line_len, maxLineGap=max_gap)
        out = binary_img.copy()
        if lines is not None:
            for l in lines:
                x1,y1,x2,y2 = l[0]
                cv2.line(out, (x1,y1), (x2,y2), 255, draw_thickness)
        return out

    def _sample_edge_with_offset(self, p1, p2, mask, num_samples=30):
        """Sample along edge but offset outward and inward"""
        x1,y1 = p1; x2,y2 = p2
        xs = np.linspace(x1, x2, num_samples)
        ys = np.linspace(y1, y2, num_samples)
        dx = x2 - x1; dy = y2 - y1
        length = math.hypot(dx, dy)
        if length == 0:
            return 0.0
        nx = -dy/length; ny = dx/length
        count_line = 0
        total = 0
        for xi, yi in zip(xs, ys):
            for side in (-1, 1):  # sample both sides
                sx = int(round(xi + side * self.offset_px * nx))
                sy = int(round(yi + side * self.offset_px * ny))
                sx = np.clip(sx, 0, mask.shape[1]-1)
                sy = np.clip(sy, 0, mask.shape[0]-1)
                total += 1
                if mask[sy, sx] != 0:
                    count_line += 1
        if total == 0:
            return 0.0
        return count_line / total

    # ---- MAIN EXTRACTION FUNCTION ----
    def extract_polygons(self, img_array, margin_threshold=None):
        """
        Extract các vùng polygon từ ảnh
        
        Args:
            img_array: ảnh đầu vào (BGR numpy array)
            margin_threshold: khoảng cách tối thiểu từ tâm đến lề (pixel)
                             Nếu None, sử dụng self.margin_threshold
            
        Returns:
            list[dict]: danh sách các region đã crop, mỗi dict chứa:
                - region_id: ID của region
                - crop_data: ảnh đã crop (numpy array)
                - bbox: [x, y, width, height]
                - center: [center_x, center_y]
                - area: diện tích
                - vertices: danh sách vertices đã filter
                - original_vertices: danh sách vertices gốc
                - source: nguồn phát hiện ("standard" hoặc "dashed_enhanced")
                - dashed_edges: số cạnh nét đứt (nếu có)
            list[dict]: skipped_margin_regions
        """
        skipped_margin_regions = []
        h, w = img_array.shape[:2]
        regions_data = []
        region_idx = 0
        existing_bboxes = []
        
        # Sử dụng margin_threshold từ parameter hoặc từ instance
        effective_margin = margin_threshold if margin_threshold is not None else self.margin_threshold
        
        # ---- PHASE 1: Standard polygon detection ----
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        gray = np.where(gray > 230, 255, gray).astype(np.uint8)
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 25, 10)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)

        # Morphological operations để extract horizontal và vertical lines
        hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w//60), 1))
        ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, h//60)))
        horiz = cv2.morphologyEx(th, cv2.MORPH_OPEN, hor_kernel, iterations=1)
        vert = cv2.morphologyEx(th, cv2.MORPH_OPEN, ver_kernel, iterations=1)
        combined = cv2.bitwise_or(horiz, vert)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
        combined = cv2.dilate(combined, np.ones((3,3),np.uint8), iterations=1)

        # Flood fill để tìm vùng khép kín
        mask_ff = np.zeros((h+2, w+2), np.uint8)
        free = cv2.bitwise_not(combined)
        flood_img = free.copy()
        cv2.floodFill(flood_img, mask_ff, (0,0), 128)
        outside = (flood_img == 128).astype(np.uint8) * 255
        enclosed = cv2.bitwise_and(free, cv2.bitwise_not(outside))
        enclosed = cv2.morphologyEx(enclosed, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
        enclosed = cv2.morphologyEx(enclosed, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=1)
        cnts, _ = cv2.findContours(enclosed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # ---- Process contours PHASE 1 ----
        min_parent_area = max(80, int((w*h)*0.00005))
        statistics = {
            "phase1_total": len(cnts),
            "phase1_cropped": 0,
            "phase1_skipped_area": 0,
            "phase1_skipped_margin": 0,
            "phase1_skipped_content": 0
        }

        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < min_parent_area:
                statistics["phase1_skipped_area"] += 1
                continue
            
            # Polygon approximation và filter concave vertices
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.005*peri, True)
            verts = [(int(p[0][0]), int(p[0][1])) for p in approx]
            filtered_vertices = self._filter_concave(verts)

            # Tính bounding box và center từ filtered vertices
            if len(filtered_vertices) < 3:
                continue
                
            xs = [p[0] for p in filtered_vertices]
            ys = [p[1] for p in filtered_vertices]
            x_min, x_max = max(0, min(xs)), min(w, max(xs))
            y_min, y_max = max(0, min(ys)), min(h, max(ys))
            
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            
            # Kiểm tra margin threshold
            if (center_x < effective_margin or 
                center_y < effective_margin or 
                center_x > (w - effective_margin) or 
                center_y > (h - effective_margin)):
                statistics["phase1_skipped_margin"] += 1
                continue

            # Crop bounding box và kiểm tra content
            crop = img_array[y_min:y_max, x_min:x_max].copy()
            if not self._has_sufficient_content(crop):
                statistics["phase1_skipped_content"] += 1
                continue

            region_idx += 1
            statistics["phase1_cropped"] += 1
            
            bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
            existing_bboxes.append(bbox)
            
            regions_data.append({
                "region_id": region_idx,
                "bbox": [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)],
                "center": [int(center_x), int(center_y)],
                "area": int(area),
                "vertices": filtered_vertices,
                "original_vertices": verts,
                "crop_data": crop,
                "source": "standard"
            })

        # ---- PHASE 2: Detect dashed line polygons ----
        # Setup for dashed line detection
        hor_kernel_w = max(15, w//60); ver_kernel_h = max(15, h//60)
        kernel_length_h = max(15, w//60); kernel_length_v = max(15, h//60)
        hough_min_line_len = max(30, w//80); hough_max_gap = 25; hough_draw_thickness = 1

        # Recompute masks for bridging
        horiz_closed = cv2.morphologyEx(horiz, cv2.MORPH_CLOSE, 
                                       cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length_h, 1)), iterations=1)
        vert_closed = cv2.morphologyEx(vert, cv2.MORPH_CLOSE,
                                      cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length_v)), iterations=1)
        combined_pre = cv2.bitwise_or(horiz_closed, vert_closed)
        combined_pre = cv2.morphologyEx(combined_pre, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
        combined_pre = cv2.dilate(combined_pre, np.ones((3,3),np.uint8), iterations=1)

        # Bridge with Hough lines
        combined_bridge = self._bridge_with_hough(combined_pre, min_line_len=hough_min_line_len, 
                                                 max_gap=hough_max_gap, draw_thickness=hough_draw_thickness)
        combined2 = combined_bridge.copy()
        combined2 = cv2.morphologyEx(combined2, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
        combined2 = cv2.morphologyEx(combined2, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=1)

        # Flood fill enclosed on bridged mask
        mask_ff2 = np.zeros((h+2, w+2), np.uint8)
        free2 = cv2.bitwise_not(combined2)
        flood_img2 = free2.copy()
        cv2.floodFill(flood_img2, mask_ff2, (0,0), 128)
        outside2 = (flood_img2 == 128).astype(np.uint8) * 255
        enclosed2 = cv2.bitwise_and(free2, cv2.bitwise_not(outside2))
        enclosed2 = cv2.morphologyEx(enclosed2, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
        enclosed2 = cv2.morphologyEx(enclosed2, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=1)

        cnts2, _ = cv2.findContours(enclosed2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Process PHASE 2 contours
        statistics.update({
            "phase2_total": len(cnts2),
            "phase2_cropped": 0,
            "phase2_skipped_area": 0,
            "phase2_skipped_dashed": 0,
            "phase2_skipped_overlap": 0,
            "phase2_skipped_margin": 0,
            "phase2_skipped_content": 0
        })

        for cnt2 in cnts2:
            area = cv2.contourArea(cnt2)
            if area < min_parent_area:
                statistics["phase2_skipped_area"] += 1
                continue
            
            peri = cv2.arcLength(cnt2, True)
            approx = cv2.approxPolyDP(cnt2, 0.005*peri, True)
            verts = [(int(p[0][0]), int(p[0][1])) for p in approx]
            if len(verts) < 3:
                continue

            # Filter concave vertices
            filtered_vertices = self._filter_concave(verts)

            # Analyze dashed edges
            dashed_count = 0
            sample_min_pts = 30
            for j in range(len(filtered_vertices)):
                p1 = filtered_vertices[j]
                p2 = filtered_vertices[(j+1) % len(filtered_vertices)]
                ang = abs(math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0]))) % 180
                is_oriented = (abs(ang-0) <= self.angle_tolerance_deg or 
                              abs(ang-180) <= self.angle_tolerance_deg or 
                              abs(ang-90) <= self.angle_tolerance_deg)
                if not is_oriented:
                    continue
                dist = int(max(sample_min_pts, math.hypot(p2[0]-p1[0], p2[1]-p1[1])//2))
                frac = self._sample_edge_with_offset(p1, p2, combined_pre, num_samples=dist)
                if frac < self.dash_frac_thresh:
                    dashed_count += 1

            if dashed_count > self.max_allowed_dashed_edges:
                statistics["phase2_skipped_dashed"] += 1
                continue

            # Check bounding box
            xs = [p[0] for p in filtered_vertices]
            ys = [p[1] for p in filtered_vertices]
            x_min, x_max = max(0, min(xs)), min(w, max(xs))
            y_min, y_max = max(0, min(ys)), min(h, max(ys))
            candidate_bbox = [x_min, y_min, x_max, y_max]

            # Check overlap with existing regions
            overlap = False
            for existing_bbox in existing_bboxes:
                if self._bbox_iou(candidate_bbox, existing_bbox) > 0.5:
                    overlap = True
                    break
            if overlap:
                statistics["phase2_skipped_overlap"] += 1
                continue

            # Check margin
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            if (center_x < effective_margin or 
                center_y < effective_margin or 
                center_x > (w - effective_margin) or 
                center_y > (h - effective_margin)):
                statistics["phase2_skipped_margin"] += 1
                skipped_margin_regions.append({
                    "region_id": f"skip_margin_{len(skipped_margin_regions)+1}",
                    "bbox": [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)],
                    "center": [int(center_x), int(center_y)],
                    "area": int(area),
                    "vertices": filtered_vertices,
                    "source": "skipped_margin_phase2",
                    "dashed_edges": dashed_count
                })
                continue

            # Crop bounding box và kiểm tra content
            crop = img_array[y_min:y_max, x_min:x_max].copy()
            if not self._has_sufficient_content(crop):
                statistics["phase2_skipped_content"] += 1
                continue

            region_idx += 1
            statistics["phase2_cropped"] += 1
            existing_bboxes.append(candidate_bbox)
            
            regions_data.append({
                "region_id": region_idx,
                "bbox": [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)],
                "center": [int(center_x), int(center_y)],
                "area": int(area),
                "vertices": filtered_vertices,
                "original_vertices": verts,
                "crop_data": crop,
                "source": "dashed_enhanced",
                "dashed_edges": dashed_count
            })

        # Add statistics to regions data
        for region in regions_data:
            region["statistics"] = statistics

        return regions_data, skipped_margin_regions

if __name__ == "__main__":
    import os
    import json
    import cv2

    # === CONFIG ===
    src_img = "./anh_test/trai.jpg"  # ảnh input
    outdir = "./output" 
    crop_dir = "./output/cropped"  # thư mục chứa các các ảnh đã crop
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(crop_dir, exist_ok=True)

    # Load image
    img = cv2.imread(src_img)
    if img is None:
        raise FileNotFoundError(f"Image not found at {src_img}")

    # Initialize extractor with default parameters
    extractor = PolygonExtractor(
        angle_thresh=80.0,
        white_pixel_threshold=230,
        white_ratio_threshold=0.95,
        dash_frac_thresh=0.35,
        max_allowed_dashed_edges=2,
        angle_tolerance_deg=15,
        offset_px=3,
        margin_threshold=0
    )

    # Extract polygons
    results, skipped = extractor.extract_polygons(img)

    # Prepare visualization
    h, w = img.shape[:2]
    annot = img.copy()
    overlay = img.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    # Process results and save crops
    parents = []
    standard_count = 0
    dashed_count = 0

    for idx, result in enumerate(results, start=1):
        # Save cropped image
        crop_filename = f"polygon_{idx:03d}.png"
        cv2.imwrite(os.path.join(crop_dir, crop_filename), result['crop_data'])

        # Visualization
        color = colors[(idx-1) % len(colors)]
        pts = np.array(result['vertices'], np.int32)
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(annot, [pts], True, color, 2)
        
        # Add label
        center_x, center_y = result['center']
        cv2.putText(annot, str(idx), (center_x-15, center_y+6), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 4, cv2.LINE_AA)
        cv2.putText(annot, str(idx), (center_x-15, center_y+6), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        # Count by source
        if result['source'] == 'standard':
            standard_count += 1
        else:
            dashed_count += 1

        # Store parent info
        parents.append({
            "parent_id": idx,
            "area": result['area'],
            "vertices": result['vertices'],
            "crop_info": {
                "bbox": result['bbox'],
                "center": result['center'],
                "size": [result['bbox'][2], result['bbox'][3]],
            },
            "crop_filename": crop_filename,
            "source": result['source'],
            "dashed_edges": result.get('dashed_edges', 0)
        })

    # Save visualization
    blended = cv2.addWeighted(overlay, 0.3, annot, 0.7, 0)
    cv2.imwrite(os.path.join(outdir, "annotated_polygons_extended.png"), blended)

    # Save results to JSON
    result_data = {
        "image": os.path.basename(src_img),
        "angle_thresh": extractor.angle_thresh,
        "margin_threshold": extractor.margin_threshold,
        "content_filter_params": {
            "white_pixel_threshold": extractor.white_pixel_threshold,
            "white_ratio_threshold": extractor.white_ratio_threshold
        },
        "standard_polygons": standard_count,
        "dashed_enhanced_polygons": dashed_count,
        "crop_directory": os.path.basename(crop_dir),
        "parents": parents
    }

    out_json = os.path.join(outdir, "parents_vertices_filtered_extended.json")
    with open(out_json, "w", encoding='utf-8') as f:
        json.dump(result_data, f, indent=2)

    print(f"Total polygons processed: {len(results)}")
    print(f"Standard polygons: {standard_count}")
    print(f"Dashed-enhanced polygons: {dashed_count}")
    print(f"Results saved to {outdir}")