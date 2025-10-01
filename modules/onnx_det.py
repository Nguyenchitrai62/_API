import cv2
import numpy as np
import onnxruntime as ort
import os
from typing import Union, Tuple, List
from shapely.geometry import Polygon
import pyclipper

# Cấu hình mặc định
DEFAULT_CONFIG = {
    'resize_long': 960,
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    'scale': 1.0 / 255.0,
    'thresh': 0.1,
    'box_thresh': 0.6,
    'max_candidates': 1000,
    'unclip_ratio': 1.5
}

def _load_image(image_input: Union[str, np.ndarray]) -> np.ndarray:
    """
    Load ảnh từ path hoặc nhận numpy array
    
    Args:
        image_input: có thể là string (path) hoặc numpy array
        
    Returns:
        numpy array: ảnh đã load
    """
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image file not found: {image_input}")
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"Cannot load image from: {image_input}")
        return img
    elif isinstance(image_input, np.ndarray):
        return image_input.copy()
    else:
        raise TypeError("Image input must be either string (path) or numpy array")

def _resize_normalize(img: np.ndarray, config: dict) -> Tuple[np.ndarray, float, float]:
    """Resize và normalize ảnh theo đúng logic PaddleOCR"""
    h, w = img.shape[:2]
    
    # Logic resize đúng theo PaddleOCR DetResizeForTest
    if max(h, w) > config['resize_long']:
        if h > w:
            ratio = float(config['resize_long']) / h
        else:
            ratio = float(config['resize_long']) / w
    else:
        ratio = 1.0
    
    # Tính kích thước mới
    resize_h = int(h * ratio)
    resize_w = int(w * ratio)
    
    # Đảm bảo chia hết cho 32
    resize_h = max(32, (resize_h + 31) // 32 * 32)
    resize_w = max(32, (resize_w + 31) // 32 * 32)
    
    # Resize ảnh
    img_resized = cv2.resize(img, (resize_w, resize_h))
    
    # Normalize - đảm bảo tất cả operations đều là float32
    mean = np.array(config['mean'], dtype=np.float32)
    std = np.array(config['std'], dtype=np.float32)
    scale = np.float32(config['scale'])
    
    img_normalized = img_resized.astype(np.float32) * scale
    img_normalized = (img_normalized - mean) / std
    
    # Transpose từ HWC sang CHW
    img_transposed = np.transpose(img_normalized, (2, 0, 1))
    
    # Thêm batch dimension và đảm bảo dtype là float32
    img_batch = np.expand_dims(img_transposed, axis=0).astype(np.float32)
    
    # Tính tỷ lệ scale để map lại tọa độ
    ratio_h = h / resize_h
    ratio_w = w / resize_w
    
    return img_batch, ratio_h, ratio_w

def _get_mini_boxes(contour):
    """Tìm bounding box nhỏ nhất"""
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
    
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2
    
    box = [points[index_1], points[index_2], points[index_3], points[index_4]]
    return box, min(bounding_box[1])

def _unclip(box, unclip_ratio: float):
    """Mở rộng box - cải thiện xử lý lỗi"""
    try:
        poly = Polygon(box)
        if not poly.is_valid or poly.area <= 0:
            return []
            
        distance = poly.area * unclip_ratio / poly.length if poly.length > 0 else 0
        offset = pyclipper.PyclipperOffset()
        
        # Chuyển đổi tọa độ sang int để tránh lỗi pyclipper
        box_int = np.array(box, dtype=np.int64).tolist()
        offset.AddPath(box_int, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = offset.Execute(distance)
        return expanded
    except Exception as e:
        print(f"Error in unclip: {e}")
        return []

def _get_contours(pred: np.ndarray, ratio_h: float, ratio_w: float, config: dict) -> Tuple[List, List]:
    """Lấy contours từ prediction map - cải thiện logic"""
    pred = pred[0, 0, :, :]  # Lấy channel đầu tiên
    
    # Threshold để tạo binary mask
    bitmap = pred > config['thresh']
    bitmap = bitmap.astype(np.uint8) * 255
    
    # Tìm contours
    contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    scores = []
    
    for contour in contours[:config['max_candidates']]:
        # Kiểm tra kích thước contour
        if cv2.contourArea(contour) < 9:  # Tăng threshold diện tích tối thiểu
            continue
            
        # Tính điểm trung bình trong contour
        mask = np.zeros_like(bitmap, dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 255)
        score = cv2.mean(pred, mask=mask)[0]
        
        if score < config['box_thresh']:
            continue
        
        # Approximate contour để giảm số điểm
        epsilon = 0.002 * cv2.arcLength(contour, True)  # Giảm epsilon
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Lấy mini box
        box, min_side = _get_mini_boxes(contour)
        
        if min_side < 3:  # Giảm threshold kích thước tối thiểu
            continue
        
        # Unclip box
        box = np.array(box, dtype=np.float32)
        expanded = _unclip(box.reshape(-1, 2), config['unclip_ratio'])
        
        if len(expanded) == 0:
            continue
        
        expanded_box = np.array(expanded[0], dtype=np.float32).reshape(-1, 2)
        
        if len(expanded_box) < 4:
            continue
        
        # Lấy lại mini box từ expanded
        try:
            expanded_contour = expanded_box.astype(np.int32)
            final_box, final_min_side = _get_mini_boxes(expanded_contour.reshape(-1, 1, 2))
            
            if final_min_side < 3:
                continue
                
            final_box = np.array(final_box, dtype=np.float32)
        except:
            continue
        
        # Scale lại tọa độ về ảnh gốc
        final_box[:, 0] = final_box[:, 0] * ratio_w
        final_box[:, 1] = final_box[:, 1] * ratio_h
        
        # Clip về kích thước ảnh gốc
        h_orig = pred.shape[0] * ratio_h
        w_orig = pred.shape[1] * ratio_w
        final_box[:, 0] = np.clip(final_box[:, 0], 0, w_orig)
        final_box[:, 1] = np.clip(final_box[:, 1], 0, h_orig)
        
        boxes.append(final_box.astype(np.int32))
        scores.append(score)
    
    return boxes, scores

def detect_text(image: Union[str, np.ndarray],
               model_path: str,
               thresh: float = None,
               box_thresh: float = None,
               resize_long: int = None,
               unclip_ratio: float = None,
               max_candidates: int = None) -> Tuple[List[np.ndarray], List[float]]:
    """
    Detect text regions trong ảnh sử dụng ONNX model
    
    Args:
        image: đường dẫn ảnh (string) hoặc ảnh numpy array
        model_path: đường dẫn đến ONNX model
        thresh: threshold cho binary map (default: 0.3)
        box_thresh: threshold cho text box score (default: 0.6)
        resize_long: kích thước resize tối đa (default: 960)
        unclip_ratio: tỷ lệ mở rộng box (default: 1.5)
        max_candidates: số lượng candidate tối đa (default: 1000)
        
    Returns:
        tuple: (boxes, scores) - danh sách bounding boxes và scores
        
    Example:
        >>> # Sử dụng với đường dẫn ảnh
        >>> boxes, scores = detect_text('./image.png', './model.onnx')
        
        >>> # Sử dụng với numpy array và custom threshold
        >>> import cv2
        >>> img = cv2.imread('./image.png')
        >>> boxes, scores = detect_text(img, './model.onnx', thresh=0.2, box_thresh=0.4)
    """
    # Kiểm tra model file
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load ONNX model
    try:
        session = ort.InferenceSession("./onnx_det_output/inference.onnx")

        print("Providers khả dụng:", ort.get_available_providers())
        print("ONNX Runtime đang dùng:", session.get_providers())
    except Exception as e:
        raise RuntimeError(f"Cannot load ONNX model: {e}")
    
    # Lấy thông tin về input/output
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    
    # Tạo config với override parameters
    config = DEFAULT_CONFIG.copy()
    if thresh is not None:
        config['thresh'] = thresh
    if box_thresh is not None:
        config['box_thresh'] = box_thresh
    if resize_long is not None:
        config['resize_long'] = resize_long
    if unclip_ratio is not None:
        config['unclip_ratio'] = unclip_ratio
    if max_candidates is not None:
        config['max_candidates'] = max_candidates
    
    # Load và preprocess ảnh
    img = _load_image(image)
    input_tensor, ratio_h, ratio_w = _resize_normalize(img, config)
    
    # Inference
    try:
        outputs = session.run(output_names, {input_name: input_tensor})
    except Exception as e:
        raise RuntimeError(f"Model inference failed: {e}")
    
    # Postprocess
    pred = outputs[0]  # Prediction map
    boxes, scores = _get_contours(pred, ratio_h, ratio_w, config)
    
    return boxes, scores

from PIL import Image, ImageDraw, ImageFont

def draw_detection_result(
        img_path: str,
        boxes: List[np.ndarray],
        scores: List[float],  # Changed from texts to scores
        output_path: str = "result.jpg",
        font_path: str = "C:/Windows/Fonts/times.ttf",  # Added .ttf extension
        font_size: int = 24,
        box_color: Tuple[int,int,int]=(0,255,0),
        text_color: Tuple[int,int,int]=(255,0,0),
        thickness: int = 2
    ):
    """
    Vẽ boxes và scores lên ảnh sử dụng font TTF.
    """
    # Load ảnh
    img_cv = cv2.imread(img_path)
    if img_cv is None:
        raise ValueError(f"Cannot load image from: {img_path}")
    
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        print(f"Warning: Cannot load font from {font_path}, using default font")
        font = ImageFont.load_default()

    for box, score in zip(boxes, scores):
        # Vẽ polygon
        polygon = [(int(x), int(y)) for x,y in box]
        draw.line(polygon + [polygon[0]], fill=box_color, width=thickness)

        # Vẽ score
        x_min, y_min = int(np.min(box[:,0])), int(np.min(box[:,1]))
        text = f"{score:.2f}"
        draw.text((x_min, max(y_min - font_size, 0)), text, font=font, fill=text_color)

    # Chuyển về OpenCV và lưu
    img_out = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, img_out)
    print(f"Đã lưu ảnh kết quả: {output_path}")

# Để maintain backward compatibility
class ONNXTextDetector:
    """
    Wrapper class để maintain backward compatibility
    Khuyến khích sử dụng function detect_text() thay vì class này
    """
    def __init__(self, model_path, config=None):
        self.model_path = model_path
        self.config = config or DEFAULT_CONFIG.copy()
        
    def detect(self, img, thresh=None, box_thresh=None):
        return detect_text(
            image=img,
            model_path=self.model_path,
            thresh=thresh or self.config.get('thresh'),
            box_thresh=box_thresh or self.config.get('box_thresh'),
            resize_long=self.config.get('resize_long'),
            unclip_ratio=self.config.get('unclip_ratio'),
            max_candidates=self.config.get('max_candidates')
        )

if __name__ == "__main__":
    try:
        import time
        t = time.time()
        # Test với function (recommended)
        boxes, scores = detect_text(
            image='./anh_test/cell.jpg',
            model_path='./onnx_det_output/inference.onnx',
            thresh=0.3,
            box_thresh=0.6
        )
        
        print(f"Detected {len(boxes)} text regions")
        
        # Vẽ và lưu kết quả - fixed parameters
        draw_detection_result(
            img_path='./anh_test/cell.jpg',  # Changed from image to img_path
            boxes=boxes,
            scores=scores,  # Changed from texts to scores
            output_path='_detection_result.jpg'
        )
        print(f"Total time: {time.time() - t:.2f}s")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure image and model files exist!")