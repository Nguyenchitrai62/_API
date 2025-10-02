from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import cv2
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from typing import List, Tuple, Union
from shapely.geometry import Polygon
import pyclipper
import base64
from openai import OpenAI
from PIL import Image
import io

# Import các hàm từ file module tương ứng
from modules.onnx_det import _resize_normalize, _get_contours

router = APIRouter()

def convert_numpy_types(obj):
    """Convert numpy data types to native Python types recursively"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

# === CONFIGURATION ===
DEFAULT_CONFIG = {
    'resize_long': 960, 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],
    'scale': 1.0 / 255.0, 'thresh': 0.1, 'box_thresh': 0.6,
    'max_candidates': 1000, 'unclip_ratio': 1.5
}
MARGIN_THRESHOLD = 0
# LƯU Ý: Thay thế bằng khóa API của bạn hoặc sử dụng biến môi trường
client = OpenAI(api_key="sk-")
INFERENCE_CHARACTER_DICT = [
    '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '}', '~', '°', '±', '²', '³', 'À', 'Á', 'Â', 'Ã', 'È', 'É', 'Ê', 'Ì', 'Í', 'Ò', 'Ó', 'Ô', 'Õ', 'Ö', '×', 'Ø', 'Ù', 'Ú', 'Ü', 'Ý', 'à', 'á', 'â', 'ã', 'è', 'é', 'ê', 'ì', 'í', 'ò', 'ó', 'ô', 'õ', 'ö', '÷', 'ù', 'ú', 'ü', 'ý', 'Ă', 'ă', 'Đ', 'đ', 'Ĩ', 'ĩ', 'Ũ', 'ũ', 'Ơ', 'ơ', 'Ư', 'ư', 'Ω', 'λ', 'μ', 'ρ', 'σ', 'τ', 'φ', 'ψ', 'ω', 'Ạ', 'ạ', 'Ả', 'ả', 'Ấ', 'ấ', 'Ầ', 'ầ', 'Ẩ', 'ẩ', 'Ẫ', 'ẫ', 'Ậ', 'ậ', 'Ắ', 'ắ', 'Ằ', 'ằ', 'Ẳ', 'ẳ', 'Ẵ', 'ẵ', 'Ặ', 'ặ', 'Ẹ', 'ẹ', 'Ẻ', 'ẻ', 'Ẽ', 'ẽ', 'Ế', 'ế', 'Ề', 'ề', 'Ể', 'ể', 'Ễ', 'ễ', 'Ệ', 'ệ', 'Ỉ', 'ỉ', 'Ị', 'ị', 'Ọ', 'ọ', 'Ỏ', 'ỏ', 'Ố', 'ố', 'Ồ', 'ồ', 'Ổ', 'ổ', 'Ỗ', 'ỗ', 'Ộ', 'ộ', 'Ớ', 'ớ', 'Ờ', 'ờ', 'Ở', 'ở', 'Ỡ', 'ỡ', 'Ợ', 'ợ', 'Ụ', 'ụ', 'Ủ', 'ủ', 'Ứ', 'ứ', 'Ừ', 'ừ', 'Ử', 'ử', 'Ữ', 'ữ', 'Ự', 'ự', 'Ỳ', 'ỳ', 'Ỵ', 'ỵ', 'Ỷ', 'ỷ', 'Ỹ', 'ỹ', '–', '…', '‰', '∆', '∞', '≈', '≠', '≤', '≥',' '
]

class ImageRequest(BaseModel):
    image_base64: str

def base64_to_cv2_image(base64_string: str) -> np.ndarray:
    try:
        img_data = base64.b64decode(base64_string)
        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None: raise ValueError("Could not decode image.")
        return img
    except Exception as e:
        raise ValueError(f"Invalid base64 string: {e}")

def _preprocess_image_dynamic(crop_img: np.ndarray, target_height: int = 48) -> np.ndarray:
    h, w = crop_img.shape[:2]
    ratio = target_height / h
    optimal_width = int(w * ratio)
    target_width = max(160, min(optimal_width, 3200))
    
    new_w, new_h = int(w * ratio), target_height
    if new_w > target_width:
        ratio = target_width / w
        new_w, new_h = target_width, int(h * ratio)
    
    crop_resized = cv2.resize(crop_img, (new_w, new_h))
    img_padded = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255
    img_padded[:new_h, :new_w] = crop_resized
    img_norm = (img_padded.astype(np.float32) / 255.0 - 0.5) / 0.5
    return np.expand_dims(np.transpose(img_norm, (2, 0, 1)), axis=0).astype(np.float32)

def detect_text_triton(image: np.ndarray, det_model_name: str, triton_url: str, **kwargs) -> Tuple[List[np.ndarray], List[float]]:
    config = {**DEFAULT_CONFIG, **kwargs}
    input_tensor, ratio_h, ratio_w = _resize_normalize(image, config)
    client = httpclient.InferenceServerClient(url=triton_url)
    inputs = httpclient.InferInput("x", input_tensor.shape, "FP32")
    inputs.set_data_from_numpy(input_tensor)
    outputs = httpclient.InferRequestedOutput("fetch_name_0")
    try:
        response = client.infer(model_name=det_model_name, inputs=[inputs], outputs=[outputs])
        pred = response.as_numpy("fetch_name_0")
    except InferenceServerException as e:
        raise RuntimeError(f"Triton inference failed: {e}")
    return _get_contours(pred, ratio_h, ratio_w, config)

def recognize_text_triton(crop_img: np.ndarray, rec_model_name: str, triton_url: str) -> str:
    input_tensor = _preprocess_image_dynamic(crop_img)
    client = httpclient.InferenceServerClient(url=triton_url)
    inputs = httpclient.InferInput("x", input_tensor.shape, "FP32")
    inputs.set_data_from_numpy(input_tensor)
    outputs = httpclient.InferRequestedOutput("fetch_name_0")
    try:
        response = client.infer(model_name=rec_model_name, inputs=[inputs], outputs=[outputs])
        pred = response.as_numpy("fetch_name_0")[0]
    except InferenceServerException as e:
        raise RuntimeError(f"Triton rec inference failed: {e}")

    decoded_text, prev_idx = "", -1
    for idx in np.argmax(pred, axis=1):
        if idx != 0 and idx != prev_idx:
            char_idx = idx - 2
            if 0 <= char_idx < len(INFERENCE_CHARACTER_DICT):
                decoded_text += INFERENCE_CHARACTER_DICT[char_idx]
        prev_idx = idx
    return decoded_text.strip()

def extract_enclosed_shapes(img_color, text_boxes=None):
    h, w = img_color.shape[:2]
    text_mask = np.zeros((h, w), np.uint8)
    if text_boxes is not None:
        for box in text_boxes:
            x_min, y_min = int(np.min(box[:,0])), int(np.min(box[:,1]))
            x_max, y_max = int(np.max(box[:,0])), int(np.max(box[:,1]))
            padding = 5
            cv2.rectangle(text_mask, (max(0,x_min-padding), max(0,y_min-padding)), (min(w,x_max+padding), min(h,y_max+padding)), 255, -1)
    
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    gray = np.where(gray > 230, 255, gray).astype(np.uint8)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    th = cv2.bitwise_and(th, cv2.bitwise_not(text_mask))
    
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w//60), 1))
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, h//60)))
    horizontals = cv2.morphologyEx(th, cv2.MORPH_OPEN, hor_kernel)
    verticals = cv2.morphologyEx(th, cv2.MORPH_OPEN, ver_kernel)
    combined = cv2.morphologyEx(cv2.bitwise_or(horizontals, verticals), cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
    combined = cv2.dilate(combined, np.ones((max(3, int(min(h,w)/500)),)*2, np.uint8))
    
    mask = np.zeros((h + 2, w + 2), np.uint8)
    flood_img = cv2.bitwise_not(combined).copy()
    cv2.floodFill(flood_img, mask, (0,0), 128)
    enclosed = cv2.bitwise_and(cv2.bitwise_not(combined), cv2.bitwise_not((flood_img==128).astype(np.uint8)*255))
    enclosed = cv2.morphologyEx(enclosed, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    enclosed = cv2.morphologyEx(enclosed, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    
    cnts, _ = cv2.findContours(enclosed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shape_crops = []
    min_parent_area = max(80, int((w*h) * 0.00005))
    for cnt in cnts:
        if cv2.contourArea(cnt) < min_parent_area: continue
        poly_mask = np.zeros((h,w), np.uint8)
        cv2.drawContours(poly_mask, [cnt], -1, 255, -1)
        inner_mask = cv2.erode(poly_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1)))
        inner_mask = cv2.morphologyEx(inner_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        
        _, labels, stats, _ = cv2.connectedComponentsWithStats(inner_mask, connectivity=8)
        for lab in range(1, len(stats)):
            x, y, wb, hb, a = stats[lab, cv2.CC_STAT_LEFT], stats[lab, cv2.CC_STAT_TOP], stats[lab, cv2.CC_STAT_WIDTH], stats[lab, cv2.CC_STAT_HEIGHT], stats[lab, cv2.CC_STAT_AREA]
            if a < max(50, int((w*h) * 0.00001)): continue
            cx, cy = x + wb//2, y + hb//2
            if (cx < MARGIN_THRESHOLD or cy < MARGIN_THRESHOLD or cx > (w-MARGIN_THRESHOLD) or cy > (h-MARGIN_THRESHOLD)): continue
            shape_crops.append({'crop': img_color[y:y+hb, x:x+wb], 'bbox': [x, y, wb, hb], 'center': [cx, cy], 'area': a})
    return shape_crops

def find_corresponding_text(shape_bbox, ocr_results, tolerance=20):
    x, y, w, h = shape_bbox
    shape_y_min, shape_y_max = y, y + h
    corresponding_texts = []
    for ocr in ocr_results:
        text_y, text_h = ocr['bbox'][1], ocr['bbox'][3]
        text_center_y = text_y + text_h // 2
        if (shape_y_min - tolerance <= text_center_y <= shape_y_max + tolerance):
            corresponding_texts.append(ocr['text'].strip())
    return ' '.join(corresponding_texts) if corresponding_texts else ""

def encode_image_to_base64(image_array):
    if isinstance(image_array, np.ndarray):
        img_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB) if len(image_array.shape)==3 else image_array
        pil_img = Image.fromarray(img_rgb)
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    return None

def is_image_mostly_white(image_array, threshold=200):
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY) if len(image_array.shape) == 3 else image_array
    return np.all(gray > threshold)

def analyze_shapes_with_ai(shape_crops):
    if not shape_crops: return []
    prompt_header = "Role: You are an expert in analyzing visual patterns in architectural and engineering drawings. Job Description: I will provide images of construction drawings. For each image, output a structured description in a single paragraph that always follows this fixed order: Geometric features, Line type and style, Semantic meaning, Functional role. Note: Always output as one coherent paragraph. No natural storytelling. Must explicitly state that the elements belong to an architectural or engineering drawing. Text must be unambiguous."
    descriptions, api_inputs = [None] * len(shape_crops), []
    for i, shape_info in enumerate(shape_crops):
        if is_image_mostly_white(shape_info['crop']):
            descriptions[i] = "Empty white region, blank background with no lines or patterns."
        else:
            image_b64 = encode_image_to_base64(shape_info['crop'])
            if image_b64: api_inputs.append((i, image_b64))

    if api_inputs:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt_header}]}]
        for _, image_b64 in api_inputs:
            messages[0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}})
        try:
            response = client.chat.completions.create(model="gpt-4o", messages=messages, temperature=0, top_p=1)
            api_descriptions = response.choices[0].message.content.strip().split("\n\n")
            for api_idx, (original_idx, _) in enumerate(api_inputs):
                descriptions[original_idx] = api_descriptions[api_idx].strip() if api_idx < len(api_descriptions) else "Analysis unavailable."
        except Exception as e:
            print(f"API call failed: {e}")
            for original_idx, _ in api_inputs:
                if descriptions[original_idx] is None: descriptions[original_idx] = "Analysis failed due to API error."
    for i in range(len(descriptions)):
        if descriptions[i] is None: descriptions[i] = "Analysis unavailable."
    return descriptions

@router.post("/gen_description")
async def generate_description(request: ImageRequest):
    try:
        img = base64_to_cv2_image(request.image_base64)
        det_model, rec_model, triton_url = "det_text", "rec_text", "10.0.10.62:8010"
        boxes, scores = detect_text_triton(img, det_model, triton_url)
        
        ocr_results = []
        for i, box in enumerate(boxes):
            x_min, y_min = int(np.min(box[:,0])), int(np.min(box[:,1]))
            x_max, y_max = int(np.max(box[:,0])), int(np.max(box[:,1]))
            text = recognize_text_triton(img[y_min:y_max, x_min:x_max], rec_model, triton_url)
            ocr_results.append({
                'id': f'text_{i+1}', 
                'bbox': [x_min, y_min, x_max-x_min, y_max-y_min], 
                'text': text, 
                'confidence': float(scores[i]) if i<len(scores) else 0.0
            })
        
        shape_crops = extract_enclosed_shapes(img, boxes)
        shape_descriptions = analyze_shapes_with_ai(shape_crops)
        
        shape_results = []
        for i, (shape_info, ai_desc) in enumerate(zip(shape_crops, shape_descriptions)):
            shape_results.append({
                'id': f'shape_{i+1}', 
                'bbox': convert_numpy_types(shape_info['bbox']), 
                'center': convert_numpy_types(shape_info['center']),
                'area': convert_numpy_types(shape_info['area']), 
                'ai_description': ai_desc,
                'origin_description': find_corresponding_text(shape_info['bbox'], ocr_results)
            })
        return convert_numpy_types(shape_results)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

@router.post("/ocr")
async def perform_ocr(request: ImageRequest):
    try:
        img = base64_to_cv2_image(request.image_base64)
        det_model, rec_model, triton_url = "det_text", "rec_text", "10.0.10.62:8010"
        boxes, scores = detect_text_triton(img, det_model, triton_url)
        
        rec_texts, rec_boxes, rec_scores = [], [], []
        for i, box in enumerate(boxes):
            x_min, y_min = int(np.min(box[:,0])), int(np.min(box[:,1]))
            x_max, y_max = int(np.max(box[:,0])), int(np.max(box[:,1]))
            text = recognize_text_triton(img[y_min:y_max, x_min:x_max], rec_model, triton_url)
            rec_texts.append(text)
            rec_boxes.append([x_min, y_min, x_max, y_max])
            rec_scores.append(float(scores[i]) if i < len(scores) else 0.0)
            
        return convert_numpy_types({"rec_texts": rec_texts, "rec_boxes": rec_boxes, "rec_scores": rec_scores})
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")