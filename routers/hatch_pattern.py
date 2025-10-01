# routers/hatch_pattern.py

import cv2
import numpy as np
import json
import time
import torch
import open_clip
from PIL import Image
import base64
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os

from modules.polygon_extractor import PolygonExtractor

router = APIRouter()

# --- Globals ---
detector: Optional['HatchPatternDetector'] = None
default_patterns: List['PatternDescription'] = []

class PatternDescription(BaseModel):
    pattern_id: str
    ai_description: str

class HatchPatternDetector:
    def __init__(self, max_batch=128, polygon_extractor_config=None):
        self.max_batch = max_batch
        if polygon_extractor_config is None:
            polygon_extractor_config = {
                "angle_thresh": 80.0, "white_pixel_threshold": 230, "white_ratio_threshold": 0.95,
                "dash_frac_thresh": 0.35, "max_allowed_dashed_edges": 2, "angle_tolerance_deg": 15,
                "offset_px": 3, "margin_threshold": 0
            }
        self.polygon_extractor = PolygonExtractor(**polygon_extractor_config)
        self._load_model()
        self.descriptions, self.pattern_ids, self.text_features = [], [], None

    def _load_model(self):
        print("🚀 Đang load OpenCLIP model...")
        t0 = time.time()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Sử dụng thiết bị: {self.device}")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-H-14-378-quickgelu', pretrained='dfn5b')
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer('ViT-H-14-378-quickgelu')
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        self.use_half = (self.device == "cuda")
        if self.use_half: self.model.half()
        print(f"✓ Model loaded trong: {time.time() - t0:.2f} giây")

    def set_descriptions(self, pattern_descriptions: List[PatternDescription]):
        print(f"📝 Đang set {len(pattern_descriptions)} descriptions...")
        self.descriptions = [item.ai_description for item in pattern_descriptions]
        self.pattern_ids = [item.pattern_id for item in pattern_descriptions]
        self.text_features = self._encode_text_features() if self.descriptions else None
        print(f"✓ Đã set {len(self.descriptions)} descriptions")

    def _encode_text_features(self):
        print("🔤 Đang encode text features...")
        t0 = time.time()
        tokens = self.tokenizer(self.descriptions).to(self.device)
        with torch.inference_mode():
            tf = torch.nn.functional.normalize(self.model.encode_text(tokens), dim=-1)
        print(f"✓ Text features encoded trong: {time.time() - t0:.2f} giây")
        return tf.half() if self.use_half else tf

    def _clip_preprocess_to_tensor(self, pil_image: Image.Image):
        t = self.preprocess(pil_image).unsqueeze(0).to(self.device, non_blocking=True)
        return t.half() if self.use_half and t.dtype == torch.float32 else t

    def _classify_regions_batched(self, regions_data):
        if not regions_data or self.text_features is None: return []
        results = []
        for start in range(0, len(regions_data), self.max_batch):
            chunk = regions_data[start:start + self.max_batch]
            tensors = [self._clip_preprocess_to_tensor(Image.fromarray(cv2.cvtColor(r["crop_data"], cv2.COLOR_BGR2RGB))) for r in chunk]
            images = torch.cat(tensors, dim=0)
            with torch.inference_mode():
                img_feat = torch.nn.functional.normalize(self.model.encode_image(images), dim=-1)
                sims = (100.0 * img_feat @ self.text_features.T).softmax(dim=-1)
            
            probs = sims.detach().float().cpu().numpy()
            for i, r in enumerate(chunk):
                # SỬA LỖI 1: Ép kiểu argmax() từ numpy.int64 -> int
                k = int(probs[i].argmax())
                # SỬA LỖI 2: Ép kiểu giá trị xác suất từ numpy.float64 -> float
                p = float(probs[i][k])
                
                # Cập nhật dictionary kết quả
                # Loại bỏ 'crop_data' để không trả về dữ liệu ảnh thô trong JSON
                region_info = {key: val for key, val in r.items() if key != 'crop_data'}
                region_info["classification"] = {
                    "predicted_class": k + 1,
                    "predicted_description": self.descriptions[k],
                    "pattern_id": self.pattern_ids[k],
                    "confidence": p
                }
                results.append(region_info)
        return results
    
    def process_image(self, img_array, margin_threshold=0, **extractor_kwargs):
        if not self.descriptions or self.text_features is None:
            return {"success": False, "message": "Chưa có descriptions. Hãy gọi set_descriptions() trước."}
        t0 = time.time()
        for key, value in extractor_kwargs.items():
            if hasattr(self.polygon_extractor, key): setattr(self.polygon_extractor, key, value)
        
        regions_data, skipped_margin_regions = self.polygon_extractor.extract_polygons(img_array, margin_threshold)
        if not regions_data:
            return {"success": False, "message": "Không tìm thấy vùng nào để phân loại.", "processing_time": round(time.time()-t0, 2)}
        
        classified_regions = self._classify_regions_batched(regions_data)
        stats = regions_data[0].get("statistics", {})
        total_regions = len(classified_regions)
        summary = {f"class_{i+1}": {"description": desc, "pattern_id": self.pattern_ids[i], "count": 0, "percentage": 0.0, "regions": []} for i, desc in enumerate(self.descriptions)}
        
        for r in classified_regions:
            k = r["classification"]["predicted_class"]
            entry = summary[f"class_{k}"]
            entry["count"] += 1
            # Loại bỏ các key không cần thiết khi đưa vào summary
            summary_region_info = {
                "region_id": r["region_id"],
                "bbox": r["bbox"],
                "center": r["center"],
                "area": r["area"],
                "vertices": r["vertices"],
                "confidence": r["classification"]["confidence"],
                "source": r["source"],
                "dashed_edges": r.get("dashed_edges", 0)
            }
            entry["regions"].append(summary_region_info)
        
        for v in summary.values():
            if total_regions > 0: v["percentage"] = round(v["count"] / total_regions * 100.0, 1)

        processing_time = time.time() - t0
        return {
            "success": True,
            "image_info": {"dimensions": {"width": img_array.shape[1], "height": img_array.shape[0]}, "total_regions_found": total_regions, "margin_threshold": margin_threshold, "processing_time": round(processing_time, 2), "device": self.device, "precision": "fp16" if self.use_half else "fp32", "detection_statistics": stats},
            "classifications": summary, 
            "out_padding": skipped_margin_regions,
            "all_regions": classified_regions 
        }

def load_default_patterns(json_file_path="./result.json"):
    try:
        if not os.path.exists(json_file_path):
            print(f"⚠️ File {json_file_path} không tồn tại")
            return []
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        patterns = [PatternDescription(**item) for item in data if isinstance(item, dict) and 'pattern_id' in item and 'ai_description' in item]
        print(f"✅ Đã load {len(patterns)} patterns từ {json_file_path}")
        return patterns
    except Exception as e:
        print(f"❌ Lỗi khi load file {json_file_path}: {str(e)}")
        return []

def initialize_hatch_detector():
    """Khởi tạo detector và load patterns."""
    global detector, default_patterns
    try:
        detector = HatchPatternDetector(max_batch=128)
        default_patterns = load_default_patterns("./result.json")
        if default_patterns:
            detector.set_descriptions(default_patterns)
        print("✅ Hatch Pattern Detector đã sẵn sàng!")
    except Exception as e:
        print(f"❌ Lỗi khởi tạo Hatch Pattern Detector: {str(e)}")
        detector = None

class DetectionRequest(BaseModel):
    image_base64: str
    pattern_descriptions: Optional[List[PatternDescription]] = None
    margin_threshold: int = 0
    angle_threshold: Optional[float] = 80.0
    white_pixel_threshold: Optional[int] = 230
    white_ratio_threshold: Optional[float] = 0.95
    dash_frac_threshold: Optional[float] = 0.35
    max_allowed_dashed_edges: Optional[int] = 2

def decode_base64_image(base64_string: str) -> np.ndarray:
    try:
        if ',' in base64_string: base64_string = base64_string.split(',')[1]
        img_array = cv2.imdecode(np.frombuffer(base64.b64decode(base64_string), np.uint8), cv2.IMREAD_COLOR)
        if img_array is None: raise ValueError("Không thể decode ảnh từ base64")
        return img_array
    except Exception as e:
        raise ValueError(f"Lỗi decode base64: {str(e)}")

@router.post("/detect_hatch_pattern")
async def detect_hatch_patterns(request: DetectionRequest):
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector chưa được khởi tạo hoặc bị lỗi.")
    
    patterns_to_use = request.pattern_descriptions or default_patterns
    if not patterns_to_use:
        raise HTTPException(status_code=400, detail="Không có pattern descriptions.")
    
    try:
        t0 = time.time()
        detector.set_descriptions(patterns_to_use)
        img_array = decode_base64_image(request.image_base64)
        
        extractor_params = {
            'angle_thresh': request.angle_threshold, 'white_pixel_threshold': request.white_pixel_threshold,
            'white_ratio_threshold': request.white_ratio_threshold, 'dash_frac_thresh': request.dash_frac_threshold,
            'max_allowed_dashed_edges': request.max_allowed_dashed_edges
        }
        result = detector.process_image(img_array, request.margin_threshold, **extractor_params)
        
        if result.get("image_info"):
            result["image_info"]["pattern_source"] = "request" if request.pattern_descriptions else "default_file"
            result["image_info"]["patterns_count"] = len(patterns_to_use)
        
        print(f"⏱️ Tổng thời gian xử lý: {time.time() - t0:.2f} giây")
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # SỬA LỖI 3: Thêm traceback để debug dễ hơn trên server log
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")

@router.get("/info")
async def get_info():
    if detector is None:
        return {"status": "Detector chưa sẵn sàng"}
    
    return {
        "status": "Sẵn sàng",
        "device": detector.device,
        "precision": "fp16" if detector.use_half else "fp32",
        "default_patterns_available": len(default_patterns),
        "available_classes": [{"class_id": i+1, "pattern_id": pid, "description": desc} for i, (pid, desc) in enumerate(zip(detector.pattern_ids, detector.descriptions))]
    }