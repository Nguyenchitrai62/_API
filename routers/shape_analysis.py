from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional
import numpy as np
import cv2
import base64
import asyncio
from pydantic import BaseModel, Field

# Import các hàm từ file module tương ứng
from modules.outline_2 import process_multiple_images_with_connection_detection

router = APIRouter()

# Model cho base64 input
class Base64ImageRequest(BaseModel):
    images: List[str] = Field(..., description="Danh sách ảnh dạng base64 string")
    scale: Optional[float] = Field(72/400, description="Hệ số scale cho tọa độ đầu ra")
    min_area_object: Optional[float] = Field(0.000001, description="Diện tích tối thiểu của đối tượng")
    min_area_outline: Optional[float] = Field(0.0005, description="Diện tích tối thiểu cho outline")
    duplicate_tolerance: Optional[int] = Field(10, description="Độ dung sai để loại bỏ contour trùng lặp")
    max_distance_ratio: Optional[float] = Field(2.5, description="Tỷ lệ khoảng cách tối đa để kiểm tra kết nối")
    small_threshold: Optional[int] = Field(3, description="Số shape tối thiểu để merge pipe group")
    min_dist_threshold: Optional[float] = Field(20, description="Khoảng cách tối thiểu để merge pipe group")

def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 string thành numpy array"""
    if base64_string.startswith('data:image'):
        base64_string = base64_string.split(',')[1]
    image_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Không thể decode ảnh từ base64")
    return image

async def process_images_async(
    images: List[np.ndarray],
    scale: float, min_area_object: float, min_area_outline: float,
    duplicate_tolerance: int, max_distance_ratio: float,
    small_threshold: int, min_dist_threshold: float
):
    """Xử lý ảnh và trả về JSON."""
    json_data, _ = await asyncio.to_thread(
        process_multiple_images_with_connection_detection,
        input_sources=images,
        scale=scale,
        min_area_object=min_area_object,
        min_area_outline=min_area_outline,
        duplicate_tolerance=duplicate_tolerance,
        max_distance_ratio=max_distance_ratio,
        small_threshold=small_threshold,
        min_dist_threshold=min_dist_threshold
    )
    return json_data

@router.post("/process-images")
async def process_images(
    files: List[UploadFile] = File(...),
    scale: Optional[float] = Form(72/400),
    min_area_object: Optional[float] = Form(0.000001),
    min_area_outline: Optional[float] = Form(0.0005),
    duplicate_tolerance: Optional[int] = Form(10),
    max_distance_ratio: Optional[float] = Form(2.5),
    small_threshold: Optional[int] = Form(3),
    min_dist_threshold: Optional[float] = Form(20)
):
    """Endpoint để xử lý ảnh từ file upload"""
    if not files:
        raise HTTPException(status_code=400, detail="Không có file nào được upload")
    
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    images = []
    
    try:
        for file in files:
            file_ext = '.' + file.filename.split('.')[-1].lower()
            if file_ext not in allowed_extensions:
                raise HTTPException(status_code=400, detail=f"File {file.filename} không được hỗ trợ.")
            
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise HTTPException(status_code=400, detail=f"Không thể đọc ảnh từ file {file.filename}")
            images.append(image)

        if len(images) > 1:
            first_shape = images[0].shape[:2]
            if any(img.shape[:2] != first_shape for img in images[1:]):
                raise HTTPException(status_code=400, detail="Các ảnh phải có cùng kích thước.")

        result = await process_images_async(
            images, scale, min_area_object, min_area_outline,
            duplicate_tolerance, max_distance_ratio, small_threshold, min_dist_threshold
        )
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")


@router.post("/process-images-base64")
async def process_images_base64(request: Base64ImageRequest):
    """Endpoint để xử lý ảnh từ base64 string"""
    if not request.images:
        raise HTTPException(status_code=400, detail="Không có ảnh nào được gửi")
    
    try:
        images = [decode_base64_image(b64) for b64 in request.images]

        if len(images) > 1:
            first_shape = images[0].shape[:2]
            if any(img.shape[:2] != first_shape for img in images[1:]):
                 raise HTTPException(status_code=400, detail="Các ảnh phải có cùng kích thước.")

        result = await process_images_async(
            images, request.scale, request.min_area_object, request.min_area_outline,
            request.duplicate_tolerance, request.max_distance_ratio, request.small_threshold,
            request.min_dist_threshold
        )
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")