from fastapi import FastAPI
import uvicorn

# Import các router từ thư mục routers
from routers import shape_analysis, description_ocr, hatch_pattern

# Import hàm khởi tạo cho hatch_pattern
from routers.hatch_pattern import initialize_hatch_detector

app = FastAPI(
    title="Combined Image Analysis API",
    description="Một API hợp nhất cho Shape Analysis, OCR/Description và Hatch Pattern Detection.",
    version="3.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Sự kiện khi khởi động server: khởi tạo các model cần thiết."""
    print("--- Server đang khởi động, bắt đầu load model ---")
    initialize_hatch_detector()
    print("--- Server đã sẵn sàng nhận request ---")


# Thêm các router vào ứng dụng chính với các prefix khác nhau
# Prefix giúp phân biệt các nhóm API, ví dụ: /shapes/process-images
app.include_router(
    shape_analysis.router,
    prefix="/shapes",
    tags=["Shape and Connection Analysis"]
)

app.include_router(
    description_ocr.router,
    prefix="/ocr",
    tags=["OCR and AI Description"]
)

app.include_router(
    hatch_pattern.router,
    prefix="/hatch",
    tags=["Hatch Pattern Detection"]
)

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Chào mừng đến với Combined Image Analysis API",
        "version": app.version,
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "available_services": {
            "shape_analysis": "Truy cập /shapes/*",
            "ocr_and_description": "Truy cập /ocr/*",
            "hatch_pattern_detection": "Truy cập /hatch/*"
        }
    }

if __name__ == "__main__":
    # Chạy ứng dụng trên một port duy nhất
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)