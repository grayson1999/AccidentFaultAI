from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import shutil
import os
import sys

# Importing recognizer modules
sys.path.append('/AccidentFaultAI/')
from recognizer.single_tsn_recognizer import Single_tsn_recognizer
from recognizer.yolo_tsn_recognizer import Yolo_tsn_recognizer

# FastAPI 애플리케이션 초기화
app = FastAPI(title="VideoRecognizerAPI", description="An API to upload videos and get recognition results", version="1.0")

# Recognizer 인스턴스 초기화
rcn = Single_tsn_recognizer()
# rcn = Yolo_tsn_recognizer()

# 템플릿 디렉토리 설정
templates = Jinja2Templates(directory="templates")

# 정적 파일 서빙 설정 (예: CSS 파일)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 기본 경로 정의
@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 비디오 예측 엔드포인트 정의
@app.post("/predict")
async def predict(video: UploadFile = File(...)):
    save_path = f"./video_save/{video.filename}"
    try:
        # 비디오 파일을 저장
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # 비디오 파일을 처리하는 함수 호출
        result = rcn.predict(save_path)

        # 결과를 JSON 형식으로 반환
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # 파일 삭제
        if os.path.exists(save_path):
            os.remove(save_path)

# FastAPI 애플리케이션 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9904)
