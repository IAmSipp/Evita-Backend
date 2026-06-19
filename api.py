from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Union
from contextlib import asynccontextmanager
import uvicorn
from vital.client import Vital
from vital.environment import VitalEnvironment
import os

from dotenv import load_dotenv
load_dotenv()

# 1. นำเข้า "พ่อครัว" (คลาสที่เราเขียนไว้ใน Pipeline.py)
from Pipeline import TyphoonLocalExtractor, DiabetesRiskPipeline

# ==========================================
# ส่วนตั้งค่า Vital API (ใส่ Key ของคุณจาก Vital Dashboard)
# ==========================================
VITAL_API_KEY = os.getenv("VITAL_API_KEY")

try:
    # ใช้โหมด SANDBOX สำหรับทดสอบ (ไม่ต้องเสียเงินจริง)
    vital_client = Vital(
        api_key=VITAL_API_KEY,
        environment=VitalEnvironment.SANDBOX
    )
except Exception as e:
    print(f"⚠️ Warning: Vital Client initialized without valid credentials. Error: {e}")
    vital_client = None

# ถังพักข้อมูลชั่วคราวสำหรับ Webhook
webhook_data_store = {}

# 2. กำหนดรูปแบบข้อมูล (Schema) ที่ Frontend ต้องส่งเข้ามา
class DailyHealthData(BaseModel):
    Date: str
    Age: float
    Gender: float
    Sleep_Duration: float
    Sleep_Efficiency: float
    Sleep_Regularity_Index: float
    Resting_Heart_Rate: float
    Sleep_HRV_Avg: float
    Daily_Spot_SpO2: float
    Total_Daily_Steps: float
    Sedentary_Hours: float
    Clinical_Note: Optional[str] = ""

class PredictionRequest(BaseModel):
    user_history: List[DailyHealthData]

# ตัวแปรสำหรับเก็บ Pipeline ไว้ใน Memory หลัก
global_pipeline = None

# 3. ฟังก์ชันโหลดโมเดลตอนเปิดเซิร์ฟเวอร์
@asynccontextmanager
async def lifespan(app: FastAPI):
    global global_pipeline
    print("⏳ [Server Startup] กำลังโหลด AI Models (LLM & XGBoost) กรุณารอสักครู่...")
    
    CALIBRATED_MODEL_PATH = "./models/smartwatch_diabetes_prediction_model_old.joblib"
    RAW_MODEL_PATH = "./models/smartwatch_diabetes_raw_prediction_model_old.json"

    try:
        print("⏳ Starting model initialization...")
        llm_service = TyphoonLocalExtractor("llama-3.3-70b-versatile") 
        
        global_pipeline = DiabetesRiskPipeline(
            calibrated_model_path=CALIBRATED_MODEL_PATH, 
            raw_model_path=RAW_MODEL_PATH,
            llm_extractor_func=llm_service
        )
        
        print("\n🟢 PIPELINE IS READY! 🟢")
    except Exception as e:
        print(f"🔴 Error loading models: {e}")
        
    yield 
    
    print("🛑 [Server Shutdown] ปิดระบบ API...")

# 4. สร้างตัว API Server
app = FastAPI(title="Smartwatch Diabetes Prediction API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://evita-frontend.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 5. VITAL API ENDPOINTS (เปลี่ยนจาก Terra เป็น Vital)
# ==========================================

# 5.1 Endpoint สร้าง Link Token สำหรับเปิดหน้าต่างเชื่อมต่อ Vital
@app.get("/api/vital/token")
async def get_vital_token(session_id: str = "test-session-1"):
    if not vital_client:
        raise HTTPException(status_code=500, detail="Vital client is not properly configured.")
    try:
        # 1. สร้าง User ในระบบของ Vital ก่อน (หรือดึงของเดิมถ้ามี)
        user = vital_client.user.create(client_user_id=session_id)
        
        # 2. ขอ Token เพื่อให้ฝั่งหน้าบ้านเอาไปเปิด Widget
        token_response = vital_client.link.token.create(user_id=user.user_id)
        return {"link_token": token_response.link_token, "user_id": user.user_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 5.2 Endpoint รับ Webhook จากเซิร์ฟเวอร์ Vital
@app.post("/api/vital/webhook")
async def vital_webhook(request: Request):
    body = await request.json()
    
    # รูปแบบ Payload ของ Vital จะต่างออกไป
    event_type = body.get("event_type")
    data = body.get("data", {})
    vital_user_id = data.get("user_id")
    
    # ดักจับเฉพาะข้อมูลสรุปรายวัน (Daily)
    if vital_user_id and event_type and "daily.data" in event_type:
        webhook_data_store[vital_user_id] = [data]
        print(f"✅ [Webhook] ได้รับข้อมูลสุขภาพจาก Vital สำหรับ User: {vital_user_id}")
        
    return {"status": "success"}

# 5.3 Endpoint สำหรับ Frontend ทำ Polling ดึงข้อมูล
@app.get("/api/debug/fetch_watch_data")
async def fetch_watch_data():
    # ระบบจำลองแบบกวาด: ถ้ามีข้อมูลใดๆ เข้ามาใน Webhook Store ให้ดึงออกไปโชว์ที่หน้าบ้านเลย
    if len(webhook_data_store) > 0:
        key = list(webhook_data_store.keys())[0]
        data = webhook_data_store.pop(key)
        return {"data": data}
        
    raise HTTPException(status_code=404, detail="Data not ready yet")


# ==========================================
# 6. PREDICTION API (ของคุณ)
# ==========================================
@app.post("/api/predict_risk")
async def predict_risk(request: Union[List[DailyHealthData], PredictionRequest]):
    if global_pipeline is None:
        raise HTTPException(status_code=500, detail="AI Model is not loaded yet.")
    
    try:
        user_history = request if isinstance(request, list) else request.user_history

        if not user_history:
            raise HTTPException(status_code=400, detail="Request body must contain at least one daily health record.")

        print(f"📩 ได้รับข้อมูลจาก Frontend จำนวน {len(user_history)} วัน")
        
        raw_data = [item.model_dump() for item in user_history]
        result = global_pipeline.process_and_predict(raw_data)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
            
        return result
    except HTTPException:
        raise
    except Exception as e:
        print(f"🔴 API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Diabetes Prediction API is running!"}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)