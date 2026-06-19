import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import json
import re
# import torch
from datetime import datetime, timedelta
import joblib
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from openai import OpenAI
import os

from dotenv import load_dotenv
load_dotenv()

print("✅ Imports successful!")

# ==========================================
# 1. คลาสสกัดข้อมูลผ่าน API (ใช้ชื่อเดิมเพื่อไม่ให้ api.py พัง)
# ==========================================
class TyphoonLocalExtractor:
    def __init__(self, model_name="qwen-2.5-coder-32b"):
        print(f"🚀 กำลังเตรียมระบบดึงข้อมูลผ่าน API ด้วยโมเดล {model_name}...")
        
        # API Key ของ Groq
        api_key = os.getenv("GROQ_API_KEY")

        # ต่อเข้า Server ของ Groq API
        self.client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        self.model = model_name
        
        self.target_features = [
            "Family_Diabetes", "Smoker", "Alcohol", "Hypertension", 
            "Prediabetes", "Obesity", "High_Cholesterol", "Fatty_Liver", 
            "Sedentary", "Poor_Diet" 
        ]
        print("✅ เชื่อมต่อ Cloud API สำเร็จ! (ไม่ต้องพึ่ง Tokenizer บนเครื่องแล้ว)")

    def __call__(self, note):
        # หากไม่มีข้อมูล ให้ส่งค่า 0 กลับไปทันที
        if pd.isna(note) or str(note).strip() in ["", "ไม่มีข้อมูลเพิ่มเติม", "ไม่ระบุ"]:
            return {feat: 0.0 for feat in self.target_features}

        system_content = (
            "You are a clinical data extractor. Read the clinical note and return a JSON object. "
            f"Extract exactly these keys: {', '.join(self.target_features)}. \n"
            "Definitions to help you:\n"
            "- Prediabetes: includes high HbA1c, high blood sugar, or insulin resistance.\n"
            "- Obesity: includes overweight, high BMI, or large waist.\n"
            "- Sedentary: includes lack of exercise, sitting all day.\n"
            "- Poor_Diet: includes eating sweet drinks, fast food, or high carbs.\n"
            "Use 1 for Yes (Present), 0 for No. If NOT mentioned, default to 0."
        )
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Clinical Note: {note}\nExtract features as JSON."}
        ]

        try:
            # 🚀 ยิงข้อความหา API ตรงๆ โดยบังคับให้ Server คืนค่ากลับมาเป็นโครงสร้าง JSON เป๊ะๆ
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"}, # ป้องกัน JSON พัง
                temperature=0.0                          # ค่า 0.0 เพื่อให้ผลลัพธ์นิ่งและแม่นยำที่สุด
            )

            clean_json_str = response.choices[0].message.content.strip()
            parsed_json = json.loads(clean_json_str)

            extracted = {}
            for feat in self.target_features:
                if feat in parsed_json:
                    val = str(parsed_json[feat]).strip().lower()
                    extracted[feat] = 1.0 if val in ['1', 'yes', 'true', 'y'] else 0.0
                else:
                    extracted[feat] = 0.0 
            return extracted
            
        except Exception as e:
            print(f"⚠️ API หรือ JSON Parsing ผิดพลาด บังคับใช้ Regex Fallback... (Error: {e})")
            # กรณีฉุกเฉินถ้าเน็ตหลุดหรือ JSON เอ๋อ ใช้ Regex สกัดด่วนแทน
            extracted = {}
            for feat in self.target_features:
                if 'clean_json_str' in locals():
                    match = re.search(fr'"{feat}":\s*["\']?([^"\'\n,]+)["\']?', clean_json_str, re.IGNORECASE)
                    if match:
                        val = match.group(1).strip().lower()
                        extracted[feat] = 1.0 if val in ['1', 'yes', 'true', 'y'] else 0.0
                        continue
                extracted[feat] = 0.0
            return extracted

# ==========================================
# 2. คลาส Pipeline หลักสำหรับประมวลผล
# ==========================================
class DiabetesRiskPipeline:
    def __init__(self, calibrated_model_path, raw_model_path, llm_extractor_func, lookback_window=7):
        # 1. โหลดโมเดล Calibration (.joblib) สำหรับ "ทำนาย Score"
        self.model = joblib.load(calibrated_model_path)
        
        # 2. โหลดโมเดล Raw (.json) สำหรับ "ทำ SHAP Values"
        self.raw_model = xgb.Booster()
        self.raw_model.load_model(raw_model_path)
        
        # 3. โยนโมเดล Raw ให้ SHAP 
        self.explainer = shap.TreeExplainer(self.raw_model)
        
        # 4. ดึงรายชื่อฟีเจอร์ไว้ใช้ดึงข้อมูล 
        if self.raw_model.feature_names is not None:
            self.feature_names = self.raw_model.feature_names
        elif hasattr(self.model, 'feature_names_in_'):
            self.feature_names = list(self.model.feature_names_in_)
        else:
            raise ValueError("ไม่สามารถดึง Feature Names จากโมเดลได้")
        
        self.llm_extractor = llm_extractor_func
        self.lookback_window = lookback_window

    def _feature_engineering(self, df):
        df = df.copy()
        
        if 'Total_Daily_Steps' in df.columns:
            df['Steps_RollMean_3'] = df['Total_Daily_Steps'].rolling(window=3, min_periods=1).mean()
        if 'Sleep_HRV_Avg' in df.columns:
            df['HRV_RollMean_3'] = df['Sleep_HRV_Avg'].rolling(window=3, min_periods=1).mean()
        if 'Daily_Spot_SpO2' in df.columns:
            df['SpO2_RollMean_3'] = df['Daily_Spot_SpO2'].rolling(window=3, min_periods=1).mean()

        scaled_cols = ['Resting_Heart_Rate', 'Sleep_HRV_Avg', 'Daily_Spot_SpO2']
        for col in scaled_cols:
            if col in df.columns:
                std = df[col].std()
                if pd.isna(std) or std == 0:
                    df[f'{col}_patient_scaled'] = 0.0
                else:
                    df[f'{col}_patient_scaled'] = (df[col] - df[col].mean()) / std

        return df

    def process_and_predict(self, user_history_json):
        df = pd.DataFrame(user_history_json)
        df = self._feature_engineering(df)
        latest_features = df.iloc[[-1]].copy()
        
        if 'Clinical_Note' in df.columns:
            valid_notes = df['Clinical_Note'].replace(["", "ไม่มีข้อมูลเพิ่มเติม", "ไม่ระบุ"], pd.NA).dropna()
            recent_notes = valid_notes.tail(self.lookback_window)[::-1]
            
            if not recent_notes.empty:
                combined_notes = []
                for i, note in enumerate(recent_notes):
                    if i == 0:
                        combined_notes.append(f"[บันทึกล่าสุด]: {note}")
                    else:
                        combined_notes.append(f"[บันทึกก่อนหน้า]: {note}")
                final_note_text = " | ".join(combined_notes)
            else:
                final_note_text = ""
        else:
            final_note_text = ""

        extracted_notes = self.llm_extractor(final_note_text)
        
        for feat, val in extracted_notes.items():
            latest_features[feat] = val
            
        try:
            X_pred = latest_features[self.feature_names]
        except KeyError as e:
            return {"error": f"Missing required features: {e}"}
        
        # ทำนาย Score 
        probabilities = self.model.predict_proba(X_pred)[0]
        prob_0 = float(probabilities[0])
        prob_1 = float(probabilities[1])
        prob_2 = float(probabilities[2])
        
        risk_score = (prob_1 * 60) + (prob_2 * 100)
        risk_score_percentage = round(risk_score, 2)
        
        if risk_score_percentage <= 30:
            risk_level = 0
        elif risk_score_percentage <= 60:
            risk_level = 1
        else:
            risk_level = 2
            
        # คำนวณ SHAP Values
        dmatrix = xgb.DMatrix(X_pred)
        raw_shap_values = self.explainer.shap_values(dmatrix)
        
        target_class = risk_level if risk_level != 0 else 1
        
        if isinstance(raw_shap_values, list):
            target_shap = raw_shap_values[target_class][0]
        else:
            if len(raw_shap_values.shape) == 3:
                target_shap = raw_shap_values[0, :, target_class]
            else:
                target_shap = raw_shap_values[0]

        feature_impacts = []
        for feat_name, shap_val, actual_val in zip(self.feature_names, target_shap, X_pred.iloc[0]):
            feature_impacts.append({
                "feature": feat_name,
                "value": float(actual_val) if not pd.isna(actual_val) else None,
                "impact_score": float(shap_val)
            })
            
        feature_impacts.sort(key=lambda x: x["impact_score"], reverse=True)

        return {
            "status": "success",
            "prediction": {
                "risk_score_percentage": risk_score_percentage,
                "risk_level": risk_level,
                "risk_score_0": round(prob_0, 4),
                "risk_score_1": round(prob_1, 4),
                "risk_score_2": round(prob_2, 4),
            },
            "insights": {
                "top_driving_factors": feature_impacts[:5],
                "top_reducing_factors": feature_impacts[-3:]
            }
        }

# ==========================================
# 3. บล็อกทดสอบภายในไฟล์ (ป้องกันการรันซ้ำซ้อนตอนเรียกใช้ผ่าน api.py)
# ==========================================
if __name__ == "__main__":
    CALIBRATED_MODEL_PATH = "./models/smartwatch_diabetes_prediction_model.joblib"
    RAW_MODEL_PATH = "./models/smartwatch_diabetes_raw_prediction_model.json"

    try:
        print("⏳ [Local Test] Starting model initialization...")
        llm_service = TyphoonLocalExtractor() 
        pipeline = DiabetesRiskPipeline(
            calibrated_model_path=CALIBRATED_MODEL_PATH, 
            raw_model_path=RAW_MODEL_PATH,
            llm_extractor_func=llm_service
        )
        print("\n🟢 PIPELINE IS READY FOR LOCAL TESTING! 🟢")
    except Exception as e:
        print(f"🔴 Error loading models: {e}")