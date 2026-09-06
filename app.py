# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from inference import pan_ocr_inference
# import pymysql
# import os
# from datetime import datetime

# app = Flask(__name__)
# CORS(app, origins="*")

# # ── DATABASE CONFIG ── change these to your MySQL details
# DB_CONFIG = {
#     "host":     "localhost",
#     "user":     "root",          # your MySQL username
#     "password": "Sonia1978",  # your MySQL password
#     "database": "pan_ocr",
#     "cursorclass": pymysql.cursors.DictCursor
# }

# def get_db():
#     return pymysql.connect(**DB_CONFIG)

# def save_to_db(results):
#     """Save a list of scan results to MySQL"""
#     conn = get_db()
#     try:
#         with conn.cursor() as cursor:
#             for r in results:
#                 cursor.execute("""
#                     INSERT INTO scan_results
#                         (filename, pan_number, name, father_name, dob,
#                          is_pan_card, confidence, status, error_msg)
#                     VALUES
#                         (%s, %s, %s, %s, %s, %s, %s, %s, %s)
#                 """, (
#                     r.get("filename",    ""),
#                     r.get("pan_number",  ""),
#                     r.get("name",        ""),
#                     r.get("father_name", ""),
#                     r.get("dob",         ""),
#                     r.get("is_pan_card", True),
#                     r.get("confidence",  0.0),
#                     r.get("status",      "success"),
#                     r.get("error",       "")
#                 ))
#         conn.commit()
#     finally:
#         conn.close()

# # ── Single image ──
# @app.route('/predict', methods=['POST'])
# def predict():
#     image_file = request.files.get('image')
#     if not image_file:
#         return jsonify({"error": "No image provided"}), 400

#     image_bytes = image_file.read()
#     ocr_results = pan_ocr_inference(image_bytes)

#     result = {
#         "filename":    image_file.filename,
#         "pan_number":  ocr_results.get("PAN Number",  {}).get("text", ""),
#         "name":        ocr_results.get("Name",        {}).get("text", ""),
#         "father_name": ocr_results.get("Father Name", {}).get("text", ""),
#         "dob":         ocr_results.get("DOB",         {}).get("text", ""),
#         "is_pan_card": True,
#         "confidence":  0.95,
#         "status":      "success"
#     }

#     # Save to database
#     save_to_db([result])

#     return jsonify(result)

# # ── Multiple images ──
# @app.route('/predict-bulk', methods=['POST'])
# def predict_bulk():
#     files = request.files.getlist('images')
#     if not files:
#         return jsonify({"error": "No images provided"}), 400

#     results = []

#     for file in files:
#         try:
#             image_bytes = file.read()
#             ocr_results = pan_ocr_inference(image_bytes)

#             results.append({
#                 "filename":    file.filename,
#                 "status":      "success",
#                 "pan_number":  ocr_results.get("PAN Number",  {}).get("text", ""),
#                 "name":        ocr_results.get("Name",        {}).get("text", ""),
#                 "father_name": ocr_results.get("Father Name", {}).get("text", ""),
#                 "dob":         ocr_results.get("DOB",         {}).get("text", ""),
#                 "is_pan_card": True,
#                 "confidence":  0.95
#             })

#         except Exception as e:
#             results.append({
#                 "filename": file.filename,
#                 "status":   "error",
#                 "error":    str(e)
#             })

#     # Save all to database
#     save_to_db(results)

#     return jsonify({
#         "total":   len(files),
#         "success": len([r for r in results if r["status"] == "success"]),
#         "failed":  len([r for r in results if r["status"] == "error"]),
#         "results": results
#     })

# # ── Get all scans from DB ──
# @app.route('/scans', methods=['GET'])
# def get_scans():
#     conn = get_db()
#     try:
#         with conn.cursor() as cursor:
#             cursor.execute("SELECT * FROM scan_results ORDER BY scanned_at DESC")
#             rows = cursor.fetchall()
#         return jsonify({"total": len(rows), "scans": rows})
#     finally:
#         conn.close()

# # ── Get scans by PAN number ──
# @app.route('/scans/<pan_number>', methods=['GET'])
# def get_scan_by_pan(pan_number):
#     conn = get_db()
#     try:
#         with conn.cursor() as cursor:
#             cursor.execute(
#                 "SELECT * FROM scan_results WHERE pan_number = %s ORDER BY scanned_at DESC",
#                 (pan_number,)
#             )
#             rows = cursor.fetchall()
#         return jsonify({"total": len(rows), "scans": rows})
#     finally:
#         conn.close()

# # ── Delete a scan ──
# @app.route('/scans/<int:scan_id>', methods=['DELETE'])
# def delete_scan(scan_id):
#     conn = get_db()
#     try:
#         with conn.cursor() as cursor:
#             cursor.execute("DELETE FROM scan_results WHERE id = %s", (scan_id,))
#         conn.commit()
#         return jsonify({"message": "Deleted successfully"})
#     finally:
#         conn.close()

# @app.route('/health')
# def health():
#     return jsonify({"status": "ok"})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5001, debug=True)
# -----------------------------------------------------------------------------

import os
import pymysql
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from inference import pan_ocr_inference
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── DB CONFIG ──
DB_CONFIG = {
    "host":     os.environ.get("MYSQL_HOST",     "mysql-service"),
    "user":     os.environ.get("MYSQL_USER",     "root"),
    "password": os.environ.get("MYSQL_PASSWORD", "RootPass123"),
    "database": os.environ.get("MYSQL_DATABASE", "pan_ocr"),
    "port":     int(os.environ.get("MYSQL_PORT",  3306)),
    "cursorclass": pymysql.cursors.DictCursor
}

def get_db():
    return pymysql.connect(**DB_CONFIG)

def save_to_db(results):
    try:
        conn = get_db()
        with conn.cursor() as cursor:
            for r in results:
                cursor.execute("""
                    INSERT INTO scan_results
                        (filename, pan_number, name, father_name,
                         dob, is_pan_card, confidence, status, error_msg)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    r.get("filename",    ""),
                    r.get("pan_number",  ""),
                    r.get("name",        ""),
                    r.get("father_name", ""),
                    r.get("dob",         ""),
                    r.get("is_pan_card", True),
                    r.get("confidence",  0.0),
                    r.get("status",      "success"),
                    r.get("error",       "")
                ))
        conn.commit()
    except Exception as e:
        print(f"[DB ERROR] {e}")
    finally:
        try:
            conn.close()
        except:
            pass

# ── ROUTES ──

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        ocr_results = pan_ocr_inference(image_bytes)

        result = {
            "filename":    image.filename or "upload.jpg",
            "pan_number":  ocr_results.get("PAN Number",  {}).get("text", ""),
            "name":        ocr_results.get("Name",        {}).get("text", ""),
            "father_name": ocr_results.get("Father Name", {}).get("text", ""),
            "dob":         ocr_results.get("DOB",         {}).get("text", ""),
            "is_pan_card": True,
            "confidence":  0.95,
            "status":      "success"
        }

        save_to_db([result])
        return result

    except Exception as e:
        err = {
            "filename": image.filename or "upload.jpg",
            "status":   "error",
            "error":    str(e)
        }
        save_to_db([err])
        return err

@app.post("/predict-bulk")
async def predict_bulk(images: List[UploadFile] = File(...)):
    results = []

    for file in images:
        try:
            image_bytes = await file.read()
            ocr_results = pan_ocr_inference(image_bytes)

            results.append({
                "filename":    file.filename or "upload.jpg",
                "status":      "success",
                "pan_number":  ocr_results.get("PAN Number",  {}).get("text", ""),
                "name":        ocr_results.get("Name",        {}).get("text", ""),
                "father_name": ocr_results.get("Father Name", {}).get("text", ""),
                "dob":         ocr_results.get("DOB",         {}).get("text", ""),
                "is_pan_card": True,
                "confidence":  0.95
            })

        except Exception as e:
            results.append({
                "filename": file.filename or "upload.jpg",
                "status":   "error",
                "error":    str(e)
            })

    save_to_db(results)

    return {
        "total":   len(images),
        "success": len([r for r in results if r["status"] == "success"]),
        "failed":  len([r for r in results if r["status"] == "error"]),
        "results": results
    }

@app.get("/scans")
def get_scans():
    try:
        conn = get_db()
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM scan_results ORDER BY scanned_at DESC"
            )
            rows = cursor.fetchall()
        conn.close()
        return {"total": len(rows), "scans": rows}
    except Exception as e:
        return {"error": str(e)}

@app.get("/scans/{pan_number}")
def get_scan_by_pan(pan_number: str):
    try:
        conn = get_db()
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM scan_results WHERE pan_number = %s ORDER BY scanned_at DESC",
                (pan_number,)
            )
            rows = cursor.fetchall()
        conn.close()
        return {"total": len(rows), "scans": rows}
    except Exception as e:
        return {"error": str(e)}

@app.delete("/scans/{scan_id}")
def delete_scan(scan_id: int):
    try:
        conn = get_db()
        with conn.cursor() as cursor:
            cursor.execute(
                "DELETE FROM scan_results WHERE id = %s", (scan_id,)
            )
        conn.commit()
        conn.close()
        return {"message": f"Scan #{scan_id} deleted"}
    except Exception as e:
        return {"error": str(e)}

# ── RUN ──
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5001))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)