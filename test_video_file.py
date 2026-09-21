"""
Test script to process frames from a video file (sample_original.mp4)
via the ANPR Backend API and verify MongoDB persistence.
"""
import os
import cv2
import sys
import time
import requests
import json
from datetime import datetime

VIDEO_PATH = "sample_original.mp4"
API_URL = "http://localhost:8000/api/v1/detect/image"

if not os.path.exists(VIDEO_PATH):
    print(f"Error: Video file not found at {VIDEO_PATH}")
    sys.exit(1)

cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration = total_frames / fps if fps > 0 else 0

print("=" * 80)
print("ANPR VIDEO TEST RUNNER")
print(f"Video Path:   {VIDEO_PATH}")
print(f"Resolution:   {width}x{height}")
print(f"FPS:          {fps:.2f}")
print(f"Total Frames: {total_frames}")
print(f"Duration:     {duration:.2f} seconds")
print(f"API Target:   {API_URL}")
print("=" * 80)

# Process 1 frame every 15 frames (approx 4 fps of video time)
# For 60s @ 60fps = 3600 frames -> 240 sampled frames
FRAME_STEP = 15

frame_idx = 0
processed_count = 0
vehicles_found = 0
plates_found = 0
mongo_records_created = 0

results = []
plates_detected_summary = {}

t_start = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % FRAME_STEP == 0:
        processed_count += 1
        sec = frame_idx / fps if fps > 0 else 0
        
        # Encode frame to JPEG
        _, img_encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_bytes = img_encoded.tobytes()

        filename = f"frame_{frame_idx:05d}_at_{sec:.1f}s.jpg"
        
        t0 = time.time()
        try:
            resp = requests.post(
                API_URL,
                files={"file": (filename, img_bytes, "image/jpeg")},
                data={"camera_id": "sample_video_cam1"},
                timeout=30
            )
            elapsed_ms = int((time.time() - t0) * 1000)
            
            if resp.status_code == 200:
                data = resp.json()
                status = data.get("status")
                plate_num = data.get("plate_number")
                ocr_conf = data.get("confidence")
                
                vehicle_info = data.get("vehicle", {})
                v_detected = vehicle_info.get("detected", False)
                v_type = vehicle_info.get("vehicle_type", "N/A")
                v_conf = vehicle_info.get("confidence", 0)

                plate_info = data.get("plate_detection", {})
                p_detected = plate_info.get("detected", False)

                storage_info = data.get("storage", {})
                plate_record_id = storage_info.get("plate_record_id")
                plate_image_id = storage_info.get("plate_image_id")

                if v_detected:
                    vehicles_found += 1
                if p_detected:
                    plates_found += 1
                if plate_record_id:
                    mongo_records_created += 1

                if plate_num:
                    plates_detected_summary[plate_num] = plates_detected_summary.get(plate_num, 0) + 1

                print(f"[{processed_count:03d} | Frame {frame_idx:04d} ({sec:.1f}s)] "
                      f"Status: {status:<15} | Vehicle: {v_type} ({v_conf:.2f} if v_conf else 0) | "
                      f"Plate: {plate_num or 'N/A'} (conf: {ocr_conf if ocr_conf else 0:.2f}) | "
                      f"MongoID: {plate_record_id or 'None'} ({elapsed_ms}ms)")

                results.append({
                    "frame_idx": frame_idx,
                    "timestamp_sec": round(sec, 2),
                    "status": status,
                    "vehicle_detected": v_detected,
                    "vehicle_type": v_type,
                    "plate_detected": p_detected,
                    "plate_number": plate_num,
                    "ocr_confidence": ocr_conf,
                    "mongo_record_id": plate_record_id,
                    "mongo_image_id": plate_image_id,
                    "response_time_ms": elapsed_ms
                })
            else:
                print(f"[{processed_count:03d} | Frame {frame_idx:04d}] HTTP {resp.status_code}: {resp.text[:100]}")

        except Exception as e:
            print(f"[{processed_count:03d} | Frame {frame_idx:04d}] Exception: {e}")

    frame_idx += 1

cap.release()
total_time = time.time() - t_start

print("\n" + "=" * 80)
print("ANPR VIDEO TEST SUMMARY")
print("=" * 80)
print(f"Total Video Frames Processed: {processed_count}/{total_frames} (sampled step={FRAME_STEP})")
print(f"Total Test Time:               {total_time:.2f} seconds")
print(f"Processing Speed:              {processed_count / total_time:.2f} frames/sec")
print(f"Vehicles Detected:             {vehicles_found}/{processed_count}")
print(f"Plates Detected:               {plates_found}/{processed_count}")
print(f"MongoDB Records Created:       {mongo_records_created}")

if plates_detected_summary:
    print("\nUNIQUE PLATES RECOGNIZED:")
    print("-" * 40)
    for p_num, count in sorted(plates_detected_summary.items(), key=lambda x: x[1], reverse=True):
        print(f"  Plate: {p_num:<15} | Occurrences: {count}")
else:
    print("\nNo valid plates were recognized in the sample frames.")

out_json = "video_test_results.json"
with open(out_json, "w") as f:
    json.dump({
        "video_path": VIDEO_PATH,
        "test_timestamp": datetime.now().isoformat(),
        "summary": {
            "total_frames_sampled": processed_count,
            "vehicles_detected": vehicles_found,
            "plates_detected": plates_found,
            "mongo_records_created": mongo_records_created,
            "unique_plates": plates_detected_summary,
            "total_test_time_sec": round(total_time, 2)
        },
        "results": results
    }, f, indent=2)

print(f"\nDetailed results written to: {out_json}")
