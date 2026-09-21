"""
Phase 5 CCTV Robustness Benchmark
Sends each test image to POST /api/v1/detect/image and collects all evidence.
NO CODE CHANGES - observation only.
"""
import requests
import json
import os
import time
from datetime import datetime

API_URL = "http://localhost:8000/api/v1/detect/image"
IMG_DIR = "phase5_test_images"

TEST_PLAN = [
    # (filename, expected_plate, category, category_label)
    # --- Category 1: Clear ---
    ("c1_MH12DE1433.jpg", "MH12DE1433", 1, "Clear"),
    ("c1_KA05MH8899.jpg", "KA05MH8899", 1, "Clear"),
    ("c1_TS09AB1234.jpg", "TS09AB1234", 1, "Clear"),
    ("c1_DL3CAK7890.jpg", "DL3CAK7890", 1, "Clear"),
    ("c1_GJ01AB9999.jpg", "GJ01AB9999", 1, "Clear"),
    # --- Category 2: Medium ---
    ("c2_TN09AB1001.jpg", "TN09AB1001", 2, "Medium"),
    ("c2_HR26DQ5555.jpg", "HR26DQ5555", 2, "Medium"),
    ("c2_UP32GH4567.jpg", "UP32GH4567", 2, "Medium"),
    ("c2_RJ14CD2020.jpg", "RJ14CD2020", 2, "Medium"),
    ("c2_MH04ER8001.jpg", "MH04ER8001", 2, "Medium"),
    # --- Category 3: Difficult CCTV ---
    ("c3_KL07AB5050.jpg", "KL07AB5050", 3, "Difficult"),
    ("c3_WB06AC3311.jpg", "WB06AC3311", 3, "Difficult"),
    ("c3_AP09EF7777.jpg", "AP09EF7777", 3, "Difficult"),
    ("c3_OD02GH9900.jpg", "OD02GH9900", 3, "Difficult"),
    ("c3_BR01XY4321.jpg", "BR01XY4321", 3, "Difficult"),
]

results = []

print("=" * 80)
print("PHASE 5 - CCTV ROBUSTNESS BENCHMARK")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

for idx, (fname, expected, cat, cat_label) in enumerate(TEST_PLAN, 1):
    fpath = os.path.join(IMG_DIR, fname)
    print(f"\n[{idx:02d}/15] {fname}  |  Category: {cat} ({cat_label})  |  Expected: {expected}")
    print("-" * 60)

    if not os.path.exists(fpath):
        print(f"  ERROR: File not found: {fpath}")
        results.append({
            "idx": idx, "file": fname, "expected": expected,
            "category": cat, "cat_label": cat_label,
            "vehicle_detected": False, "vehicle_conf": None, "vehicle_class": None,
            "plate_detected": False, "plate_conf": None, "crop_w": None, "crop_h": None,
            "ocr_raw": None, "ocr_conf": None,
            "final_plate": None, "match": False, "mongo_id": None,
            "failure_stage": "FILE_NOT_FOUND", "response_time_ms": 0
        })
        continue

    with open(fpath, "rb") as f:
        file_bytes = f.read()

    t0 = time.time()
    try:
        resp = requests.post(
            API_URL,
            files={"file": (fname, file_bytes, "image/jpeg")},
            timeout=60
        )
        elapsed_ms = int((time.time() - t0) * 1000)
        status_code = resp.status_code

        try:
            data = resp.json()
        except Exception:
            data = {}

        print(f"  HTTP {status_code}  ({elapsed_ms} ms)")

    except requests.exceptions.RequestException as e:
        elapsed_ms = int((time.time() - t0) * 1000)
        print(f"  REQUEST ERROR: {e}")
        results.append({
            "idx": idx, "file": fname, "expected": expected,
            "category": cat, "cat_label": cat_label,
            "vehicle_detected": False, "vehicle_conf": None, "vehicle_class": None,
            "plate_detected": False, "plate_conf": None, "crop_w": None, "crop_h": None,
            "ocr_raw": None, "ocr_conf": None,
            "final_plate": None, "match": False, "mongo_id": None,
            "failure_stage": "REQUEST_ERROR", "response_time_ms": elapsed_ms
        })
        continue

    # ---- Parse response ----
    vehicles = data.get("vehicles", [])
    vehicle_detected = len(vehicles) > 0
    vehicle_conf = None
    vehicle_class = None
    plate_detected = False
    plate_conf = None
    crop_w = None
    crop_h = None
    ocr_raw = None
    ocr_conf = None
    final_plate = None
    mongo_id = data.get("record_id") or data.get("_id") or data.get("id")
    failure_stage = None

    if vehicle_detected:
        v = vehicles[0]
        vehicle_conf = round(v.get("confidence", 0), 4)
        vehicle_class = v.get("class_name") or v.get("class") or "unknown"
        plates_list = v.get("plates", [])

        print(f"  Vehicle: DETECTED | class={vehicle_class} | conf={vehicle_conf}")

        if plates_list:
            plate_detected = True
            p = plates_list[0]
            plate_conf = round(p.get("confidence", 0), 4)
            crop_info = p.get("crop_size") or p.get("crop_dimensions") or {}
            if isinstance(crop_info, dict):
                crop_w = crop_info.get("width") or crop_info.get("w")
                crop_h = crop_info.get("height") or crop_info.get("h")
            ocr_info = p.get("ocr") or p.get("text_info") or {}
            if isinstance(ocr_info, dict):
                ocr_raw = ocr_info.get("raw_text") or ocr_info.get("text")
                ocr_conf = round(ocr_info.get("confidence", 0), 4) if ocr_info.get("confidence") else None
            else:
                ocr_raw = p.get("ocr_text") or p.get("text")
                ocr_conf = round(p.get("ocr_confidence", 0), 4) if p.get("ocr_confidence") else None
            final_plate = p.get("normalized_plate") or p.get("plate_number") or ocr_raw

            print(f"  Plate:   DETECTED | conf={plate_conf} | crop={crop_w}x{crop_h}")
            print(f"  OCR:     raw='{ocr_raw}' | conf={ocr_conf}")
            print(f"  Normalized: '{final_plate}'")

            if not ocr_raw:
                failure_stage = "OCR"
        else:
            failure_stage = "PLATE_DETECTION"
            print(f"  Plate:   NOT DETECTED")
    else:
        # Try top-level fields in case response structure is flat
        if "plate_number" in data or "ocr_text" in data or "normalized_plate" in data:
            vehicle_detected = True
            vehicle_conf = data.get("vehicle_confidence")
            vehicle_class = data.get("vehicle_class", "detected")
            plate_detected = True
            plate_conf = data.get("plate_confidence")
            ocr_raw = data.get("ocr_text") or data.get("raw_text")
            ocr_conf = data.get("ocr_confidence")
            final_plate = data.get("normalized_plate") or data.get("plate_number") or ocr_raw
            crop_w = data.get("crop_width")
            crop_h = data.get("crop_height")
            mongo_id = data.get("record_id") or data.get("id")
            print(f"  Vehicle: DETECTED (flat response)")
            print(f"  Plate:   DETECTED | conf={plate_conf}")
            print(f"  OCR:     raw='{ocr_raw}' | conf={ocr_conf}")
            print(f"  Normalized: '{final_plate}'")
        else:
            failure_stage = "VEHICLE_DETECTION"
            print(f"  Vehicle: NOT DETECTED")
            print(f"  Full response: {json.dumps(data, indent=2)[:500]}")

    # Match check
    norm_final = (final_plate or "").upper().replace(" ", "").replace("-", "")
    norm_expected = expected.upper().replace(" ", "").replace("-", "")
    match = norm_final == norm_expected

    if not failure_stage and not match:
        failure_stage = "NORMALIZATION_MISMATCH"

    status_str = "✓ MATCH" if match else f"✗ MISMATCH (got '{norm_final}', expected '{norm_expected}')"
    print(f"  Match:   {status_str}")
    print(f"  MongoDB: record_id={mongo_id}")
    if failure_stage:
        print(f"  FAILURE STAGE: {failure_stage}")

    results.append({
        "idx": idx, "file": fname, "expected": expected,
        "category": cat, "cat_label": cat_label,
        "vehicle_detected": vehicle_detected, "vehicle_conf": vehicle_conf, "vehicle_class": vehicle_class,
        "plate_detected": plate_detected, "plate_conf": plate_conf, "crop_w": crop_w, "crop_h": crop_h,
        "ocr_raw": ocr_raw, "ocr_conf": ocr_conf,
        "final_plate": final_plate, "match": match, "mongo_id": str(mongo_id) if mongo_id else None,
        "failure_stage": failure_stage, "response_time_ms": elapsed_ms
    })

    time.sleep(0.5)  # rate-limit

# ============================================================
# SUMMARY
# ============================================================
print("\n")
print("=" * 80)
print("PHASE 5 RESULTS SUMMARY")
print("=" * 80)

# Per-category stats
for cat_id, cat_name in [(1, "Clear"), (2, "Medium"), (3, "Difficult")]:
    cat_results = [r for r in results if r["category"] == cat_id]
    vdet = sum(1 for r in cat_results if r["vehicle_detected"])
    pdet = sum(1 for r in cat_results if r["plate_detected"])
    matches = sum(1 for r in cat_results if r["match"])
    print(f"\nCategory {cat_id} ({cat_name}):")
    print(f"  Vehicle Detection: {vdet}/{len(cat_results)}")
    print(f"  Plate Detection:   {pdet}/{len(cat_results)}")
    print(f"  OCR Match:         {matches}/{len(cat_results)}")

total_vdet = sum(1 for r in results if r["vehicle_detected"])
total_pdet = sum(1 for r in results if r["plate_detected"])
total_match = sum(1 for r in results if r["match"])
print(f"\nOVERALL:")
print(f"  Vehicle Detection: {total_vdet}/15  ({100*total_vdet//15}%)")
print(f"  Plate Detection:   {total_pdet}/15  ({100*total_pdet//15}%)")
print(f"  OCR Accuracy:      {total_match}/15  ({100*total_match//15}%)")

# Failure classification
failures = [r for r in results if r["failure_stage"]]
if failures:
    print(f"\nFAILURE CLASSIFICATION ({len(failures)} failures):")
    from collections import Counter
    fc = Counter(r["failure_stage"] for r in failures)
    for stage, count in fc.most_common():
        print(f"  {stage}: {count}")

# Accuracy table
print("\n")
print("ACCURACY TABLE")
print(f"{'#':<4} {'File':<22} {'Cat':<10} {'Expected':<14} {'OCR Output':<16} {'Final Output':<14} {'Conf':<8} {'Status'}")
print("-" * 110)
for r in results:
    status = "✓ MATCH" if r["match"] else (f"✗ {r['failure_stage']}" if r["failure_stage"] else "✗ MISMATCH")
    conf_str = f"{r['ocr_conf']:.2f}" if r["ocr_conf"] else "N/A"
    ocr_str = (r["ocr_raw"] or "N/A")[:14]
    final_str = (r["final_plate"] or "N/A")[:12]
    print(f"{r['idx']:<4} {r['file']:<22} {r['cat_label']:<10} {r['expected']:<14} {ocr_str:<16} {final_str:<14} {conf_str:<8} {status}")

# Save JSON
out_json = "phase5_results.json"
with open(out_json, "w") as f:
    json.dump({"timestamp": datetime.now().isoformat(), "results": results}, f, indent=2)
print(f"\nFull results saved to: {out_json}")
