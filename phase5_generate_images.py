"""
Phase 5 CCTV Robustness Testing - Image Generator
Generates 15 realistic test images across 3 categories:
  Cat1 (clear): 5 images
  Cat2 (medium): 5 images with angle/blur/smaller plate
  Cat3 (difficult): 5 images with low light/heavy blur/distance/multiple vehicles
"""
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import random

OUT_DIR = "phase5_test_images"
os.makedirs(OUT_DIR, exist_ok=True)

# Indian plates for each category
PLATES = {
    "cat1": [
        ("c1_MH12DE1433", "MH 12 DE 1433", "Sedan"),
        ("c1_KA05MH8899", "KA 05 MH 8899", "SUV"),
        ("c1_TS09AB1234", "TS 09 AB 1234", "Hatchback"),
        ("c1_DL3CAK7890", "DL 3C AK 7890", "Truck"),
        ("c1_GJ01AB9999", "GJ 01 AB 9999", "Car"),
    ],
    "cat2": [
        ("c2_TN09AB1001", "TN 09 AB 1001", "Car"),
        ("c2_HR26DQ5555", "HR 26 DQ 5555", "SUV"),
        ("c2_UP32GH4567", "UP 32 GH 4567", "Sedan"),
        ("c2_RJ14CD2020", "RJ 14 CD 2020", "Auto"),
        ("c2_MH04ER8001", "MH 04 ER 8001", "Car"),
    ],
    "cat3": [
        ("c3_KL07AB5050", "KL 07 AB 5050", "Car"),
        ("c3_WB06AC3311", "WB 06 AC 3311", "Truck"),
        ("c3_AP09EF7777", "AP 09 EF 7777", "SUV"),
        ("c3_OD02GH9900", "OD 02 GH 9900", "Car"),
        ("c3_BR01XY4321", "BR 01 XY 4321", "Car"),
    ]
}

VEHICLE_COLORS = {
    "Sedan":    [(45, 52, 54), (99, 110, 114), (178, 190, 195), (108, 92, 231)],
    "SUV":      [(30, 39, 46), (84, 153, 199), (39, 174, 96), (192, 57, 43)],
    "Hatchback":[(241, 196, 15), (236, 240, 241), (149, 165, 166), (52, 73, 94)],
    "Truck":    [(39, 60, 117), (96, 108, 56), (188, 143, 143), (128, 128, 0)],
    "Auto":     [(230, 126, 34), (241, 196, 15), (46, 204, 113), (52, 152, 219)],
    "Car":      [(231, 76, 60), (52, 73, 94), (189, 195, 199), (44, 62, 80)],
}

def draw_vehicle_scene(img_size=(640, 480), vehicle_type="Car", plate_text="MH 12 DE 1433",
                       small_plate=False, angle=0, brightness=1.0, blur_sigma=0,
                       extra_vehicles=False, bg_style="day"):
    """Render a synthetic vehicle scene with license plate."""
    W, H = img_size
    img = Image.new("RGB", (W, H))
    draw = ImageDraw.Draw(img)

    # --- Background ---
    if bg_style == "day":
        # Sky gradient
        for y in range(H // 2):
            t = y / (H // 2)
            r = int(135 + (200 - 135) * t)
            g = int(206 + (220 - 206) * t)
            b = int(235 + (235 - 235) * t)
            draw.line([(0, y), (W, y)], fill=(r, g, b))
        # Road
        for y in range(H // 2, H):
            t = (y - H // 2) / (H // 2)
            shade = int(80 + 30 * t)
            draw.line([(0, y), (W, y)], fill=(shade, shade, shade))
        # Road markings
        for i in range(5):
            x = W // 6 + i * W // 5
            draw.rectangle([(x - 5, H // 2 + 10), (x + 5, H // 2 + 40)], fill=(255, 255, 200))
    elif bg_style == "night":
        img.paste((10, 10, 20), [0, 0, W, H])
        # street lights
        for lx in [W//4, W//2, 3*W//4]:
            draw.ellipse([(lx-30, 5), (lx+30, 35)], fill=(255, 220, 100))
            draw.polygon([(lx, 35), (lx-5, H//2), (lx+5, H//2)], fill=(80, 80, 60))
        for y in range(H // 2, H):
            shade = int(20 + 15 * (y - H//2)/(H//2))
            draw.line([(0, y), (W, y)], fill=(shade, shade, shade))
    elif bg_style == "overcast":
        for y in range(H // 2):
            shade = int(160 + 30 * y/(H//2))
            draw.line([(0, y), (W, y)], fill=(shade, shade, shade+10))
        for y in range(H // 2, H):
            t = (y - H//2)/(H//2)
            shade = int(60 + 20*t)
            draw.line([(0, y), (W, y)], fill=(shade, shade, shade))

    # --- Extra vehicles (background) ---
    if extra_vehicles:
        for ex in [W//4 - 60, 3*W//4 + 20]:
            ew, eh = 80, 50
            ey = H // 2 + 20
            ec = random.choice([(100,100,100),(180,60,60),(60,60,180)])
            draw.rectangle([(ex, ey), (ex+ew, ey+eh)], fill=ec)
            draw.rectangle([(ex+10, ey-20), (ex+ew-10, ey+5)], fill=ec)

    # --- Primary vehicle ---
    vcolor = random.choice(VEHICLE_COLORS.get(vehicle_type, [(100,100,100)]))
    vx = W // 2 - 150
    vy = H // 2 + 10
    vw, vh = 300, 120

    if vehicle_type == "Truck":
        vw, vh = 320, 100
        draw.rectangle([(vx, vy), (vx+vw, vy+vh)], fill=vcolor)
        draw.rectangle([(vx+200, vy-60), (vx+vw, vy+10)], fill=vcolor)
    else:
        # Body
        draw.rectangle([(vx, vy), (vx+vw, vy+vh)], fill=vcolor)
        # Cabin
        roof_pts = [(vx+40, vy), (vx+80, vy-50), (vx+vw-80, vy-50), (vx+vw-40, vy)]
        draw.polygon(roof_pts, fill=vcolor)
        # Windows
        win_c = (180, 220, 255) if bg_style == "day" else (20, 40, 80)
        draw.polygon([(vx+50, vy-5), (vx+85, vy-42), (vx+160, vy-42), (vx+165, vy-5)], fill=win_c)
        draw.polygon([(vx+175, vy-5), (vx+175, vy-42), (vx+vw-85, vy-42), (vx+vw-50, vy-5)], fill=win_c)

    # Wheels
    for wx in [vx+40, vx+vw-40]:
        draw.ellipse([(wx-25, vy+vh-10), (wx+25, vy+vh+40)], fill=(30, 30, 30))
        draw.ellipse([(wx-12, vy+vh+3), (wx+12, vy+vh+27)], fill=(80, 80, 80))

    # Headlights
    light_c = (255, 255, 200) if bg_style == "night" else (200, 200, 150)
    draw.ellipse([(vx+10, vy+10), (vx+40, vy+35)], fill=light_c)
    draw.ellipse([(vx+vw-40, vy+10), (vx+vw-10, vy+35)], fill=(255, 80, 80))

    # --- License plate ---
    if small_plate:
        pw, ph = 70, 20
    else:
        pw, ph = 110, 30

    px = vx + vw // 2 - pw // 2
    py = vy + vh - ph - 5

    # Plate border & background
    draw.rectangle([(px-2, py-2), (px+pw+2, py+ph+2)], fill=(40,40,40))
    draw.rectangle([(px, py), (px+pw, py+ph)], fill=(255, 255, 255))

    # IND stripe
    ind_h = ph // 5
    draw.rectangle([(px, py), (px+pw, py+ind_h)], fill=(0, 0, 150))
    try:
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(7, ph//5))
        font_plate = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(9, ph-ind_h-4))
    except:
        font_small = ImageFont.load_default()
        font_plate = ImageFont.load_default()

    draw.text((px+2, py+1), "IND", fill=(255,255,255), font=font_small)
    draw.text((px+2, py+ind_h+2), plate_text, fill=(0,0,0), font=font_plate)

    # --- Apply angle distortion ---
    if angle != 0:
        img_arr = np.array(img)
        h_arr, w_arr = img_arr.shape[:2]
        pts1 = np.float32([[0,0],[w_arr,0],[0,h_arr],[w_arr,h_arr]])
        shift = int(angle * h_arr / 100)
        pts2 = np.float32([[shift,0],[w_arr-shift,0],[0,h_arr],[w_arr,h_arr]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img_arr = cv2.warpPerspective(img_arr, M, (w_arr, h_arr))
        img = Image.fromarray(img_arr)

    # --- Brightness ---
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)

    # --- Blur ---
    if blur_sigma > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_sigma))

    return img


def add_noise(img, amount=20):
    arr = np.array(img, dtype=np.int16)
    noise = np.random.randint(-amount, amount, arr.shape, dtype=np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


print("Generating Phase 5 test images...")

# Category 1: Clear
for fname, plate, vtype in PLATES["cat1"]:
    img = draw_vehicle_scene((640, 480), vehicle_type=vtype, plate_text=plate,
                             small_plate=False, angle=0, brightness=1.0,
                             blur_sigma=0, extra_vehicles=False, bg_style="day")
    img.save(os.path.join(OUT_DIR, f"{fname}.jpg"), quality=95)
    print(f"  [CAT1] {fname}.jpg saved")

# Category 2: Medium (angle, smaller plate, slight blur)
for i, (fname, plate, vtype) in enumerate(PLATES["cat2"]):
    angle = [5, 8, 10, 6, 12][i]
    blur = [0.8, 1.2, 1.0, 0.6, 1.5][i]
    small = [False, True, True, False, True][i]
    bg = ["day", "overcast", "day", "day", "overcast"][i]
    img = draw_vehicle_scene((640, 480), vehicle_type=vtype, plate_text=plate,
                             small_plate=small, angle=angle, brightness=0.85,
                             blur_sigma=blur, extra_vehicles=False, bg_style=bg)
    img = add_noise(img, 10)
    img.save(os.path.join(OUT_DIR, f"{fname}.jpg"), quality=85)
    print(f"  [CAT2] {fname}.jpg saved")

# Category 3: Difficult (low light, heavy blur, distance, multiple vehicles)
for i, (fname, plate, vtype) in enumerate(PLATES["cat3"]):
    angle = [15, 3, 20, 10, 8][i]
    blur = [2.5, 3.0, 2.0, 4.0, 2.8][i]
    bright = [0.4, 0.5, 0.35, 0.55, 0.45][i]
    small = [True, True, True, True, True][i]
    bg_list = ["night", "night", "overcast", "night", "overcast"]
    extra = [True, False, True, True, False][i]
    img = draw_vehicle_scene((640, 480), vehicle_type=vtype, plate_text=plate,
                             small_plate=small, angle=angle, brightness=bright,
                             blur_sigma=blur, extra_vehicles=extra, bg_style=bg_list[i])
    img = add_noise(img, 25)
    img.save(os.path.join(OUT_DIR, f"{fname}.jpg"), quality=75)
    print(f"  [CAT3] {fname}.jpg saved")

print(f"\nDone! 15 images saved to {OUT_DIR}/")
