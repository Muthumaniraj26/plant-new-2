import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# =========================
# 1. IMAGE INPUT
# =========================
image_path = input("Enter leaf image path: ").strip().strip('"').strip("'")
if not os.path.exists(image_path):
    print("❌ File not found")
    exit()

img = cv2.resize(cv2.imread(image_path), (600, 600))
original = img.copy()

# =========================
# 2. ADVANCED LEAF SEGMENTATION
# =========================
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

leaf_mask = cv2.inRange(hsv, (20, 30, 20), (100, 255, 255))
leaf_mask = cv2.morphologyEx(
    leaf_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)
)
leaf_only = cv2.bitwise_and(img, img, mask=leaf_mask)

# =========================
# 3. SYNTHETIC NDVI
# =========================
b, g, r = cv2.split(img.astype(float))
nir_sim = g * 1.2
ndvi = (nir_sim - r) / (nir_sim + r + 1e-5)
ndvi_norm = cv2.normalize(ndvi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
ndvi_color = cv2.applyColorMap(ndvi_norm, cv2.COLORMAP_SUMMER)

# =========================
# 4. DISEASE DETECTION
# =========================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

yellow_mask = cv2.inRange(hsv, (15, 80, 80), (35, 255, 255))
_, black_mask = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)

disease_mask = cv2.bitwise_and(
    cv2.bitwise_or(yellow_mask, black_mask), leaf_mask
)

# =========================
# 5. METRICS & DIAGNOSIS
# =========================
leaf_px = np.count_nonzero(leaf_mask)
dis_px = np.count_nonzero(disease_mask > 20)
pct = (dis_px / leaf_px) * 100 if leaf_px > 0 else 0

is_fungal = np.count_nonzero(black_mask & leaf_mask) > np.count_nonzero(yellow_mask & leaf_mask)
diagnosis = "Fungal Pathogen (Black Spots)" if is_fungal else "Nutrient Deficiency / Chlorosis"
treatment = "Apply Copper-based Fungicide." if is_fungal else "Add Nitrogen-rich fertilizer."

severity = "CRITICAL" if pct > 25 else "WARNING" if pct > 5 else "STABLE"
severity_color = "#ff3333" if pct > 25 else "#ffaa00" if pct > 5 else "#00ff00"

# =========================
# 6. THERMAL MAP
# =========================
thermal = cv2.applyColorMap(
    cv2.normalize(disease_mask, None, 0, 255, cv2.NORM_MINMAX),
    cv2.COLORMAP_JET
)
thermal_overlay = cv2.addWeighted(leaf_only, 0.5, thermal, 0.5, 0)

# =========================
# 7. DASHBOARD LAYOUT
# =========================
fig = plt.figure(figsize=(18, 11), facecolor="#0d1117")
gs = fig.add_gridspec(3, 2, width_ratios=[1.3, 1])

fig.suptitle(
    f"AI Plant DOC Report — {pct:.1f}% Infection",
    color="white",
    fontsize=22,
    fontweight="bold"
)

# ---------- LEFT COLUMN : IMAGES ----------

ax1 = fig.add_subplot(gs[0, 0])
ax1.set_title("Infection Localization", color="cyan")
ax1.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
ax1.contour(disease_mask, colors="red", linewidths=0.6)
ax1.axis("off")

ax2 = fig.add_subplot(gs[1, 0])
ax2.set_title("Synthetic NDVI (Vegetation Health)", color="cyan")
ax2.imshow(cv2.cvtColor(ndvi_color, cv2.COLOR_BGR2RGB))
ax2.axis("off")

ax3 = fig.add_subplot(gs[2, 0])
ax3.set_title("Thermal Stress Profile", color="cyan")
ax3.imshow(cv2.cvtColor(thermal_overlay, cv2.COLOR_BGR2RGB))
ax3.axis("off")

# ---------- RIGHT COLUMN : CONTENT ----------

ax4 = fig.add_subplot(gs[:, 1])
ax4.axis("off")

# Severity bar
ax4.barh(["Infection Severity"], [pct], color=severity_color)
ax4.set_xlim(5, 100)
ax4.set_title("Disease Severity (%)", color="cyan")
ax4.tick_params(colors="white")

# Text report
report_text = f"""
DIAGNOSIS
----------------------------------------
{diagnosis}

SEVERITY STATUS
----------------------------------------
{severity}

INFECTED SURFACE
----------------------------------------
{pct:.2f}% of total leaf area

TREATMENT PLAN
----------------------------------------
{treatment}

NEXT ACTION
----------------------------------------
• Isolate affected plants
• Monitor again after 48 hours
• Repeat analysis after treatment
"""

ax4.text(
    0.02, -0.35,
    report_text,
    fontsize=13,
    color="white",
    family="monospace",
    bbox=dict(facecolor="black", edgecolor="cyan", boxstyle="round,pad=1.2")
)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
