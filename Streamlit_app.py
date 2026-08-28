# ============================================================
# PETRI-DISH COLONY ANALYSIS — STREAMLIT WEB APP
# ============================================================
# Wraps the original Colab colony-counting script into an
# interactive web app. Upload a Petri dish photo, get:
#   - Annotated image with colony count
#   - Full measurements table (+ CSV download)
#   - Summary report
#   - 5 analysis graphs
#
# Deploy free on Streamlit Community Cloud (see README.md).
# ============================================================

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend, required for servers
import matplotlib.pyplot as plt
import math
import io
import streamlit as st
from PIL import Image


def analyze_colony_image(pil_image):
    """
    Runs the full colony detection + analysis pipeline on an
    uploaded image and returns everything the UI needs.
    """

    img_rgb = np.array(pil_image.convert("RGB"))
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    height, width = img.shape[:2]

    # Optional resizing
    MAX_SIZE = 1600
    if max(height, width) > MAX_SIZE:
        scale_resize = MAX_SIZE / max(height, width)
        img = cv2.resize(img, None, fx=scale_resize, fy=scale_resize,
                          interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Detect Petri dish circle
    circles = cv2.HoughCircles(
        gray_blur, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=min(width, height) * 0.4,
        param1=100, param2=60,
        minRadius=int(min(width, height) * 0.30),
        maxRadius=int(min(width, height) * 0.55)
    )

    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        x, y, radius = max(circles, key=lambda c: c[2])
    else:
        x = img.shape[1] // 2
        y = img.shape[0] // 2
        radius = int(min(img.shape[1], img.shape[0]) * 0.45)

    dish_radius = int(radius * 0.94)
    dish_mask = np.zeros_like(gray)
    cv2.circle(dish_mask, (x, y), dish_radius, 255, -1)

    # Local contrast enhancement (color-independent)
    scale = radius / 200.0
    kernel_size = max(5, int(round(9 * scale)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    blackhat[dish_mask == 0] = 0

    dish_values = blackhat[dish_mask > 0]
    otsu_threshold, _ = cv2.threshold(
        dish_values.astype(np.uint8), 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    threshold = max(5, otsu_threshold * 0.95)

    colony_mask = np.zeros_like(gray)
    colony_mask[(blackhat >= threshold) & (dish_mask > 0)] = 255

    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    colony_mask = cv2.morphologyEx(colony_mask, cv2.MORPH_OPEN, small_kernel)
    colony_mask = cv2.morphologyEx(colony_mask, cv2.MORPH_CLOSE, small_kernel)

    contours, _ = cv2.findContours(colony_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Automatic size limits
    min_area = max(3, 4 * scale * scale)
    max_area = 120 * scale * scale
    min_radius = max(1.0, 1.2 * scale)
    max_radius = 8.0 * scale

    results = []
    annotated = img.copy()
    colony_number = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= 0:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = (4 * math.pi * area) / (perimeter ** 2)
        (cx_circle, cy_circle), enclosing_radius = cv2.minEnclosingCircle(contour)

        if area < min_area or area > max_area:
            continue
        if enclosing_radius < min_radius or enclosing_radius > max_radius:
            continue
        if circularity < 0.35:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        distance_from_center = math.sqrt((cx - x) ** 2 + (cy - y) ** 2)
        if distance_from_center > dish_radius:
            continue

        rr = max(3, int(enclosing_radius * 2.5))
        x1 = max(0, cx - rr)
        x2 = min(gray.shape[1], cx + rr)
        y1 = max(0, cy - rr)
        y2 = min(gray.shape[0], cy + rr)
        local_region = gray[y1:y2, x1:x2]

        local_mask = np.zeros(local_region.shape, dtype=np.uint8)
        local_cx = cx - x1
        local_cy = cy - y1
        cv2.circle(local_mask, (local_cx, local_cy), max(1, int(enclosing_radius)), 255, -1)
        colony_pixels = local_region[local_mask > 0]

        background_mask = np.ones(local_region.shape, dtype=np.uint8) * 255
        cv2.circle(background_mask, (local_cx, local_cy), max(2, int(enclosing_radius * 1.8)), 0, -1)
        background_pixels = local_region[background_mask > 0]

        if len(colony_pixels) > 0 and len(background_pixels) > 0:
            colony_intensity = float(np.mean(colony_pixels))
            background_intensity = float(np.mean(background_pixels))
            contrast = abs(background_intensity - colony_intensity)
        else:
            colony_intensity = 0
            background_intensity = 0
            contrast = 0

        diameter_pixels = 2 * enclosing_radius
        colony_number += 1

        results.append({
            "Colony_ID": colony_number,
            "X_pixel": cx,
            "Y_pixel": cy,
            "Area_pixels": area,
            "Diameter_pixels": diameter_pixels,
            "Circularity": circularity,
            "Colony_intensity": colony_intensity,
            "Background_intensity": background_intensity,
            "Contrast": contrast,
            "Enclosing_radius_pixels": enclosing_radius
        })

        cv2.circle(annotated, (cx, cy), max(2, int(enclosing_radius)), (0, 255, 0), 1)
        cv2.putText(annotated, str(colony_number), (cx + 3, cy - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)

    df = pd.DataFrame(results)

    if len(df) == 0:
        return None  # signals "no colonies found" to the UI

    # Summary stats
    total_colonies = len(df)
    mean_area = df["Area_pixels"].mean()
    median_area = df["Area_pixels"].median()
    mean_diameter = df["Diameter_pixels"].mean()
    mean_circularity = df["Circularity"].mean()
    mean_contrast = df["Contrast"].mean()
    total_colony_area = df["Area_pixels"].sum()
    petri_area_pixels = math.pi * (dish_radius ** 2)
    coverage_percent = (total_colony_area / petri_area_pixels) * 100
    density_per_10000_pixels = (total_colonies / petri_area_pixels) * 10000

    area_25 = df["Area_pixels"].quantile(0.25)
    area_75 = df["Area_pixels"].quantile(0.75)

    def classify_size(area):
        if area < area_25:
            return "Small"
        elif area > area_75:
            return "Large"
        return "Medium"

    def classify_shape(c):
        if c >= 0.80:
            return "Highly circular"
        elif c >= 0.60:
            return "Moderately circular"
        return "Irregular"

    df["Size_class"] = df["Area_pixels"].apply(classify_size)
    df["Shape_class"] = df["Circularity"].apply(classify_shape)

    # Annotated image
    cv2.circle(annotated, (x, y), dish_radius, (255, 0, 0), 2)
    cv2.putText(annotated, f"Colonies: {total_colonies}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    annotated_pil = Image.fromarray(annotated_rgb)

    # Graphs
    def fig_to_pil(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)

    fig1 = plt.figure(figsize=(6, 4))
    plt.hist(df["Area_pixels"], bins=20, edgecolor="black")
    plt.xlabel("Colony Area (pixels²)")
    plt.ylabel("Number of Colonies")
    plt.title("Colony Size Distribution")
    plt.tight_layout()
    size_dist_img = fig_to_pil(fig1)

    fig2 = plt.figure(figsize=(6, 4))
    plt.hist(df["Circularity"], bins=20, edgecolor="black")
    plt.xlabel("Circularity")
    plt.ylabel("Number of Colonies")
    plt.title("Colony Circularity Distribution")
    plt.tight_layout()
    circ_dist_img = fig_to_pil(fig2)

    fig3 = plt.figure(figsize=(6, 4))
    plt.hist(df["Contrast"], bins=20, edgecolor="black")
    plt.xlabel("Colony-to-Background Contrast")
    plt.ylabel("Number of Colonies")
    plt.title("Colony Contrast Distribution")
    plt.tight_layout()
    contrast_dist_img = fig_to_pil(fig3)

    fig4 = plt.figure(figsize=(5.5, 5.5))
    plt.scatter(df["X_pixel"], df["Y_pixel"], s=15, alpha=0.7)
    plt.gca().invert_yaxis()
    plt.xlabel("X position (pixels)")
    plt.ylabel("Y position (pixels)")
    plt.title("Spatial Distribution of Colonies")
    plt.axis("equal")
    plt.tight_layout()
    spatial_img = fig_to_pil(fig4)

    fig5 = plt.figure(figsize=(6, 4))
    plt.scatter(df["Area_pixels"], df["Circularity"], s=15, alpha=0.7)
    plt.xlabel("Colony Area (pixels²)")
    plt.ylabel("Circularity")
    plt.title("Colony Size vs Circularity")
    plt.tight_layout()
    size_vs_circ_img = fig_to_pil(fig5)

    report = f"""PETRI-DISH COLONY ANALYSIS REPORT
=================================

Petri dish center: ({x}, {y})
Petri dish radius: {dish_radius} pixels

TOTAL COLONIES: {total_colonies}

COLONY SIZE
-----------
Mean area:    {mean_area:.2f} pixels²
Median area:  {median_area:.2f} pixels²
Mean diameter:{mean_diameter:.2f} pixels

COLONY SHAPE
------------
Mean circularity: {mean_circularity:.3f}

COLONY CONTRAST
---------------
Mean contrast: {mean_contrast:.2f}

COLONY COVERAGE
---------------
{coverage_percent:.2f}%

COLONY DENSITY
--------------
{density_per_10000_pixels:.2f} colonies per 10,000 pixels²
"""

    return {
        "annotated": annotated_pil,
        "report": report,
        "df": df,
        "graphs": {
            "Colony Size Distribution": size_dist_img,
            "Circularity Distribution": circ_dist_img,
            "Contrast Distribution": contrast_dist_img,
            "Spatial Distribution": spatial_img,
            "Size vs Circularity": size_vs_circ_img,
        },
    }


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(page_title="Petri-Dish Colony Analysis", layout="wide")

st.title("🦠 Petri-Dish Colony Analysis")
st.write(
    "Upload a photo of a Petri dish and this tool will automatically detect "
    "colonies (regardless of color), count them, and measure their size, "
    "shape, contrast, coverage, and density."
)

uploaded_file = st.file_uploader("Upload Petri dish image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)

    with st.spinner("Analyzing colonies..."):
        result = analyze_colony_image(pil_image)

    if result is None:
        st.error(
            "No colonies detected in this image. Try a clearer, well-lit "
            "photo of the Petri dish, or a different image."
        )
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Annotated Result")
            st.image(result["annotated"], use_container_width=True)

        with col2:
            st.subheader("Summary Report")
            st.text(result["report"])

        df = result["df"]

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        report_bytes = result["report"].encode("utf-8")

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                "Download CSV",
                data=csv_bytes,
                file_name="colony_measurements.csv",
                mime="text/csv",
            )
        with dl_col2:
            st.download_button(
                "Download Report (.txt)",
                data=report_bytes,
                file_name="colony_analysis_report.txt",
                mime="text/plain",
            )

        st.subheader("Colony Measurements Table")
        st.dataframe(df, use_container_width=True)

        st.subheader("Analysis Graphs")
        graphs = result["graphs"]
        g_col1, g_col2 = st.columns(2)
        with g_col1:
            st.image(graphs["Colony Size Distribution"], caption="Colony Size Distribution", use_container_width=True)
            st.image(graphs["Contrast Distribution"], caption="Contrast Distribution", use_container_width=True)
        with g_col2:
            st.image(graphs["Circularity Distribution"], caption="Circularity Distribution", use_container_width=True)
            st.image(graphs["Spatial Distribution"], caption="Spatial Distribution", use_container_width=True)
        st.image(graphs["Size vs Circularity"], caption="Size vs Circularity", use_container_width=True)
else:
    st.info("Upload an image above to get started.")