import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="🖼 Image Editor + Denoise", layout="wide")


def to_bgr(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.shape[-1] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def to_pil(bgr: np.ndarray) -> Image.Image:
    if bgr.ndim == 2:
        return Image.fromarray(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def spatial_denoise(img_bgr: np.ndarray, method: str, params: dict) -> np.ndarray:
    if method == "None":
        return img_bgr
    if method == "Median":
        k = params.get("ksize", 3)
        return cv2.medianBlur(img_bgr, k)
    if method == "Bilateral":
        d = params.get("d", 9)
        sigmaColor = params.get("sigmaColor", 75)
        sigmaSpace = params.get("sigmaSpace", 75)
        return cv2.bilateralFilter(img_bgr, d, sigmaColor, sigmaSpace)
    if method == "Fast NLM (colored)":
        if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        if img_bgr.dtype != np.uint8:
            img_bgr = cv2.normalize(img_bgr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        h = params.get("h", 10)
        hColor = params.get("hColor", 10)
        templateWindowSize = params.get("tws", 7)
        searchWindowSize = params.get("sws", 21)
        return cv2.fastNlMeansDenoisingColored(img_bgr, None, h, hColor, templateWindowSize, searchWindowSize)

    return img_bgr


def gaussian_lowpass_freq(img_bgr: np.ndarray, cutoff: float) -> np.ndarray:
    channels = cv2.split(img_bgr)
    out_channels = []

    for ch in channels:
        rows, cols = ch.shape
        crow, ccol = rows // 2, cols // 2

        x = np.arange(-ccol, ccol)
        y = np.arange(-crow, crow)
        X, Y = np.meshgrid(x, y)

        # adjust if off-by-one (odd/even dimension mismatch)
        if X.shape[0] != rows or X.shape[1] != cols:
            X = np.linspace(-ccol, ccol - 1, cols)
            Y = np.linspace(-crow, crow - 1, rows)
            X, Y = np.meshgrid(X, Y)

        # Gaussian mask
        D2 = X**2 + Y**2
        gaussian = np.exp(-D2 / (2 * (cutoff**2))).astype(np.float32)
        gaussian = np.fft.ifftshift(gaussian)

        # DFT
        f = np.float32(ch)
        dft = cv2.dft(f, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shifted = np.fft.fftshift(dft)

        # apply mask
        dft_shifted[:, :, 0] *= gaussian
        dft_shifted[:, :, 1] *= gaussian

        # inverse transform
        dft_ishift = np.fft.ifftshift(dft_shifted)
        img_back = cv2.idft(dft_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        # normalize result
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
        out_channels.append(np.uint8(img_back))

    return cv2.merge(out_channels)


def apply_filter(processed_img: np.ndarray, filter_name: str):
    if filter_name == "None":
        return processed_img
    if filter_name == "Grayscale":
        return cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    if filter_name == "Sepia":
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        out = cv2.transform(processed_img, sepia_filter)
        return np.clip(out, 0, 255).astype(np.uint8)
    if filter_name == "Invert Colors":
        return cv2.bitwise_not(processed_img)
    if filter_name == "Pencil Sketch":
        if processed_img.ndim == 3:
            gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
            _, sketch = cv2.pencilSketch(processed_img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
            return sketch
        else:
            return processed_img
    return processed_img

st.markdown("""
    <style>
        .main {
            background-color: #f6f5f3;
        }
        header {
            background-color: #e0dede;
            padding: 20px;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='background-color: #e0dede; padding: 20px; border-radius: 8px;'>
    <h1 style='text-align: center; color: #160148'>VisionCraft: Image Enhancement Studio🎨</h1>
    <p style='text-align: center; color: #160148'>Upload an image, crop, apply filters, smoothing/sharpening, and denoise using spatial or frequency-domain methods.</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"]) 

if uploaded_file:
    # Load image safely
    pil_image = Image.open(uploaded_file).convert('RGB')
    img_bgr = to_bgr(pil_image)

    st.sidebar.header("Crop & Quick Info")
    h, w = img_bgr.shape[:2]
    st.sidebar.write(f"Image dimensions: {w} x {h} (W x H)")

    # Crop controls
    x = st.sidebar.slider("Start X", 0, max(0, w - 1), 0)
    y = st.sidebar.slider("Start Y", 0, max(0, h - 1), 0)
    crop_width = st.sidebar.slider("Crop Width", 1, w - x, min(w - x, w // 2))
    crop_height = st.sidebar.slider("Crop Height", 1, h - y, min(h - y, h // 2))
    cropped_img = img_bgr[y:y+crop_height, x:x+crop_width]

    processed_img = cropped_img.copy()

    st.header("Controls")
    # Filters
    st.subheader("🎨 Filters")
    filter_choice = st.selectbox("Choose a filter", ["None", "Grayscale", "Sepia", "Invert Colors", "Pencil Sketch"]) 
    processed_img = apply_filter(processed_img, filter_choice)

    # Smoothing/Sharpening
    st.subheader("🔧 Smoothing & Sharpening")
    smooth_option = st.selectbox("Choose smoothing or sharpening", ["None", "Sharpen", "Gaussian Blur"]) 
    if smooth_option == "Sharpen":
        kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
        processed_img = cv2.filter2D(processed_img, -1, kernel)
    elif smooth_option == "Gaussian Blur":
        ksize = st.slider("Blur Kernel Size", 1, 25, 5, step=2)
        processed_img = cv2.GaussianBlur(processed_img, (ksize, ksize), 0)

    # Edge Detection
    st.subheader("🧠 Edge Detection")
    if st.checkbox("Apply Canny Edge Detection"):
        if len(processed_img.shape) == 3:
            gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = processed_img
        processed_img = cv2.Canny(gray, 100, 200)

    # Noise removal
    st.subheader("🧼 Noise Removal (Denoising)")
    denoise_mode = st.selectbox("Choose denoising domain", ["None", "Spatial", "Frequency Domain (Gaussian Low-pass)"])

    if denoise_mode == "Spatial":
        spatial_method = st.selectbox("Spatial method", ["None", "Median", "Bilateral", "Fast NLM (colored)"])
        if spatial_method == "Median":
            k = st.slider("Median kernel size (odd)", 1, 31, 3, step=2)
            processed_img = spatial_denoise(processed_img, "Median", {"ksize": k})
        elif spatial_method == "Bilateral":
            d = st.slider("Diameter (d)", 1, 50, 9)
            sc = st.slider("SigmaColor", 1, 200, 75)
            ss = st.slider("SigmaSpace", 1, 200, 75)
            processed_img = spatial_denoise(processed_img, "Bilateral", {"d": d, "sigmaColor": sc, "sigmaSpace": ss})
        elif spatial_method == "Fast NLM (colored)":
            h = st.slider("h (filter strength)", 1, 50, 10)
            hColor = st.slider("hColor (color strength)", 1, 50, 10)
            tws = st.slider("Template window size", 1, 15, 7)
            sws = st.slider("Search window size", 1, 35, 21)
            processed_img = spatial_denoise(processed_img, "Fast NLM (colored)", {"h": h, "hColor": hColor, "tws": tws, "sws": sws})

    elif denoise_mode == "Frequency Domain (Gaussian Low-pass)":
        cutoff = st.slider("Frequency cutoff (sigma)", 1.0, 200.0, 30.0)
        processed_img = gaussian_lowpass_freq(processed_img, cutoff)

    # Show results and download
    st.markdown("---")
    st.subheader("🖼 Original vs Processed Image")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("*Original (Cropped)*")
        st.image(to_pil(cropped_img), use_container_width=True)

    with col2:
        st.markdown("*Processed*")
        if processed_img.ndim == 2:
            st.image(processed_img, use_container_width=True, channels="GRAY")
        else:
            st.image(to_pil(processed_img), use_container_width=True)

    st.markdown("---")
    st.subheader("⬇ Download Your Image")
    if processed_img.ndim == 2:
        result = Image.fromarray(processed_img)
    else:
        result = to_pil(processed_img)

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button("📥 Click to Download", data=byte_im, file_name="edited_image.png", mime="image/png")

else:
    st.info("Upload an image to get started — supported formats: jpg, jpeg, png")