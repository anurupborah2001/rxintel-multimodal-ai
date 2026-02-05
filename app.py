try:
    # load environment variables from .env file (requires `python-dotenv`)
    from dotenv import load_dotenv

    load_dotenv(override=True)
except ImportError:
    pass
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import easyocr
from streamlit_drawable_canvas import st_canvas
from datetime import datetime
from PIL import Image
import json
from chains.medicine_chain import get_medicine_info

# Enhanced Custom CSS for a more impressive UI
st.markdown(
    """
<style>
    .main {
        background-color: #f8f9fa;
        padding: 30px;
        height: 100vh;
        display: flex;
        flex-direction: column;
        font-family: 'Arial', sans-serif;
    }
    .stApp {
        background: linear-gradient(to bottom right, #e3f2fd, #bbdefb);
    }
    .main-title {
        color: #1a237e;
        text-align: center;
        font-size: 3.1rem;
        font-weight: 800;
        margin: 0.5rem 0 0.3rem;
        background: linear-gradient(90deg, #1e88e5, #42a5f5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
    }
    .title {
        color: #0d47a1;
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 20px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .subheader {
        color: #1565c0;
        font-size: 1.5em;
        margin-top: 20px;
    }
    .subtitle {
        text-align: center;
        color: #424242;
        font-size: 1.25rem;
        margin-bottom: 1.8rem;
        line-height: 1.5;
    }
    .feature-chip {
        display: inline-block;
        background: #e3f2fd;
        color: #1565c0;
        padding: 0.4em 1em;
        border-radius: 2em;
        font-size: 0.95rem;
        margin: 0.4em 0.5em;
        border: 1px solid #bbdefb;
        font-weight: 500;
    }
    .info-box {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .stButton > button {
        background-color: #1e88e5;
        color: white;
        border-radius: 25px;
        padding: 10px 20px;
        font-weight: bold;
        transition: background-color 0.3s ease, transform 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #1565c0;
        transform: scale(1.05);
    }
    .stTextInput > div > div > input {
        border-radius: 25px;
        padding: 12px 20px;
        border: 1px solid #ced4da;
        font-size: 1em;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
    }
    .stFileUploader {
        border: 2px dashed #1e88e5;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        transition: border-color 0.3s ease;
    }
    .stFileUploader:hover {
        border-color: #1565c0;
    }
    .stTable {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
    .llm-table {
        width: 100%;
        border-collapse: collapse;
        background-color: #ffffff;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .llm-table th {
        background-color: #e3f2fd;
        color: #0d47a1;
        font-weight: bold;
        text-align: left;
        padding: 12px 16px;
        border-bottom: 2px solid #dee2e6;
    }
    .llm-table td {
        padding: 12px 16px;
        border-bottom: 1px solid #dee2e6;
        vertical-align: top;
    }
    .llm-bubble {
        background-color: #e6ecf0;
        color: #333;
        border-radius: 16px;
        padding: 12px 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        word-wrap: break-word;
        white-space: pre-wrap;
    }
    .llm-timestamp {
        font-size: 0.75em;
        color: #888;
        opacity: 0.7;
        text-align: right;
        margin-top: 5px;
    }
    .sidebar .stMarkdown {
        color: #333;
        line-height: 1.6;
    }
    .sidebar h3 {
        color: #0d47a1;
        margin-bottom: 10px;
    }
    .sidebar ul {
        list-style-type: disc;
        padding-left: 20px;
    }
    @media (max-width: 600px) {
        .title {
            font-size: 2em;
        }
        .subheader {
            font-size: 1.2em;
        }
        .llm-table th, .llm-table td {
            padding: 8px;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)

# st.markdown('<div class="title">Pharmacy Receipt Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">RxIntel</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI-Powered Prescription & Side-Effect Analysis using Large Language Models</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
<div style="text-align:center; margin:1.6rem 0 2.2rem;">
    <span class="feature-chip">YOLOv12 Detection</span>
    <span class="feature-chip">EasyOCR Extraction</span>
    <span class="feature-chip">LLM-powered Insights</span>
    <span class="feature-chip">Drug Safety Information</span>
</div>
""",
    unsafe_allow_html=True,
)

# Sidebar with more information
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/pill.png", width=80)
    st.markdown("## RxIntel 🧬")
    st.markdown("**Smarter Medication Understanding**")
    st.markdown("---")

    st.markdown("### About This App")

    st.markdown(
        """
    RxIntel helps you quickly understand your medicines by analyzing pharmacy receipts, medicine strips or packaging using modern AI:

    • Photograph or upload an image
    • Automatically finds medicine names with **YOLOv12** object detection
    • Extracts text accurately using **EasyOCR**
    • Delivers structured, readable information powered by large language models:
      - What the medicine is used for
      - Typical dosages & forms
      - Important **warnings** & **contraindications**
      - Common **side effects**
      - Key **precautions**
      - Main active ingredients

    **Ideal for:**
    - Understanding new prescriptions the same day
    - Checking side effects before starting treatment
    - Quick reference during pharmacy counselling
    - Learning about unfamiliar medications
    """
    )

    st.markdown("### Capabilities & Important Notes")
    with st.expander("Show details", expanded=False):
        st.markdown(
            """
        **Works well:**
        • Clear photos of receipts / strips / boxes
        • Printed English text (best results)
        • Session history of recent lookups

        **Limitations – please be aware:**
        • Not a substitute for doctor or pharmacist advice
        • LLM knowledge may not reflect very recent changes
        • Accuracy depends strongly on image quality
        • No drug–drug interaction checking (planned feature)
        • Limited performance on handwritten text or very small fonts
        """
        )

    st.markdown("### Built With")
    st.markdown(
        """
    • Ultralytics YOLOv12
    • EasyOCR
    • LangChain + OpenAI (GPT-4o class models)
    • Streamlit
    """
    )

    st.markdown("---")
    st.caption("Made with care to help people better understand their medicines")
    st.caption("v0.9.2 • February 2025")
    st.markdown("Developed with ❤️ by Anurup Borah")

# Initialize session state
if "extracted_info" not in st.session_state:
    st.session_state["extracted_info"] = []
if "llm_responses" not in st.session_state:
    st.session_state["llm_responses"] = []
if "timestamp_history" not in st.session_state:
    st.session_state["timestamp_history"] = []
if "camera_active" not in st.session_state:
    st.session_state["camera_active"] = False
if "captured_image" not in st.session_state:
    st.session_state["captured_image"] = None

# Load YOLOv12 model
model = YOLO("./train-runs/v5/weights/best.pt")

# Initialize EasyOCR
reader = easyocr.Reader(["en"])


def extract_text_from_box(image, x, y, w, h):
    roi = image[int(y) : int(y + h), int(x) : int(x + w)]
    text = reader.readtext(roi, detail=0)
    return " ".join(text) if text else ""


# Image input
st.markdown(
    '<div class="subheader">Upload or Capture Receipt</div>', unsafe_allow_html=True
)
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Upload Receipt", type=["jpg", "png", "jpeg"])
with col2:
    if st.button("Open Camera", key="open_camera"):
        st.session_state["camera_active"] = True
    if st.session_state["camera_active"]:
        img_file = st.camera_input(
            "Take a photo of the receipt/medicine", key="camera_input"
        )
        if img_file:
            # Store raw bytes to avoid consumption
            st.session_state["captured_image"] = img_file.getvalue()
            # Convert to CV2 format for processing
            file_bytes = np.asarray(
                bytearray(st.session_state["captured_image"]), dtype=np.uint8
            )
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is None:
                st.error("Failed to decode the captured image. Please try again.")
            else:
                h, w, _ = image.shape
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
                # Drawable canvas for bounding box
                st.write("Draw bounding boxes around text regions ⬛")
                canvas_result = st_canvas(
                    fill_color="rgba(0,0,0,0)",  # Transparent fill
                    stroke_color="red",
                    stroke_width=2,
                    background_image=pil_image,
                    update_streamlit=True,
                    height=h,
                    width=w,
                    drawing_mode="rect",
                    key="canvas",
                )
                # Extract & Send to LLM button
                if st.button("Extract & Send to LLM"):
                    extracted_texts = []
                    if canvas_result.json_data is not None:
                        for obj in canvas_result.json_data["objects"]:
                            if obj["type"] == "rect":
                                left = int(obj["left"])
                                top = int(obj["top"])
                                width = int(obj["width"])
                                height = int(obj["height"])
                                # Crop the region
                                crop = image[top : top + height, left : left + width]
                                # OCR on cropped image
                                result = reader.readtext(crop)
                                text = " ".join([res[1] for res in result])
                                extracted_texts.append(text)
                    if extracted_texts:
                        st.success("📝 Extracted Texts:")
                        for t in extracted_texts:
                            st.write(f"- {t}")
                        # Send to LLM using the first extracted text as medicine name
                        medicine_name = extracted_texts[0] if extracted_texts else ""
                        if medicine_name:
                            with st.spinner("Fetching medicine information..."):
                                med_info = get_medicine_info(medicine_name)
                                st.session_state["llm_responses"].append(med_info)
                                st.session_state["timestamp_history"].append(
                                    datetime.now().strftime("%H:%M")
                                )
                                st.session_state["extracted_info"].append(
                                    {
                                        "medicine_name": medicine_name,
                                        "medicine_info": med_info,
                                    }
                                )
                    else:
                        st.warning("⚠️ No bounding boxes drawn or no text found.")
                    st.session_state["camera_active"] = False

# Manual medicine name input
st.markdown(
    '<div class="subheader">Manually Enter Medicine Name</div>', unsafe_allow_html=True
)
with st.form(key="medicine_form"):
    manual_medicine_name = st.text_input(
        "Enter Medicine Name", placeholder="e.g., Paracetamol"
    )
    submit_button = st.form_submit_button("Search Medicine")
    if submit_button and manual_medicine_name:
        with st.spinner("Fetching medicine information..."):
            med_info = get_medicine_info(manual_medicine_name)
            st.session_state["llm_responses"].append(med_info)
            st.session_state["timestamp_history"].append(
                datetime.now().strftime("%H:%M")
            )

# Process uploaded image or captured image outside camera workflow
image = None
is_camera_image = False
if uploaded_file is not None:
    image = cv2.imdecode(
        np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR
    )
elif st.session_state["captured_image"] is not None and not st.session_state.get(
    "camera_active", False
):
    # Process captured image only if not currently in camera mode
    file_bytes = np.asarray(
        bytearray(st.session_state["captured_image"]), dtype=np.uint8
    )
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    is_camera_image = True

if (
    image is not None and not is_camera_image
):  # Process only if not already handled by camera
    st.image(image, caption="Processed Receipt", width="stretch")
    extracted_info = {}

    # For uploaded images, use YOLOv12
    results = model(image)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    class_names = model.names
    detected_classes = [class_names[int(cls)] for cls in classes]
    extracted_info = {}
    for box, cls in zip(boxes, detected_classes):
        x1, y1, x2, y2 = map(int, box)
        roi = image[y1:y2, x1:x2]
        text = reader.readtext(roi, detail=0)
        extracted_info[cls] = " ".join(text) if text else ""
    medicine_name = extracted_info.get("medicine_name", "")
    if medicine_name:
        st.markdown(
            f'<div class="success-message">Detected Medicine: {medicine_name}</div>',
            unsafe_allow_html=True,
        )
        with st.spinner("Fetching medicine information..."):
            med_info = get_medicine_info(medicine_name)
            extracted_info["medicine_info"] = med_info
            st.session_state["llm_responses"].append(med_info)
            st.session_state["timestamp_history"].append(
                datetime.now().strftime("%H:%M")
            )
    else:
        st.markdown(
            '<div class="warning-message">Medication name not detected. Draw a bounding box around it or enter manually.</div>',
            unsafe_allow_html=True,
        )
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=3,
            stroke_color="#ff4500",
            background_image=pil_image,
            height=image.shape[0],
            width=image.shape[1],
            drawing_mode="rect",
            key="canvas_upload",
        )
        if canvas_result.json_data is not None and canvas_result.json_data["objects"]:
            rect = canvas_result.json_data["objects"][0]
            x, y, w, h = rect["left"], rect["top"], rect["width"], rect["height"]
            medicine_name = extract_text_from_box(image, x, y, w, h)
            if medicine_name:
                extracted_info["medicine_name"] = medicine_name
                st.markdown(
                    f'<div class="success-message">Extracted Medicine from Bounding Box: {medicine_name}</div>',
                    unsafe_allow_html=True,
                )
                with st.spinner("Fetching medicine information..."):
                    med_info = get_medicine_info(medicine_name)
                    extracted_info["medicine_info"] = med_info
                    st.session_state["llm_responses"].append(med_info)
                    st.session_state["timestamp_history"].append(
                        datetime.now().strftime("%H:%M")
                    )

    # Display extracted info in table
    if extracted_info:
        # df = pd.DataFrame([extracted_info])
        # st.table(df)
        st.session_state["extracted_info"].append(extracted_info)

# # Display LLM responses in a nicely formatted table
# if st.session_state["llm_responses"]:
#     st.markdown('<div class="subheader">Medicine Information</div>', unsafe_allow_html=True)

#     responses = [
#         html.escape(str(resp))  # escape HTML characters
#         .replace("```", "")
#         .replace("<", "&lt;")
#         .replace(">", "&gt;")
#         for resp in st.session_state["llm_responses"]
#     ]
#     timestamps = st.session_state["timestamp_history"]

#     html_table = """
#     <table style="width:100%; border-collapse: collapse;">
#         <thead>
#             <tr style="background-color:#e3f2fd; color:#0d47a1;">
#                 <th style="padding:12px; text-align:left;">Medicine Details</th>
#                 <th style="padding:12px; text-align:left;">Time</th>
#             </tr>
#         </thead>
#         <tbody>
#     """
#     for resp, ts in zip(responses, timestamps):
#         html_table += f"""
#             <tr style="border-bottom:1px solid #dee2e6;">
#                 <td>
#                     <div style="
#                         background-color:#f1f3f5;
#                         padding:12px;
#                         border-radius:12px;
#                         word-wrap:break-word;
#                         white-space:pre-wrap;
#                         max-height:200px;
#                         overflow:auto;
#                     ">{resp}</div>
#                 </td>
#                 <td style="vertical-align:top; padding:12px;">
#                     <div style="color:#888; font-size:0.85em;">{ts}</div>
#                 </td>
#             </tr>
#         """
#     html_table += "</tbody></table>"

#     st.markdown(html_table, unsafe_allow_html=True)

# Display LLM responses as larger, readable chat bubbles
if st.session_state["llm_responses"]:
    st.markdown(
        '<div class="subheader">Medicine Information</div>', unsafe_allow_html=True
    )

    for resp, ts in zip(
        st.session_state["llm_responses"], st.session_state["timestamp_history"]
    ):
        try:
            # Normalize to dict
            if isinstance(resp, str):
                cleaned = resp.replace("```json", "").replace("```", "").strip()
                med_info = json.loads(cleaned)
            elif hasattr(resp, "dict"):  # Pydantic model like DrugInfo
                med_info = resp.dict()
            else:
                med_info = resp  # already a dict
        except Exception as e:
            st.error(f"⚠️ Failed to parse JSON: {e}")
            st.code(resp, language="json")
            continue

        # Display as nicely formatted sections
        st.markdown(f"**🕒 Retrieved at {ts}**")
        st.markdown(
            f"### 💊 {med_info.get('normalized_name', med_info.get('query', 'Unknown'))}"
        )

        st.write(f"**RxCUI:** {med_info.get('rxcui', '-')}")
        st.write(f"**Composition:** {', '.join(med_info.get('composition', []))}")

        st.markdown("#### 📌 Indications")
        st.info(med_info.get("indications", "Not available"))

        st.markdown("#### 💊 Dosage")
        st.success(med_info.get("dosage", "Not available"))

        st.markdown("#### ⚠️ Warnings")
        st.warning(med_info.get("warnings", "Not available"))

        st.markdown("#### 🚫 Contraindications")
        st.error(med_info.get("contraindications", "Not available"))

        st.markdown("#### 🤕 Side Effects")
        st.write(med_info.get("side_effects", "Not available"))

        st.markdown("#### 🛡 Precautions")
        st.write(med_info.get("precautions", "Not available"))

        st.markdown("#### 📚 Sources")
        sources = med_info.get("sources", [])
        if sources:
            for src in sources:
                st.markdown(f"- [{src}]({src})")
        else:
            st.write("No sources available.")
