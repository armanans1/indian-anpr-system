import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import os

# Import your AI engine directly!
import main as anpr

# --- PAGE CONFIG & STYLING ---
st.set_page_config(page_title="ANPR System", page_icon="🚘", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    h1 { color: #1e3a8a; }
    .stAlert { border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("🚘 Automatic Number Plate Recognition (ANPR)")
st.write("Upload vehicle images below. The AI will locate the Indian license plate, correct OCR errors, and log it to the database.")

st.divider()

# --- LAYOUT ---
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("1. Image Input Area")
    # accept_multiple_files=True allows users to drag & drop 10 photos at once!
    uploaded_files = st.file_uploader("Upload Vehicle Images (JPG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        if st.button("🚀 Process Images", type="primary", use_container_width=True):
            
            # Loop through all uploaded photos
            for uploaded_file in uploaded_files:
                st.markdown(f"### Results for: `{uploaded_file.name}`")
                
                # Convert uploaded web image to OpenCV format
                image_pil = Image.open(uploaded_file).convert('RGB')
                image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

                with st.spinner(f"Analyzing {uploaded_file.name}..."):
                    # Pass it to your AI engine (this saves to CSV automatically)
                    annotated_cv, results = anpr.process_frame(image_cv, save_to_csv=True)
                
                # Convert back to RGB for Streamlit display
                annotated_rgb = cv2.cvtColor(annotated_cv, cv2.COLOR_BGR2RGB)
                
                # Display the image with the green box
                st.image(annotated_rgb, use_container_width=True)
                
                # Display success or failure messages
                if results:
                    for res in results:
                        st.success(f"✅ Plate Detected: **{res['plate']}** (Confidence: {res['confidence']:.2f})")
                else:
                    st.error("❌ No valid Indian license plate detected in this image.")
                
                st.markdown("---")

with col2:
    st.subheader("2. Live Database Log")
    
    if os.path.exists("results.csv"):
        # Load and display CSV
        df = pd.read_csv("results.csv")
        
        # Show metric summary
        st.metric(label="Total Plates Scanned", value=len(df))
        
        # Show dataframe
        st.dataframe(df.tail(15), use_container_width=True, hide_index=True)
        
        # Download button
        with open("results.csv", "rb") as file:
            st.download_button(
                label="📥 Download Full CSV Report",
                data=file,
                file_name="plate_database.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        if st.button("🗑️ Clear Database", use_container_width=True):
            os.remove("results.csv")
            st.rerun()
    else:
        st.info("Database is empty. Upload images to generate logs.")