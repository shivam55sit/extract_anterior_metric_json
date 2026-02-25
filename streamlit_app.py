import streamlit as st
import cv2
import numpy as np
import os
import json
from PIL import Image
from image_crop import crop_left_panel
from ocr import extract_values_at_positions

# Page Configuration
st.set_page_config(
    page_title="Cornea Front Analyzer",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a premium look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .stDownloadButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #28a745;
        color: white;
        font-weight: bold;
    }
    .status-box {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    h1 {
        color: #1e293b;
        font-family: 'Inter', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.title("👁️ Cornea Front Metric Extractor")
    st.markdown("Upload your Oculyzer or Pentacam maps to extract 'Cornea Front' metrics automatically.")

    with st.sidebar:
        st.header("Instructions")
        st.info("""
        1. Upload up to 4 eye map images.
        2. The app will automatically crop the **Cornea Front** section.
        3. OCR will extract **K1, K2, Astig, and Axis** values.
        4. Review the results and download the JSON data.
        """)
        
        st.divider()
        st.markdown("### Supported Devices")
        st.markdown("- Pentacam")
        st.markdown("- Oculyzer")

    # File Uploader
    uploaded_files = st.file_uploader("Choose eye map images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

    if uploaded_files:
        st.divider()
        
        # Results container
        all_results = []
        
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            with st.expander(f"Processing: {file_name}", expanded=True):
                col1, col2, col3 = st.columns([1, 1, 1])
                
                # Load image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)
                
                # Save temp file for processing modules
                temp_input_path = f"temp_input_{file_name}"
                temp_output_path = f"temp_crop_{file_name}"
                cv2.imwrite(temp_input_path, image)
                
                with col1:
                    st.markdown("**Original Image**")
                    st.image(uploaded_file, use_container_width=True)
                
                with st.spinner(f"Analyzing {file_name}..."):
                    # 1. Crop
                    try:
                        cropped_img = crop_left_panel(temp_input_path, temp_output_path)
                        
                        if cropped_img is not None:
                            with col2:
                                st.markdown("**Cornea Front Crop**")
                                st.image(temp_output_path, use_container_width=True)
                            
                            # 2. OCR
                            extracted_data = extract_values_at_positions(temp_output_path)
                            
                            with col3:
                                st.markdown("**Extracted Metrics**")
                                if extracted_data:
                                    st.json(extracted_data)
                                    all_results.append(extracted_data)
                                else:
                                    st.warning("No metrics found in this crop.")
                        else:
                            st.error("Failed to crop 'Cornea Front' section.")
                    except Exception as e:
                        st.error(f"Error processing image: {e}")
                    finally:
                        # Cleanup temp files
                        if os.path.exists(temp_input_path):
                            os.remove(temp_input_path)
                        if os.path.exists(temp_output_path):
                            # We keep it temporarily for st.image if not already displayed
                            pass 

        if all_results:
            st.divider()
            st.header("Results Summary")
            
            # If single image, output a flat dict; otherwise a list
            output_data = all_results[0] if len(all_results) == 1 else all_results
            final_json = json.dumps(output_data)
            st.code(final_json, language='json')
            
            # Download Button
            st.download_button(
                label="📥 Download Extracted Data (JSON)",
                data=final_json,
                file_name="cornea_metrics.json",
                mime="application/json"
            )
            
            # Cleanup all crop temp files at the end
            for uf in uploaded_files:
                path = f"temp_crop_{uf.name}"
                if os.path.exists(path):
                    os.remove(path)

    else:
        # Empty state
        st.info("Please upload one or more map images to begin.")
        
        # Display a sample or placeholder if needed
        # st.image("placeholder.png")

if __name__ == "__main__":
    main()
