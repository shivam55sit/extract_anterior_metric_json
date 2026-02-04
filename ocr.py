import cv2
import numpy as np
import easyocr
import re
import json
import os

# Initialize EasyOCR Reader (loads model once)
reader = easyocr.Reader(['en'])

# ============================================================================
# INDEX MAPPING - Extract values based on their position in the OCR result list
# ============================================================================
INDEX_MAPPING = {
    2: "K1",
    4: "K2",
    11: "Astig",
    10: "Axis"
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_text_with_boxes(image):
    """
    Run EasyOCR and return list of (bbox, text, confidence).
    detail=1 gives us bounding boxes.
    """
    return reader.readtext(image, detail=1)




def extract_numeric_value(text):
    """
    Extract the numeric value from OCR text.
    Handles patterns like:
    - "43.0 D"
    - "43.0D"
    - "136.4"
    - "-7.2"
    - "7.2D"
    """
    # Find all numbers (including negative and decimals)
    match = re.search(r'([+\-]?\d+\.?\d*)', text)
    if match:
        return match.group(1)
    return None


def extract_values_at_positions(image_path):
    """
    Extract OCR values based on specified indices from the list of detected elements.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image with cv2.")

    # Run OCR
    ocr_results = extract_text_with_boxes(img)
    
    print(f"[DEBUG] OCR detected {len(ocr_results)} text elements")
    print("\nAll detected elements (Index: Value):")
    for i, (bbox, text, conf) in enumerate(ocr_results):
        y_center = np.mean([bbox[0][1], bbox[2][1]])
        x_center = np.mean([bbox[0][0], bbox[2][0]])
        print(f"  [{i}] Y={y_center:6.1f} X={x_center:6.1f} '{text}' (conf={conf:.2f})")
    
    # Extract values based on INDEX_MAPPING
    result = {}
    
    for idx, key in INDEX_MAPPING.items():
        print(f"\n[DEBUG] Looking for '{key}' at index {idx}")
        
        if idx < len(ocr_results):
            bbox, text, conf = ocr_results[idx]
            # Extract numeric value from the text
            value = extract_numeric_value(text)
            
            if value:
                result[key] = value
                print(f"  ✓ Extracted value: {value} from '{text}'")
            else:
                print(f"  ✗ Could not extract numeric value from '{text}' at index {idx}")
        else:
            print(f"  ✗ Index {idx} is out of range (total elements: {len(ocr_results)})")
    
    return result


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    img_path = r"cropped_panel.jpg"

    result = extract_values_at_positions(img_path)

    # Pretty-print JSON
    json_output = json.dumps(result, indent=2)
    print("\n" + "="*50)
    print("EXTRACTED JSON")
    print("="*50)
    print(json_output)

    # Save to file
    out_json_path = "cornea_front.json"
    with open(out_json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_json_path}")