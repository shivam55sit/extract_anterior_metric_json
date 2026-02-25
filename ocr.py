import cv2
import numpy as np
import easyocr
import re
import json
import os

# Initialize EasyOCR Reader (loads model once)
reader = easyocr.Reader(['en'])

# ============================================================================
# EXTRACTION CONFIGURATION
# ============================================================================
# Preferred method: Keyword search with proximity
KEYWORDS = {
    "K1": ["K1", "K 1"],
    "K2": ["K2", "K 2"],
    "Astig": ["Astig", "Cyl", "Asth"],
    "Axis": ["Axis", "Ax", "@", "Deg"]
}

# Fallback: Index-based mapping if keywords fail
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
        return float(match.group(1))
    return None


def extract_values_at_positions(image_path):
    """
    Extract OCR values using a robust strategy:
    1. Try to find labels (K1, K2, etc.) and get numeric values near them.
    2. Fallback to index-based mapping if labels are missing.
    3. Apply sanity checks for Astig/Axis ranges.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image with cv2.")

    # Run OCR
    ocr_results = extract_text_with_boxes(img)
    
    print(f"[DEBUG] OCR detected {len(ocr_results)} text elements")
    elements = []
    for i, (bbox, text, conf) in enumerate(ocr_results):
        y_center = np.mean([bbox[0][1], bbox[2][1]])
        x_center = np.mean([bbox[0][0], bbox[2][0]])
        elements.append({
            'index': i,
            'text': text.strip(),
            'conf': conf,
            'x': x_center,
            'y': y_center,
            'num': extract_numeric_value(text)
        })
        print(f"  [{i}] Y={y_center:6.1f} X={x_center:6.1f} '{text}' (conf={conf:.2f})")
    
    result = {}
    
    # --- Strategy 1: Keyword-based Search ---
    for key, aliases in KEYWORDS.items():
        # Find element containing the keyword
        for el in elements:
            text_upper = el['text'].upper()
            if any(alias.upper() in text_upper for alias in aliases):
                # We found a label. Now look for the nearest number.
                # Usually the number is in the same element or the next few elements
                if el['num'] is not None:
                    result[key] = el['num']
                    break
                
                # Check neighbors (within 5 indices and reasonable distance)
                best_neighbor = None
                min_dist = float('inf')
                
                for other in elements:
                    if other['num'] is not None and other['index'] != el['index']:
                        # Distance check (prefer right or down-right)
                        dx = other['x'] - el['x']
                        dy = other['y'] - el['y']
                        dist = np.sqrt(dx**2 + dy**2)
                        
                        # Heuristic: limit search area to reasonable proximity
                        if dist < 150 and dist < min_dist:
                            min_dist = dist
                            best_neighbor = other
                
                if best_neighbor:
                    result[key] = best_neighbor['num']
                    print(f"[STRATEGY 1] Found '{key}' via label '{el['text']}' -> {best_neighbor['num']}")
                    break

    # --- Strategy 2: Fallback to Index-based Mapping ---
    for idx, key in INDEX_MAPPING.items():
        if key not in result:
            if idx < len(elements):
                val = elements[idx]['num']
                if val is not None:
                    result[key] = val
                    print(f"[STRATEGY 2] Found '{key}' via index {idx} -> {val}")

    # --- Strategy 3: Sanity Check & Fix (Astig/Axis) ---
    if "Astig" in result and "Axis" in result:
        ast = result["Astig"]
        ax = result["Axis"]
        
        # Heuristic: Axis is usually an integer or > 10, Astig is usually < 10 (or negative)
        # If they look swapped (e.g., Astig=150.0, Axis=1.5), swap them back.
        if abs(ast) > abs(ax) and abs(ast) > 15 and abs(ax) < 15:
            print(f"[FIX] Swapping suspected swapped values: Astig={ast}, Axis={ax} -> Astig={ax}, Axis={ast}")
            result["Astig"], result["Axis"] = ax, ast
        
        # Ensure Axis is within 0-180 (common mapping error)
        if result["Axis"] > 180:
             # Sometimes OCR reads degree symbol as extra zero or something
             # Just a simple clamp or warning for now
             pass

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