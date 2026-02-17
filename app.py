
import streamlit as st
import sys
import os
import pandas as pd
from pathlib import Path

# Add root to sys.path
sys.path.append(os.getcwd())

from src.inference import EmbeddingInference

st.set_page_config(page_title="Product Model Demo", page_icon="🛍️", layout="wide")

st.title("🛍️ Product Model Identity Verification Demo")
st.markdown("""
This demo showcases the **Product Model's** ability to verify if two product listings represent the same physical item.
Key use case: **Matching Scale A (e.g., Your Catalog) vs Site B (e.g., Competitor/Marketplace)**.
""")

# Load Model
@st.cache_resource
def load_model():
    model_path = "."
    if not os.path.exists("pytorch_model.bin"):
        return None
    return EmbeddingInference.from_pretrained(model_path)

model = load_model()

if not model:
    st.error("❌ Model not found in `models/product`. Please train the model first.")
    st.stop()

st.success(f"✅ Product Model Loaded! (Vocab: {len(model.tokenizer.word_to_id)}, Dim: {model.model.d_model})")

# Settings
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Match Threshold", 0.5, 1.0, 0.82, 0.01, help="Score above which products are considered a match.")

# Data Input
st.subheader("1. Input Data")

# Session state for text areas (widget keys)
if 'txt_a' not in st.session_state:
    st.session_state.txt_a = """Apple iPhone 14 128GB Midnight
Samsung Galaxy S23 Ultra 256GB Black
Sony WH-1000XM5 Wireless Headphones
Nintendo Switch OLED White
Logitech MX Master 3S Performance Mouse
Nike Air Force 1 '07 White
Dyson V15 Detect Vacuum"""

if 'txt_b' not in st.session_state:
    st.session_state.txt_b = """iPhone 14 (128GB) - Midnight Black
Samsung S23 Ultra 5G (256GB Storage)
Sony Noise Cancelling Headphones WH1000XM5
Nintendo Switch Console - OLED Model
Logitech Mouse MX Master 3S
Nike Men's Air Force 1 Sneakers
Dyson V15 Detect Cordless Vacuum Cleaner
Apple iPhone 13 128GB
Samsung Galaxy S22 Ultra
Sony WH-1000XM4
Nintendo Switch Lite"""

# Button to load large dataset
if st.button("📚 Load Large Benchmark Dataset (100+ items)"):
    # Generate large dataset
    base_products = [
        ("iPhone 14 Pro 128GB Space Black", "Apple iPhone 14 Pro (128 GB) - Space Black", "iPhone 14 Pro Max 128GB"),
        ("Samsung Galaxy S23 Ultra 512GB", "Samsung S23 Ultra 5G (512GB Storage)", "Samsung Galaxy S23 512GB"),
        ("Sony WH-1000XM5 Headphones", "Sony Noise Cancelling Wireless Headphones WH1000XM5", "Sony WH-1000XM4 Headphones"),
        ("MacBook Air M2 13-inch 256GB", "Apple MacBook Air Laptop: M2 chip, 13.6-inch, 256GB", "MacBook Pro M2 13-inch"),
        ("Dyson V15 Detect Vacuum", "Dyson V15 Detect Cordless Vacuum Cleaner", "Dyson V12 Detect Slim"),
        ("Logitech MX Master 3S", "Logitech Master Series MX 3S Mouse", "Logitech MX Master 3"),
        ("Kindle Paperwhite 16GB", "Amazon Kindle Paperwhite (16 GB) - 6.8 display", "Kindle Paperwhite 8GB"),
        ("Nintendo Switch OLED White", "Nintendo Switch – OLED Model w/ White Joy-Con", "Nintendo Switch Lite Blue"),
        ("PlayStation 5 Console", "Sony PS5 Console Disc Edition", "PlayStation 5 Digital Edition"),
        ("Xbox Series X", "Microsoft Xbox Series X 1TB Console", "Xbox Series S"),
        ("AirPos Pro 2nd Gen", "Apple AirPods Pro (2nd Generation) with MagSafe", "Apple AirPods 3rd Gen"),
        ("Fitbit Charge 6", "Fitbit Charge 6 Fitness Tracker with Google apps", "Fitbit Charge 5"),
        ("Garmin Forerunner 265", "Garmin Forerunner 265 Running Smartwatch", "Garmin Forerunner 965"),
        ("Yeti Rambler 20oz Tumbler", "YETI Rambler 20 oz Stainless Steel Vacuum Insulated", "Yeti Rambler 30oz"),
        ("Stanley Quencher H2.0 40oz", "Stanley The Quencher H2.0 FlowState Tumbler 40oz", "Stanley IceFlow Flip Straw"),
        ("Canon EOS R6 Mark II", "Canon Mirrorless Camera EOS R6 Mark II Body", "Canon EOS R5 Body"),
        ("Nikon Z6 II Body", "Nikon Z 6II FX-Format Mirrorless Camera", "Nikon Z7 II Body"),
        ("DJI Mini 3 Pro", "DJI Mini 3 Pro (DJI RC)", "DJI Mini 2 SE"),
        ("GoPro HERO11 Black", "GoPro HERO11 Black - Waterproof Action Camera", "GoPro HERO10 Black"),
        ("Razer DeathAdder V3 Pro", "Razer DeathAdder V3 Pro Wireless Gaming Mouse", "Razer Viper V2 Pro"),
        ("Keychron K2 Pro Keyboard", "Keychron K2 Pro QMK/VIA Wireless Mechanical Keyboard", "Keychron K2 Version 2"),
        ("Herman Miller Aeron Chair", "Herman Miller Aeron Ergonomic Office Chair", "Herman Miller Mirra 2"),
        ("Instant Pot Duo Plus 6qt", "Instant Pot Duo Plus 9-in-1 Electric Pressure Cooker", "Instant Pot Duo 7-in-1"),
        ("Ninja AF101 Air Fryer", "Ninja AF101 Air Fryer that Crisps, Roasts", "Ninja AF161 Max XL"),
        ("Vitamix 5200 Blender", "Vitamix 5200 Blender Professional-Grade", "Vitamix E310 Explorean"),
        ("Roomba j7+ Vacuum", "iRobot Roomba j7+ (7550) Self-Emptying Robot Vacuum", "Roomba i3+ EVO"),
        ("Sonos Arc Soundbar", "Sonos Arc - The Premium Smart Soundbar", "Sonos Beam Gen 2"),
        ("Bose QuietComfort 45", "Bose QuietComfort 45 Bluetooth Wireless Noise Cancelling", "Bose QuietComfort Earbuds II"),
        ("iPad Air 5th Gen 64GB", "Apple iPad Air (5th Generation): M1 chip, 64GB", "iPad 10th Gen 64GB"),
        ("Samsung T7 Shield 1TB", "Samsung T7 Shield 1TB Portable SSD", "Samsung T7 Touch 1TB"),
        ("SanDisk Extreme 2TB SSD", "SanDisk 2TB Extreme Portable SSD", "SanDisk Extreme Pro 2TB"),
        ("LG C3 OLED TV 65-inch", "LG 65-Inch Class C3 Series OLED evo 4K", "LG B3 OLED TV 65-inch"),
        ("Samsung QN90C 55-inch", "Samsung 55-Inch Class Neo QLED 4K QN90C", "Samsung QN85C 55-inch"),
        ("Google Pixel 7 Pro 128GB", "Google Pixel 7 Pro - 5G Android Phone 128GB", "Google Pixel 7 128GB"),
        ("OnePlus 11 5G 16GB RAM", "OnePlus 11 5G | 16GB RAM+256GB", "OnePlus 10T 5G"),
        ("Asus ROG Zephyrus G14", "ASUS Rogers Zephyrus G14 14” 165Hz Gaming Laptop", "Asus TUF Gaming F15"),
        ("Dell XPS 15 9530", "Dell XPS 15 Laptop, 13th Gen Intel Core", "Dell Inspiron 16 Plus"),
        ("Lenovo ThinkPad X1 Carbon Gen 11", "Lenovo ThinkPad X1 Carbon Gen 11 14 inch", "Lenovo ThinkPad T14s"),
        ("HP Spectre x360 14", "HP Spectre x360 2-in-1 Laptop 13.5t", "HP Envy x360 15"),
        ("Microsoft Surface Pro 9", "Microsoft Surface Pro 9 (2022), 13 2-in-1", "Microsoft Surface Laptop 5"),
    ]

    a_list = []
    b_list = []
    
    # 1. Add core pairs
    for a, b, distractor in base_products:
        a_list.append(a)
        b_list.append(b)
        b_list.append(distractor)

    # 2. Add algorithmic filler (increase to 35 iterations for >100 total)
    for i in range(35):
        a_list.append(f"Generic Widget Model X-{i+100} Pro")
        b_list.append(f"Generic Widget Series X {i+100} Professional Edition")
        b_list.append(f"Generic Widget Model X-{i+100} Lite") # Distractor
        
        a_list.append(f"Industrial Part #44-A{i}")
        b_list.append(f"Genuine Industrial Part Number 44-A{i} Replacement")
        b_list.append(f"Industrial Part #44-B{i}") # Distractor

    import random
    random.shuffle(b_list)
    
    # Update specific keys used by text_area to ensure UI refresh
    st.session_state.txt_a = "\n".join(a_list)
    st.session_state.txt_b = "\n".join(b_list)
    
    # Keep backing val updated too
    st.session_state.site_a_val = st.session_state.txt_a
    st.session_state.site_b_val = st.session_state.txt_b
    
    st.success(f"Loaded {len(a_list)} items with hard negatives!")
    st.rerun()

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Site A (Your Catalog)**")
    # Use key to bind to session state
    site_a_text = st.text_area("One product per line", key="txt_a", height=300)

with col2:
    st.markdown("**Site B (Competitor/Marketplace)**")
    site_b_text = st.text_area("One product per line", key="txt_b", height=300)

# Process
if st.button("🚀 Run Comparison", type="primary"):
    site_a = [x.strip() for x in site_a_text.split('\n') if x.strip()]
    site_b = [x.strip() for x in site_b_text.split('\n') if x.strip()]
    
    if not site_a or not site_b:
        st.warning("Please provide data for both sites.")
        st.stop()
        
    st.subheader("2. Matching Results")
    
    results = []
    
    progress_bar = st.progress(0)
    
    for i, product_a in enumerate(site_a):
        # Search for best match
        matches = model.search(product_a, site_b, top_k=1)
        
        if matches:
            best = matches[0]
            score = best['score']
            match_product = best['text']
            is_match = score >= threshold
            
            results.append({
                "Site A Product": product_a,
                "Best Match (Site B)": match_product,
                "Confidence": score,
                "Status": "✅ Match" if is_match else "❌ Different"
            })
        else:
            results.append({
                "Site A Product": product_a,
                "Best Match (Site B)": "No candidate found",
                "Confidence": 0.0,
                "Status": "❌ No Data"
            })
        
        progress_bar.progress((i + 1) / len(site_a))
            
    df = pd.DataFrame(results)
    
    # Sort by Confidence (Desc)
    df = df.sort_values(by="Confidence", ascending=False)
    
    # Styling
    def color_status(val):
        color = '#d4edda' if val == "✅ Match" else '#f8d7da'
        return f'background-color: {color}'

    st.dataframe(
        df.style.applymap(color_status, subset=['Status'])
        .format({"Confidence": "{:.4f}"}),
        use_container_width=True
    )
    
    # Stats
    match_count = df[df['Status'] == "✅ Match"].shape[0]
    st.metric("Total Matches Found", f"{match_count} / {len(site_a)}")
