import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="LeafScan AI — Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">

<style>
/* ── Root Variables ── */
:root {
    --green-deep:   #1b5e20;
    --green-mid:    #33691e;
    --green-bright: #2e7d32;
    --green-light:  #43a047;
    --green-pale:   #f0fff0;
    --gold:         #c8a84b;
    --text-dark:    #1b3a1e;
    --text-mid:     #2e5933;
    --white:        #ffffff;
    --shadow:       0 8px 32px rgba(27,94,32,0.18);
    --radius:       18px;
}

/* ── Global Reset ── */
* { box-sizing: border-box; }

.stApp {
    background-image:
        linear-gradient(rgba(240,255,240,0.88), rgba(240,255,240,0.88)),
        url("https://images.unsplash.com/photo-1501004318641-b39e6451bec6?q=80&w=1600&auto=format&fit=crop");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    min-height: 100vh;
    font-family: 'DM Sans', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #e8f5e9 0%, #f0fff0 100%) !important;
    border-right: 2px solid rgba(27,94,32,0.15) !important;
}
[data-testid="stSidebar"] * { color: #1b5e20 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #1b5e20 !important;
    font-family: 'Playfair Display', serif !important;
}
[data-testid="stSidebar"] .stInfo {
    background: rgba(27,94,32,0.08) !important;
    border: 1px solid rgba(27,94,32,0.25) !important;
    border-radius: 12px !important;
    color: #1b5e20 !important;
}
[data-testid="stSidebar"] .stSuccess {
    background: rgba(46,125,50,0.12) !important;
    border: 1px solid rgba(46,125,50,0.3) !important;
    border-radius: 12px !important;
}

/* ── Hero Header ── */
.hero-wrap {
    text-align: center;
    padding: 48px 24px 24px;
    position: relative;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(90deg, #1b5e20, #2e7d32);
    color: #fff;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    padding: 6px 20px;
    border-radius: 100px;
    margin-bottom: 18px;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(36px, 6vw, 68px);
    font-weight: 900;
    color: #1b5e20;
    line-height: 1.08;
    margin: 0 0 16px;
    text-shadow: 0 2px 12px rgba(27,94,32,0.12);
}
.hero-title span { color: #43a047; }
.hero-sub {
    font-size: 17px;
    color: #33691e;
    font-weight: 400;
    letter-spacing: 0.4px;
    max-width: 520px;
    margin: 0 auto 40px;
}

/* ── Upload Zone ── */
.upload-card {
    background: rgba(255,255,255,0.65);
    border: 2px dashed rgba(27,94,32,0.4);
    border-radius: var(--radius);
    padding: 36px 24px;
    text-align: center;
    margin: 0 auto 32px;
    max-width: 560px;
    backdrop-filter: blur(8px);
    transition: border-color 0.3s, background 0.3s;
}
.upload-card:hover {
    border-color: #1b5e20;
    background: rgba(255,255,255,0.80);
}
.upload-icon { font-size: 48px; margin-bottom: 10px; }
.upload-label {
    font-size: 15px;
    color: #33691e;
    margin: 0;
}

/* ── Result Card ── */
.result-card {
    background: rgba(255,255,255,0.72);
    border: 1px solid rgba(27,94,32,0.2);
    border-radius: var(--radius);
    padding: 32px 28px;
    backdrop-filter: blur(16px);
    box-shadow: var(--shadow);
}
.result-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #2e7d32;
    margin-bottom: 10px;
}
.result-class-healthy {
    font-family: 'Playfair Display', serif;
    font-size: 28px;
    font-weight: 700;
    color: #1b5e20;
    margin: 4px 0 18px;
    line-height: 1.2;
}
.result-class-disease {
    font-family: 'Playfair Display', serif;
    font-size: 26px;
    font-weight: 700;
    color: #b71c1c;
    margin: 4px 0 18px;
    line-height: 1.2;
}
.confidence-pill {
    display: inline-block;
    background: linear-gradient(90deg, #1b5e20, #2e7d32);
    color: #fff;
    font-size: 14px;
    font-weight: 600;
    padding: 6px 18px;
    border-radius: 100px;
    margin-bottom: 22px;
}

/* ── Top-3 Bar ── */
.pred-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
}
.pred-name {
    width: 200px;
    font-size: 12px;
    color: #33691e;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.pred-bar-bg {
    flex: 1;
    height: 8px;
    background: rgba(27,94,32,0.12);
    border-radius: 100px;
    overflow: hidden;
}
.pred-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #2e7d32, #43a047);
    transition: width 0.8s ease;
}
.pred-pct {
    width: 44px;
    font-size: 12px;
    font-weight: 600;
    color: #1b5e20;
    text-align: right;
}

/* ── Disease Info Card ── */
.info-card {
    background: rgba(255,255,255,0.72);
    border: 1px solid rgba(27,94,32,0.18);
    border-radius: var(--radius);
    padding: 28px 30px;
    margin-top: 28px;
    backdrop-filter: blur(12px);
    box-shadow: var(--shadow);
}
.info-card-title {
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    font-weight: 700;
    color: #1b5e20;
    margin-bottom: 6px;
}
.info-card-desc {
    font-size: 14px;
    color: #33691e;
    margin-bottom: 22px;
    line-height: 1.6;
}
.tag-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; }
.tag {
    background: rgba(27,94,32,0.10);
    border: 1px solid rgba(27,94,32,0.25);
    color: #1b5e20;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    padding: 4px 14px;
    border-radius: 100px;
    text-transform: uppercase;
}
.tips-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.tip-item {
    background: rgba(240,255,240,0.85);
    border-radius: 12px;
    padding: 14px 16px;
    display: flex;
    gap: 10px;
    align-items: flex-start;
    border: 1px solid rgba(27,94,32,0.12);
}
.tip-icon { font-size: 20px; flex-shrink: 0; margin-top: 1px; }
.tip-text { font-size: 13px; color: #33691e; line-height: 1.5; }
.tip-text strong { color: #1b5e20; display: block; margin-bottom: 2px; }
.severity-healthy { color: #1b5e20; }
.severity-low     { color: #f57f17; }
.severity-medium  { color: #e65100; }
.severity-high    { color: #b71c1c; }

/* ── Image container ── */
.img-frame {
    border-radius: var(--radius);
    overflow: hidden;
    border: 2px solid rgba(27,94,32,0.2);
    box-shadow: 0 12px 40px rgba(27,94,32,0.15);
}

/* ── Divider ── */
.fancy-divider {
    text-align: center;
    margin: 40px 0 24px;
    position: relative;
}
.fancy-divider::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(27,94,32,0.3), transparent);
}
.fancy-divider span {
    background: #f0fff0;
    padding: 0 16px;
    position: relative;
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #2e7d32;
    font-weight: 600;
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 30px 0 16px;
    font-size: 13px;
    color: rgba(27,94,32,0.55);
    letter-spacing: 0.5px;
}

/* ── Streamlit overrides ── */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.55) !important;
    border: 2px dashed rgba(27,94,32,0.35) !important;
    border-radius: 14px !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #1b5e20 !important;
    background: rgba(255,255,255,0.75) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] { color: #33691e !important; }
[data-testid="stSpinner"] { color: #2e7d32 !important; }
.stProgress > div > div {
    background: linear-gradient(90deg, #2e7d32, #43a047) !important;
    border-radius: 100px !important;
}
.stProgress { background: rgba(27,94,32,0.12) !important; border-radius: 100px !important; }
h1, h2, h3 { font-family: 'Playfair Display', serif !important; color: #1b5e20 !important; }
p, li, span { color: #33691e; }
.stMarkdown p { color: #33691e; }
</style>
""", unsafe_allow_html=True)

# ---------------- DISEASE DATABASE ----------------
DISEASE_INFO = {
    "Pepper__bell___Bacterial_spot": {
        "display": "Bell Pepper — Bacterial Spot",
        "emoji": "🌶️",
        "severity": "medium",
        "severity_label": "Moderate",
        "description": "Caused by Xanthomonas campestris, this bacterial infection creates water-soaked lesions on leaves that turn brown and scabby on fruits. Spreads rapidly in warm, wet conditions.",
        "tags": ["Bacterial", "Warm & Humid", "Contagious"],
        "tips": [
            ("🗑️", "Remove & Destroy", "Immediately remove and dispose of all infected plant material away from the garden."),
            ("💧", "Drip Irrigation", "Switch to drip irrigation to keep foliage dry and prevent splash-spreading bacteria."),
            ("🧪", "Copper Fungicide", "Apply copper-based bactericides (e.g., Kocide 3000) every 7–10 days during wet periods."),
            ("🌬️", "Improve Airflow", "Space plants adequately and prune dense canopy to reduce leaf wetness duration."),
        ],
    },
    "Pepper__bell___healthy": {
        "display": "Bell Pepper — Healthy",
        "emoji": "🌿",
        "severity": "healthy",
        "severity_label": "Healthy",
        "description": "Your bell pepper plant looks vigorous and disease-free. Maintain current care practices and stay vigilant for early signs of stress.",
        "tags": ["No Disease", "Thriving"],
        "tips": [
            ("☀️", "Consistent Sunlight", "Ensure 6–8 hours of direct sunlight daily for optimal fruit production."),
            ("💧", "Deep Watering", "Water deeply but infrequently (every 3–5 days) to encourage strong root systems."),
            ("🌱", "Fertilise Regularly", "Apply a balanced NPK fertiliser every 2 weeks during the growing season."),
            ("👀", "Scout Weekly", "Check undersides of leaves weekly for early insect or disease signs."),
        ],
    },
    "Potato___Early_blight": {
        "display": "Potato — Early Blight",
        "emoji": "🥔",
        "severity": "medium",
        "severity_label": "Moderate",
        "description": "Caused by Alternaria solani, this fungal disease appears as dark concentric 'target board' rings on older leaves, reducing photosynthesis and yield significantly.",
        "tags": ["Fungal", "Alternaria", "Target Lesions"],
        "tips": [
            ("🍂", "Remove Lower Leaves", "Prune and destroy lower infected leaves as soon as lesions appear."),
            ("🧴", "Mancozeb Spray", "Apply Mancozeb or chlorothalonil fungicide every 7 days in humid conditions."),
            ("🔄", "Crop Rotation", "Rotate potatoes with non-solanaceous crops for at least 2–3 years."),
            ("🧹", "Debris Cleanup", "Remove plant debris after harvest to eliminate overwintering fungal spores."),
        ],
    },
    "Potato___Late_blight": {
        "display": "Potato — Late Blight",
        "emoji": "🥔",
        "severity": "high",
        "severity_label": "Severe",
        "description": "The most devastating potato disease, caused by Phytophthora infestans (responsible for the Irish Famine). Spreads explosively in cool, moist weather and can wipe out an entire crop within days.",
        "tags": ["Oomycete", "Rapid Spread", "Critical"],
        "tips": [
            ("🚨", "Act Immediately", "Late blight is an emergency — act within 24 hours of detection to contain spread."),
            ("🗑️", "Destroy All Infected", "Remove and bag ALL infected material; do not compost. Burn if legally permitted."),
            ("💊", "Metalaxyl Fungicide", "Apply systemic fungicides like Ridomil Gold or Revus immediately and repeatedly."),
            ("📡", "Monitor Forecasts", "Use blight forecasting tools (BlightPro) to schedule preventive sprays before rain events."),
        ],
    },
    "Potato___healthy": {
        "display": "Potato — Healthy",
        "emoji": "🌿",
        "severity": "healthy",
        "severity_label": "Healthy",
        "description": "Your potato plants are in great condition. Focus on preventive practices to protect your crop through maturity.",
        "tags": ["No Disease", "Thriving"],
        "tips": [
            ("🌾", "Hill Your Plants", "Mound soil around stems as they grow to protect tubers from greening and blight."),
            ("💧", "Even Moisture", "Maintain consistent soil moisture to prevent hollow heart and cracking in tubers."),
            ("🛡️", "Preventive Spray", "Apply a preventive copper-based spray before extended wet weather periods."),
            ("🔍", "Monitor Closely", "Inspect daily during cool, wet weather when late blight risk is highest."),
        ],
    },
    "Tomato_Bacterial_spot": {
        "display": "Tomato — Bacterial Spot",
        "emoji": "🍅",
        "severity": "medium",
        "severity_label": "Moderate",
        "description": "Caused by Xanthomonas species, bacterial spot creates small, water-soaked lesions on leaves, stems, and fruits. Fruits develop raised, corky scabs that reduce market value.",
        "tags": ["Bacterial", "Xanthomonas", "Fruit Damage"],
        "tips": [
            ("💧", "Avoid Overhead Water", "Use drip irrigation; overhead watering splashes bacteria from soil to leaves."),
            ("🧪", "Copper + Mancozeb", "Tank-mix copper bactericide with Mancozeb for improved bacterial control."),
            ("🌱", "Resistant Varieties", "Plant varieties with bacterial spot resistance for future seasons (e.g., BHN 589)."),
            ("✂️", "Sanitation First", "Disinfect pruning tools with 10% bleach solution between plants."),
        ],
    },
    "Tomato_Early_blight": {
        "display": "Tomato — Early Blight",
        "emoji": "🍅",
        "severity": "medium",
        "severity_label": "Moderate",
        "description": "Alternaria solani causes dark brown lesions with concentric rings and yellow halos, typically starting on lower leaves and moving upward, stressing the plant and reducing yield.",
        "tags": ["Fungal", "Alternaria", "Lower Leaves First"],
        "tips": [
            ("🍂", "Remove Infected Leaves", "Prune affected lower leaves immediately; dispose of them away from the garden."),
            ("🧴", "Chlorothalonil Spray", "Apply chlorothalonil or copper fungicide every 7–10 days preventively."),
            ("🌿", "Mulch Heavily", "Apply 3–4 inches of mulch to prevent spore splash-up from soil."),
            ("🔄", "3-Year Rotation", "Avoid planting tomatoes or other solanums in the same bed for 3 years."),
        ],
    },
    "Tomato_Late_blight": {
        "display": "Tomato — Late Blight",
        "emoji": "🍅",
        "severity": "high",
        "severity_label": "Severe",
        "description": "Phytophthora infestans causes greasy, irregular dark lesions that rapidly expand. White sporulation appears on the underside of leaves in humid conditions. Entire plants can collapse in days.",
        "tags": ["Oomycete", "Rapid Collapse", "Critical"],
        "tips": [
            ("🚨", "Emergency Response", "Remove all visibly infected plants and parts immediately to limit spread to neighbours."),
            ("💊", "Systemic Fungicide", "Apply Ridomil Gold, Revus, or Zampro — contact-only fungicides are not sufficient."),
            ("☀️", "Improve Air Circulation", "Stake, cage, and aggressively prune plants to reduce the humid microclimate blight loves."),
            ("📡", "Forecast-Based Spray", "Apply preventive sprays before predicted rain or cool-humid weather windows."),
        ],
    },
    "Tomato_Leaf_Mold": {
        "display": "Tomato — Leaf Mold",
        "emoji": "🍅",
        "severity": "low",
        "severity_label": "Low–Moderate",
        "description": "Caused by Passalora fulva (formerly Cladosporium fulvum), leaf mold thrives in greenhouses and tunnels. Pale yellow patches on leaf surfaces match olive-brown fuzzy spore masses underneath.",
        "tags": ["Fungal", "Greenhouse", "Humidity-Driven"],
        "tips": [
            ("🌬️", "Ventilate Aggressively", "Keep relative humidity below 85% — open vents, use fans, and space plants wider."),
            ("🧴", "Copper Fungicide", "Apply copper oxychloride or mancozeb sprays every 7 days on upper and lower leaf surfaces."),
            ("🌡️", "Temperature Control", "Maintain night temperatures above 15°C (59°F) to slow fungal development."),
            ("🌱", "Use Resistant Cultivars", "Modern hybrid tomatoes with Cf-gene resistance are far less susceptible."),
        ],
    },
    "Tomato_Septoria_leaf_spot": {
        "display": "Tomato — Septoria Leaf Spot",
        "emoji": "🍅",
        "severity": "medium",
        "severity_label": "Moderate",
        "description": "Septoria lycopersici causes numerous small, circular spots with dark borders and lighter centres, often with visible black pycnidia (spore-producing bodies) inside the spots.",
        "tags": ["Fungal", "Septoria", "Small Circular Spots"],
        "tips": [
            ("🍂", "Remove Lower Leaves", "Septoria starts at the bottom — immediately prune and dispose of all spotted leaves."),
            ("💧", "Drip Irrigation Only", "Keep water off foliage; Septoria spores are spread by water splash."),
            ("🧴", "Chlorothalonil", "Apply Daconil (chlorothalonil) on a 7-day schedule during warm, wet conditions."),
            ("🌿", "Thick Mulch", "Mulch with straw or wood chips to prevent soil-splash onto lower leaves."),
        ],
    },
    "Tomato_Spider_mites": {
        "display": "Tomato — Spider Mites",
        "emoji": "🍅",
        "severity": "medium",
        "severity_label": "Moderate",
        "description": "Tetranychus urticae are not insects but tiny arachnids that puncture leaf cells to feed, causing stippled, bronzed foliage. Webbing beneath leaves is a diagnostic sign; hot, dry conditions trigger outbreaks.",
        "tags": ["Arachnid", "Hot & Dry", "Stippling & Webbing"],
        "tips": [
            ("💦", "Strong Water Spray", "Blast the undersides of leaves with a firm jet of water to dislodge and drown mites."),
            ("🧴", "Neem Oil / Insecticidal Soap", "Apply neem oil or insecticidal soap every 3–5 days for at least 3 cycles."),
            ("🐞", "Beneficial Predators", "Release predatory mites (Phytoseiulus persimilis) for sustainable biological control."),
            ("💧", "Increase Humidity", "Mites hate moisture — mulch, mist pathways, and avoid water stress in plants."),
        ],
    },
    "Tomato_Target_Spot": {
        "display": "Tomato — Target Spot",
        "emoji": "🍅",
        "severity": "medium",
        "severity_label": "Moderate",
        "description": "Corynespora cassiicola causes brown lesions with concentric target-like rings on leaves, stems, and fruits, often confused with early blight. It thrives in warm, humid tropical climates.",
        "tags": ["Fungal", "Tropical", "Concentric Rings"],
        "tips": [
            ("🧴", "Azoxystrobin Spray", "Apply strobilurin fungicides (azoxystrobin, trifloxystrobin) on a 10-day rotation."),
            ("✂️", "Aggressive Pruning", "Remove all infected leaves and maintain open plant canopy to reduce humidity."),
            ("🔄", "Rotate Fungicide Classes", "Rotate between fungicide classes to prevent resistance development."),
            ("🌱", "Post-Season Cleanup", "Completely remove and destroy all plant debris after the growing season ends."),
        ],
    },
    "Tomato_Mosaic_virus": {
        "display": "Tomato — Mosaic Virus",
        "emoji": "🍅",
        "severity": "high",
        "severity_label": "Severe",
        "description": "Tomato Mosaic Virus (ToMV) causes characteristic mosaic patterning, leaf distortion, and stunting. Extremely persistent in soil and on tools; spreads through mechanical contact and seed.",
        "tags": ["Viral", "No Cure", "Tool-Transmitted"],
        "tips": [
            ("🗑️", "Remove Infected Plants", "There is no cure — remove and destroy all infected plants to protect healthy ones."),
            ("🧼", "Sanitise Everything", "Disinfect tools, hands, and clothing with 10% bleach or trisodium phosphate after handling."),
            ("🌱", "Plant Certified Seeds", "Use certified virus-free or virus-resistant seed/transplants in future crops."),
            ("🚭", "Tobacco Users Beware", "Tobacco products carry ToMV — wash hands thoroughly before touching plants."),
        ],
    },
    "Tomato_Yellow_Leaf_Curl_Virus": {
        "display": "Tomato — Yellow Leaf Curl Virus",
        "emoji": "🍅",
        "severity": "high",
        "severity_label": "Severe",
        "description": "TYLCV is transmitted exclusively by the silverleaf whitefly (Bemisia tabaci). Infected plants show severe stunting, upward leaf curling, and yellowing. Crop losses can reach 100%.",
        "tags": ["Viral", "Whitefly-Transmitted", "Critical"],
        "tips": [
            ("🐝", "Control Whiteflies", "TYLCV is whitefly-transmitted — yellow sticky traps, neonicotinoid drenches, and reflective mulches help."),
            ("🗑️", "Rogue Infected Plants", "Remove infected plants early before whitefly populations build up on them."),
            ("🌱", "Resistant Varieties", "Plant TYLCV-resistant hybrids in future seasons (look for the 'Ty' gene marker)."),
            ("🕸️", "Insect-Proof Netting", "Grow under 50-mesh insect netting in whitefly-pressure areas."),
        ],
    },
    "Tomato_healthy": {
        "display": "Tomato — Healthy",
        "emoji": "🌿",
        "severity": "healthy",
        "severity_label": "Healthy",
        "description": "Your tomato plant looks excellent! Continue your current care regimen and apply the preventive measures below to ensure a bountiful harvest.",
        "tags": ["No Disease", "Thriving"],
        "tips": [
            ("☀️", "Full Sun Daily", "Tomatoes need 8+ hours of sun; position or stake plants for maximum light exposure."),
            ("💧", "Consistent Watering", "Water deeply and evenly — inconsistent moisture causes blossom end rot and cracking."),
            ("🌿", "Preventive Spray", "Apply a copper-based preventive spray every 2 weeks during humid periods."),
            ("✂️", "Sucker Pruning", "Remove suckers and lower leaves to improve airflow and focus energy on fruits."),
        ],
    },
}


def severity_class(sev):
    return f"severity-{sev}"


def render_disease_card(info):
    sev_cls  = severity_class(info["severity"])
    tags_html = "".join(f'<span class="tag">{t}</span>' for t in info["tags"])
    tips_html = ""
    for icon, title, text in info["tips"]:
        tips_html += f"""
        <div class="tip-item">
            <span class="tip-icon">{icon}</span>
            <div class="tip-text"><strong>{title}</strong>{text}</div>
        </div>"""

    return f"""
    <div class="info-card">
        <div class="result-label">Disease Analysis & Care Guide</div>
        <div class="info-card-title">{info["emoji"]} {info["display"]}</div>
        <p class="info-card-desc">{info["description"]}</p>
        <div class="tag-row">
            {tags_html}
            <span class="tag {sev_cls}" style="background:rgba(0,0,0,0.2);">Severity: {info["severity_label"]}</span>
        </div>
        <div class="result-label" style="margin-bottom:12px;">Recommended Actions</div>
        <div class="tips-grid">{tips_html}</div>
    </div>
    """


def top3_bars_html(top3_indices, predictions, class_names):
    rows = ""
    for idx in top3_indices:
        pct  = predictions[idx] * 100
        name = class_names[idx].replace("_", " ").replace("  ", " — ")
        rows += f"""
        <div class="pred-row">
            <div class="pred-name" title="{name}">{name}</div>
            <div class="pred-bar-bg">
                <div class="pred-bar-fill" style="width:{pct:.1f}%"></div>
            </div>
            <div class="pred-pct">{pct:.1f}%</div>
        </div>"""
    return rows


# ---------------- LOAD MODEL ----------------
MODEL_PATH = "plant_disease_model.keras"
IMG_SIZE   = (224, 224)

@st.cache_resource
def load_ml_model():
    return load_model(MODEL_PATH)

model = load_ml_model()

class_names = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites",
    "Tomato_Target_Spot",
    "Tomato_Mosaic_virus",
    "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_healthy",
]

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown(
        '<h2 style="color:#1b5e20;font-family:Playfair Display,serif;">🌱 LeafScan AI</h2>',
        unsafe_allow_html=True
    )
    st.info("""
**AI-powered plant health diagnostics.**

Upload a leaf photo to get an instant diagnosis with:
- ✅ Disease identification
- 📊 Confidence scoring
- 🔬 Top 3 predictions
- 💊 Tailored treatment guide

**Supported crops:** Tomato · Potato · Bell Pepper
    """)
    st.success("✅ Model loaded & ready")
    st.markdown("---")
    st.markdown("**15 classes** · CNN · TensorFlow")
    st.markdown("*For best results, upload a clear, well-lit close-up of a single leaf.*")

# ---------------- HERO ----------------
st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge">AI Plant Health Diagnostics</div>
    <div class="hero-title">Detect Leaf Disease<br><span>In Seconds</span></div>
    <div class="hero-sub">Upload a leaf image and let our CNN model diagnose disease, measure confidence, and prescribe targeted treatment.</div>
</div>
""", unsafe_allow_html=True)

# ---------------- FILE UPLOADER ----------------
uploaded_file = st.file_uploader(
    "Upload a leaf image (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)


# ---------------- CONFIDENCE THRESHOLD ----------------
CONFIDENCE_THRESHOLD = 60.0  # below this → treated as out-of-scope

# ---------------- PREDICTION ----------------
def predict_disease(image):
    image    = image.resize(IMG_SIZE)
    arr      = img_to_array(image) / 255.0
    arr      = np.expand_dims(arr, axis=0)
    preds    = model.predict(arr, verbose=0)
    pred_idx = int(np.argmax(preds[0]))
    pred_cls = class_names[pred_idx]
    conf     = float(preds[0][pred_idx]) * 100
    top3     = np.argsort(preds[0])[::-1][:3]
    return pred_cls, conf, preds[0], top3


# ---------------- MAIN ----------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1.1, 1], gap="large")

    with col1:
        st.markdown('<div class="img-frame">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.spinner("🔬 Analysing leaf tissue…"):
        predicted_class, confidence, predictions, top3_indices = predict_disease(image)

    # ── Out-of-scope detection ──
    out_of_scope = confidence < CONFIDENCE_THRESHOLD

    if out_of_scope:
        with col2:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Prediction Result</div>
                <div style="font-family:'Playfair Display',serif;font-size:26px;font-weight:700;
                            color:#b45309;margin:4px 0 14px;line-height:1.2;">
                    ❓ Image Not Recognised
                </div>
                <div style="background:rgba(251,191,36,0.15);border:1px solid rgba(180,83,9,0.3);
                            border-radius:12px;padding:16px 18px;margin-bottom:18px;">
                    <p style="color:#92400e;font-size:14px;margin:0;line-height:1.6;">
                        <strong>This image does not appear to be a supported plant leaf.</strong><br>
                        The model's best guess was <em>{predicted_class.replace("_"," ")}</em> with only
                        <strong>{confidence:.1f}% confidence</strong> — well below the {CONFIDENCE_THRESHOLD:.0f}%
                        threshold required for a reliable diagnosis.
                    </p>
                </div>
                <div class="result-label" style="margin-bottom:10px;">What to try instead</div>
                <div style="display:flex;flex-direction:column;gap:10px;">
                    <div class="tip-item">
                        <span class="tip-icon">🍃</span>
                        <div class="tip-text"><strong>Use a leaf close-up</strong>
                        Photograph a single leaf filling most of the frame, in good natural light.</div>
                    </div>
                    <div class="tip-item">
                        <span class="tip-icon">🌿</span>
                        <div class="tip-text"><strong>Supported crops only</strong>
                        This model recognises Tomato, Potato, and Bell Pepper leaves only.</div>
                    </div>
                    <div class="tip-item">
                        <span class="tip-icon">☀️</span>
                        <div class="tip-text"><strong>Avoid blur & shadows</strong>
                        Sharp, evenly lit images produce the most accurate predictions.</div>
                    </div>
                </div>
                <div style="margin-top:18px;">
                    <div class="result-label" style="margin-bottom:8px;">Model's raw top-3 guesses</div>
                    {top3_bars_html(top3_indices, predictions, class_names)}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Out-of-scope info card below
        st.markdown("""
        <div class="info-card" style="border-color:rgba(180,83,9,0.25);">
            <div class="result-label" style="color:#b45309;">Supported Plant Classes</div>
            <div class="info-card-title">🌱 What This Model Can Diagnose</div>
            <p class="info-card-desc">
                This CNN was trained on 15 specific classes across three crops.
                Uploading images of other plants, animals, objects, or people will produce
                unreliable results. Always use clear, close-up leaf photographs.
            </p>
            <div class="tag-row">
                <span class="tag">🌶️ Bell Pepper</span>
                <span class="tag">🥔 Potato</span>
                <span class="tag">🍅 Tomato</span>
                <span class="tag">15 Disease Classes</span>
            </div>
            <div class="tips-grid">
                <div class="tip-item"><span class="tip-icon">🌶️</span>
                    <div class="tip-text"><strong>Bell Pepper</strong>Bacterial Spot · Healthy</div></div>
                <div class="tip-item"><span class="tip-icon">🥔</span>
                    <div class="tip-text"><strong>Potato</strong>Early Blight · Late Blight · Healthy</div></div>
                <div class="tip-item"><span class="tip-icon">🍅</span>
                    <div class="tip-text"><strong>Tomato (10 classes)</strong>Bacterial Spot · Early Blight · Late Blight · Leaf Mold · Septoria · Spider Mites · Target Spot · Mosaic Virus · Yellow Leaf Curl · Healthy</div></div>
                <div class="tip-item"><span class="tip-icon">📷</span>
                    <div class="tip-text"><strong>Image Tips</strong>Close-up, single leaf, natural light, sharp focus, plain or field background.</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Normal in-scope result ──
        info = DISEASE_INFO.get(predicted_class, None)
        is_healthy   = "healthy" in predicted_class.lower()
        display_name = info["display"] if info else predicted_class.replace("_", " ")

        with col2:
            cls_div     = "result-class-healthy" if is_healthy else "result-class-disease"
            status_icon = "✅" if is_healthy else "⚠️"

            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Prediction Result</div>
                <div class="{cls_div}">{status_icon} {display_name}</div>
                <div class="confidence-pill">Confidence: {confidence:.2f}%</div>
            """, unsafe_allow_html=True)

            st.progress(int(confidence))

            st.markdown(f"""
                <div class="result-label" style="margin-top:22px; margin-bottom:10px;">Top 3 Predictions</div>
                {top3_bars_html(top3_indices, predictions, class_names)}
            </div>
            """, unsafe_allow_html=True)

        # Disease Info
        if info:
            st.markdown(render_disease_card(info), unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="upload-card">
        <div class="upload-icon">🍃</div>
        <p class="upload-label">Drag & drop a leaf image above, or click <strong>Browse files</strong><br>
        <small>Supports JPG · JPEG · PNG &nbsp;|&nbsp; Bell Pepper · Potato · Tomato</small></p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    Built with CNN · TensorFlow · Streamlit &nbsp;·&nbsp; LeafScan AI &nbsp;·&nbsp; 15 Plant Disease Classes
</div>
""", unsafe_allow_html=True)