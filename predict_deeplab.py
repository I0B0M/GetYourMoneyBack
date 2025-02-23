import torch
import torchvision.transforms as T
from torchvision import models
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import glob
import re
import logging
from skimage.exposure import equalize_adapthist
from skimage.metrics import structural_similarity as ssim
from skimage.filters import sobel
from skimage.transform import resize
from skimage.measure import label, regionprops
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as ReportLabImage
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logging.debug("✅ Script started")

# ------------------ LOADING SEGMENTATION MODEL ------------------
logging.debug("🔄 Loading segmentation model...")
try:
    segmentation_model = models.segmentation.deeplabv3_resnet101(weights="DeepLabV3_ResNet101_Weights.DEFAULT")
    segmentation_model.eval()
    logging.debug("✅ Segmentation model loaded successfully")
except Exception as e:
    logging.error(f"❌ Failed to load segmentation model: {e}")
    exit(1)

# ------------------ FOLDER SETUP ------------------
INPUT_FOLDER = "/Users/ibm/Desktop/hurricane_project/cropped_images/"
OUTPUT_FOLDER = "/Users/ibm/Desktop/hurricane_project/results/"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
logging.debug(f"📂 Input Folder: {INPUT_FOLDER}")
logging.debug(f"📂 Output Folder: {OUTPUT_FOLDER}")

# ------------------ CONSTANTS & TRANSFORMS ------------------
FIXED_SIZE = (256, 256)
transform = T.Compose([
    T.Resize(FIXED_SIZE),
    T.ToTensor(),
])
MULTIPLIERS = {"Minor": 1.0, "Moderate": 1.3, "Major": 1.6}
BASE_COSTS = {"Roof": 10000, "Pool": 3000, "Garden": 2000, "Debris Cleanup": 1000}
MINOR_BASELINE = 1000

# ------------------ IMAGE CLARIFICATION ------------------
def clarify_image(image_path):
    from skimage.exposure import equalize_adapthist
    from PIL import Image
    import numpy as np
    image = Image.open(image_path).convert("L")
    image_np = np.array(image, dtype=float) / 255.0
    clarified = equalize_adapthist(image_np, clip_limit=0.03)
    return clarified

# ------------------ SIMPLE COLOR-BASED POOL DETECTION (AQUA) ------------------
def detect_pool_existence_aqua(image_path, fraction_threshold=0.05):
    """
    Heuristic: we define an 'aqua' pixel if:
      - 0.1 <= R <= 0.6
      - 0.4 <= G <= 1.0
      - 0.5 <= B <= 1.0
      - B >= G >= R
    If more than fraction_threshold (e.g. 5%) of bottom 25% of the image are 'aqua', we say a pool is present.
    """
    from PIL import Image
    import numpy as np
    image = Image.open(image_path).convert("RGB").resize(FIXED_SIZE, Image.LANCZOS)
    width, height = image.size
    # Crop bottom 25%
    pool_region = image.crop((0, int(height*0.75), width, height))
    pool_np = np.array(pool_region, dtype=np.float32) / 255.0
    
    R = pool_np[:,:,0]
    G = pool_np[:,:,1]
    B = pool_np[:,:,2]
    cond_r = (R >= 0.1) & (R <= 0.6)
    cond_g = (G >= 0.4) & (G <= 1.0)
    cond_b = (B >= 0.5) & (B <= 1.0)
    cond_order = (B >= G) & (G >= R)
    aqua_mask = cond_r & cond_g & cond_b & cond_order
    fraction_aqua = np.mean(aqua_mask)
    logging.debug(f"Aqua fraction in bottom 25%: {fraction_aqua:.3f}")
    return (fraction_aqua > fraction_threshold)

# ------------------ HELPER FUNCTIONS ------------------
def resize_image(image_path):
    logging.debug(f"🔄 Resizing image: {image_path}")
    from PIL import Image
    image = Image.open(image_path).convert("RGB")
    return image.resize(FIXED_SIZE, Image.LANCZOS)

def segment_image_mask(image_path):
    try:
        image = resize_image(image_path)
        image_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = segmentation_model(image_tensor)['out'][0]
            prediction = output.argmax(0).byte().cpu().numpy()
        return prediction
    except Exception as e:
        logging.error(f"❌ Error in segmentation: {e}")
        return None

def compute_ssim_diff(before_path, after_path):
    from skimage.metrics import structural_similarity as ssim
    import numpy as np
    before_clar = clarify_image(before_path)
    after_clar = clarify_image(after_path)
    after_clar_resized = resize(after_clar, before_clar.shape, anti_aliasing=True)
    sim, _ = ssim(before_clar, after_clar_resized, full=True, data_range=1.0)
    diff = 1.0 - sim
    logging.debug(f"SSIM difference (1-sim): {diff:.3f}")
    return diff

def detect_debris_sobel(image_path, edge_threshold=0.2, min_area=200):
    import numpy as np
    from skimage.filters import sobel
    from skimage.measure import label, regionprops
    clarified = clarify_image(image_path)
    edges = sobel(clarified)
    binary_edges = edges > edge_threshold
    labeled = label(binary_edges)
    count = sum(1 for region in regionprops(labeled) if region.area >= min_area)
    logging.debug(f"Detected {count} debris objects via skimage edges.")
    return count

def classify_damage(percentage):
    if percentage < 0.2:
        return "Minor"
    elif percentage < 0.5:
        return "Moderate"
    else:
        return "Major"

def classify_debris(debris_count):
    return "Major" if debris_count >= 1 else "Minor"

def detect_pool_condition(pool_mask, after_image_path):
    """
    1) Check if a pool is present using the 'aqua' approach in detect_pool_existence_aqua.
    2) If not, return "No Pool".
    3) Otherwise, examine pool_mask:
       - average > 0.5 => "Has Debris"
       - average < 0.1 => "Needs Refill"
       - else => "Good Condition"
    """
    pool_exists = detect_pool_existence_aqua(after_image_path, fraction_threshold=0.05)
    if not pool_exists:
        return "No Pool"
    pool_pixels = np.sum(pool_mask) / pool_mask.size
    if pool_pixels > 0.5:
        return "Has Debris"
    elif pool_pixels < 0.1:
        return "Needs Refill"
    else:
        return "Good Condition"

def compare_damage(before_path, after_path, before_mask, after_mask):
    logging.debug("🔍 Comparing images for damage detection...")
    try:
        ssim_diff = compute_ssim_diff(before_path, after_path)
        debris_count = detect_debris_sobel(after_path, edge_threshold=0.2, min_area=200)
        
        diff_mask = abs(before_mask - after_mask)
        height, width = diff_mask.shape
        roof_area = diff_mask[: height // 3, :]
        pool_area = diff_mask[height // 2 : height // 2 + height // 6, :]
        garden_area = diff_mask[height // 2 :, :]
        
        roof_damage = (roof_area.sum() / roof_area.size) * 100
        pool_damage = (pool_area.sum() / pool_area.size) * 100
        garden_damage = (garden_area.sum() / garden_area.size) * 100
        
        # Scale down roof & garden damage
        effective_roof_damage = roof_damage * 0.5
        effective_garden_damage = garden_damage * 0.5
        
        roof_sev = classify_damage(effective_roof_damage)
        garden_sev = classify_damage(effective_garden_damage)
        
        pool_cond = detect_pool_condition(pool_area, after_path)
        if pool_cond == "No Pool":
            # Instead of "N/A", say "No Pool Present in House"
            pool_sev = "No Pool Present in House"
        else:
            if pool_cond == "Has Debris":
                pool_sev = "Major"
            elif pool_cond == "Needs Refill":
                # Instead of just "Minor (Needs Refill)", say "Minor (Needs Refill + Debris Cleanup)"
                pool_sev = "Minor (Needs Refill + Debris Cleanup)"
            else:
                # "Good Condition" => just "Minor"
                pool_sev = "Minor"
        
        debris_sev = classify_debris(debris_count)
        
        damage_levels = {
            "Roof": roof_sev,
            "Pool": pool_sev,
            "Garden": garden_sev,
            "Debris Cleanup": debris_sev,
        }
        
        # Overall severity
        def severity_value(s):
            s_lower = s.lower()
            if s_lower.startswith("major"):
                return 3
            elif s_lower.startswith("minor"):
                return 1
            else:
                return 2  # fallback: moderate
        
        overall_severity = max(
            severity_value(roof_sev),
            severity_value(pool_sev),
            severity_value(garden_sev),
            severity_value(debris_sev)
        )
        if overall_severity == 3:
            overall_damage = "Major"
        elif overall_severity == 1:
            overall_damage = "Minor"
        else:
            overall_damage = "Moderate"
        
        damage_levels["Overall Damage"] = overall_damage
        
        # Cost calculation
        base_cost_non_debris = MINOR_BASELINE
        
        if "No Pool" in pool_sev:
            pool_cost = 0
        else:
            if "Needs Repair" in pool_cond:
                pool_cost = 5000
            else:
                pool_cost = BASE_COSTS["Pool"]
        
        cost_estimates = {
            "Roof": BASE_COSTS["Roof"],
            "Pool": pool_cost,
            "Garden": BASE_COSTS["Garden"],
            "Debris Cleanup": BASE_COSTS["Debris Cleanup"] * debris_count,
        }
        
        cost_breakdown = {}
        
        def severity_for_cost(cat_sev):
            low = cat_sev.lower()
            if low.startswith("major"):
                return "Major"
            elif low.startswith("minor"):
                return "Minor"
            else:
                return "Moderate"
        
        for category in cost_estimates:
            sev = severity_for_cost(damage_levels[category])
            if sev == "Minor":
                cost_breakdown[category] = base_cost_non_debris
            else:
                cost_breakdown[category] = cost_estimates[category] * MULTIPLIERS[sev]
        
        total_estimated_cost = sum(cost_breakdown.values())
        
        logging.debug(f"✅ Damage Levels: {damage_levels}")
        logging.debug(f"✅ Total Estimated Repair Cost: ${total_estimated_cost:,.2f}")
        return damage_levels, total_estimated_cost, debris_count, cost_breakdown
    except Exception as e:
        logging.error(f"❌ Error in damage comparison: {e}")
        return None, None, None, None

def generate_pdf_report(before, after, damage_data, total_cost, cost_breakdown, output_pdf):
    try:
        doc = SimpleDocTemplate(output_pdf, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()
        title_style = styles['Title']
        header_style = styles['Heading2']
        bold_style = styles['Heading3']
        
        elements.append(Paragraph("Property Damage Assessment Report", title_style))
        elements.append(Spacer(1, 20))
        
        before_img_path = os.path.join(INPUT_FOLDER, before)
        after_img_path = os.path.join(INPUT_FOLDER, after)
        img_table = Table([
            [
                ReportLabImage(before_img_path, width=3.5 * inch, height=3.5 * inch),
                ReportLabImage(after_img_path, width=3.5 * inch, height=3.5 * inch)
            ],
            ["Before Damage", "After Damage"]
        ], colWidths=[3.5 * inch, 3.5 * inch])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('TEXTCOLOR', (0, 1), (-1, 1), colors.black),
            ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 1), (-1, 1), 10),
        ]))
        elements.append(img_table)
        elements.append(Spacer(1, 20))
        
        minor_bg = colors.Color(1, 1, 0.8)
        moderate_bg = colors.Color(1, 0.9, 0.8)
        major_bg = colors.Color(1, 0.8, 0.8)
        
        def get_bg_color(severity_text):
            sev = severity_text.split()[0].lower()
            if sev.startswith("minor"):
                return minor_bg
            elif sev.startswith("moderate"):
                return moderate_bg
            elif sev.startswith("major"):
                return major_bg
            else:
                return colors.white
        
        # Increase the second column width to accommodate longer text
        table_data = [
            [Paragraph("<b>Category</b>", bold_style),
             Paragraph("<b>Severity</b>", bold_style),
             Paragraph("<b>Estimated Cost</b>", bold_style)],
            ["Roof", damage_data["Roof"], f"${cost_breakdown.get('Roof', 0):,.2f}"],
            ["Pool", damage_data["Pool"], f"${cost_breakdown.get('Pool', 0):,.2f}"],
            ["Garden", damage_data["Garden"], f"${cost_breakdown.get('Garden', 0):,.2f}"],
            ["Debris Cleanup", damage_data["Debris Cleanup"], f"${cost_breakdown.get('Debris Cleanup', 0):,.2f}"],
            [Paragraph("<b>Overall Damage</b>", bold_style),
             Paragraph(f"<b>{damage_data['Overall Damage']}</b>", bold_style),
             Paragraph(f"<b>${total_cost:,.2f}</b>", bold_style)]
        ]
        
        # Here we changed the second column (Severity) to 220 width to accommodate longer text
        table = Table(table_data, colWidths=[200, 220, 150])
        
        base_style = [
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ]
        header_style_cmd = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.green),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ]
        table_style = TableStyle(base_style + header_style_cmd)
        
        # Color each severity cell
        for row in range(1, 5):
            severity_value = table_data[row][1]
            bg_color = get_bg_color(severity_value)
            table_style.add('BACKGROUND', (1, row), (1, row), bg_color)
        
        # Color overall severity cell
        overall_sev_text = damage_data["Overall Damage"]
        overall_bg = get_bg_color(overall_sev_text)
        table_style.add('BACKGROUND', (1, 5), (1, 5), overall_bg)
        
        table.setStyle(table_style)
        
        elements.append(Paragraph("Damage Breakdown", header_style))
        elements.append(table)
        elements.append(Spacer(1, 20))
        elements.append(Paragraph(f"<b>Total Estimated Repair Cost: ${total_cost:,.2f}</b>", bold_style))
        elements.append(Spacer(1, 20))
        
        doc.build(elements)
        logging.debug(f"📄 PDF Report successfully saved: {output_pdf}")
        
    except Exception as e:
        logging.error(f"❌ PDF generation failed: {e}")

# ------------------ MAIN EXECUTION ------------------
if __name__ == "__main__":
    logging.debug("🚀 Running damage assessment for multiple image pairs...")
    import os, glob, re
    
    image_files = glob.glob(os.path.join(INPUT_FOLDER, "*.jpg"))
    pattern = re.compile(r"Houses(\d+)_(Before|After)\.jpg", re.IGNORECASE)
    groups = {}
    
    for filepath in image_files:
        filename = os.path.basename(filepath)
        match = pattern.match(filename)
        if match:
            house_id = match.group(1)
            tag = match.group(2).lower()
            groups.setdefault(house_id, {})[tag] = filename
        else:
            logging.warning(f"Filename {filename} does not match expected pattern.")
    
    if not groups:
        logging.error("❌ No image pairs found. Check naming convention.")
        exit(1)
    
    for house_id, files in groups.items():
        if "before" in files and "after" in files:
            before_image = files["before"]
            after_image = files["after"]
            logging.debug(f"🚀 Processing pair for House {house_id}: {before_image} & {after_image}")
            
            before_mask = segment_image_mask(os.path.join(INPUT_FOLDER, before_image))
            after_mask = segment_image_mask(os.path.join(INPUT_FOLDER, after_image))
            
            if before_mask is None or after_mask is None:
                logging.error(f"❌ Segmentation failed for House {house_id}. Skipping...")
                continue
            
            damage_levels, total_cost, debris_count, cost_breakdown = compare_damage(
                os.path.join(INPUT_FOLDER, before_image),
                os.path.join(INPUT_FOLDER, after_image),
                before_mask,
                after_mask
            )
            
            if damage_levels is None:
                logging.error(f"❌ Damage data is missing for House {house_id}. Skipping PDF generation.")
                continue
            
            pdf_output_path = os.path.join(OUTPUT_FOLDER, f"damage_report_House{house_id}.pdf")
            generate_pdf_report(before_image, after_image, damage_levels, total_cost, cost_breakdown, pdf_output_path)
            logging.debug(f"✅ Damage assessment and PDF generation completed for House {house_id}!")
        else:
            logging.warning(f"House {house_id} does not have both Before and After images. Skipping...")