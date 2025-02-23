from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import logging

# Import functions from your damage detection pipeline (predict_deeplab.py)
from predict_deeplab import segment_image_mask, compare_damage, generate_pdf_report

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Define folder paths (adjust these if necessary)
BASE_DIR = os.getcwd()
INPUT_FOLDER = os.path.join(BASE_DIR, "cropped_images")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "results")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def get_demo_images(address):
    """
    Match real-world addresses to predefined demo images.
    """
    address_lower = address.lower().strip()
    
    if "9880 sw 87th ave" in address_lower:  # Damage Report 1
        before = os.path.join(INPUT_FOLDER, "Houses1_Before.jpg")
        after = os.path.join(INPUT_FOLDER, "Houses1_After.jpg")
    elif "15620 kinross cir" in address_lower:  # Damage Report 2
        before = os.path.join(INPUT_FOLDER, "Houses2_Before.jpg")
        after = os.path.join(INPUT_FOLDER, "Houses2_After.jpg")
    else:
        # Simulate a real address: use the same image for before and after.
        before = os.path.join(INPUT_FOLDER, "Houses1_Before.jpg")
        after = before

    logging.debug(f"Selected images for address '{address}': {before} and {after}")
    return before, after

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        address = request.form.get("address")
        logging.debug(f"Address received: {address}")
        before_path, after_path = get_demo_images(address)
        if before_path is None or after_path is None:
            return "Failed to retrieve images.", 500
        # Pass only the filenames (assuming images are in the INPUT_FOLDER)
        return redirect(url_for("report", before=os.path.basename(before_path), after=os.path.basename(after_path)))
    return render_template("index.html")

@app.route("/report")
def report():
    before_file = request.args.get("before")
    after_file = request.args.get("after")
    if not before_file or not after_file:
        return "Missing image information.", 400

    before_path = os.path.join(INPUT_FOLDER, before_file)
    after_path = os.path.join(INPUT_FOLDER, after_file)

    # Run segmentation to get masks from the images
    before_mask = segment_image_mask(before_path)
    after_mask = segment_image_mask(after_path)
    if before_mask is None or after_mask is None:
        return "Segmentation failed.", 500

    # Compare damage using your pipeline (this returns damage data, total cost, etc.)
    damage_levels, total_cost, debris_count, cost_breakdown = compare_damage(
        before_path, after_path, before_mask, after_mask
    )
    if damage_levels is None:
        return "Damage data is missing.", 500

    # Generate the PDF report
    pdf_filename = f"damage_report_{os.path.splitext(before_file)[0]}.pdf"
    pdf_output_path = os.path.join(OUTPUT_FOLDER, pdf_filename)
    generate_pdf_report(before_file, after_file, damage_levels, total_cost, cost_breakdown, pdf_output_path)

    # Serve the PDF report to the user as a download
    return send_file(pdf_output_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)