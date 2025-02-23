import ee
import requests
from PIL import Image
from io import BytesIO
import os

# PositionStack API key (replace with your actual key)
API_KEY = '29e5b2182f173975fa9e719c401a3769'

def get_naip_image_for_address(address, start_date='2022-01-01', end_date='2023-12-31', output_folder='results'):
    """
    Given a street address, geocode it with PositionStack, then fetch NAIP imagery
    from Earth Engine and download the image locally.
    
    Returns:
        A tuple (filename, message) where filename is the path to the downloaded image
        (or None if something went wrong), and message is a status string.
    """
    # 1. Geocode the address using PositionStack
    url = f'https://api.positionstack.com/v1/forward?access_key={API_KEY}&query={address}'
    response = requests.get(url)

    if response.status_code != 200:
        return None, f"Geocoding Error: {response.status_code}"

    data = response.json()
    if not data['data']:
        return None, "Address not found by PositionStack."

    latitude = data['data'][0]['latitude']
    longitude = data['data'][0]['longitude']
    if not latitude or not longitude:
        return None, "Invalid coordinates from geocoding."

    # 2. Initialize Earth Engine
    try:
        ee.Initialize()
    except Exception as e:
        return None, f"Earth Engine initialization error: {e}"

    # 3. Define the geometry and load NAIP imagery
    geometry = ee.Geometry.Point([longitude, latitude]).buffer(50)
    image_collection = (ee.ImageCollection("USDA/NAIP/DOQQ")
                        .filterBounds(geometry)
                        .filterDate(start_date, end_date))
    
    size_info = image_collection.size().getInfo()
    if size_info == 0:
        return None, "No valid NAIP images found in that date range."

    # 4. Get the first image
    image = image_collection.first()
    info = image.getInfo()
    if info is None:
        return None, "No valid NAIP image found for the specified date range."

    # 5. Clip and get the thumbnail URL
    image = image.clip(geometry)
    vis_params = {
        'bands': ['R', 'G', 'B'],
        'min': 0,
        'max': 255,
        'scale': 1,  # NAIP ~1 m per pixel
    }
    thumb_url = image.getThumbURL(vis_params)
    if not thumb_url:
        return None, "Failed to get a thumbnail URL from Earth Engine."

    # 6. Download the image
    r = requests.get(thumb_url)
    if r.status_code != 200:
        return None, f"Failed to retrieve image from NAIP (status code {r.status_code})."

    # 7. Save and optionally show
    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.join(output_folder, f"naip_house_image_{start_date}_{end_date}.png")
    with open(filename, "wb") as file:
        file.write(r.content)

    # Optionally load into PIL for further processing or display
    # img = Image.open(BytesIO(r.content))
    # img.show()

    return filename, f"Successfully saved image as {filename}"