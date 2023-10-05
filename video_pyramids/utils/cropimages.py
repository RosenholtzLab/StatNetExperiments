import os
from PIL import Image

main_folder = 'C:\\Users\\chris\\Documents\\MIT\\Data\\data\\dtd_torch\\dtd\\dtd\\images'

for root, dirs, files in os.walk(main_folder):
    for file in files:
        if file.endswith('.jpg'):  # Or whatever file extension your images have
            path = os.path.join(root, file)

            # Open image
            img = Image.open(path)

            # Check if either dimension is smaller than 256, skip those images
            if img.size[0] < 256 or img.size[1] < 256:
                continue

            # Calculate coordinates for centered crop
            left = (img.width - 256)/2
            top = (img.height - 256)/2
            right = (img.width + 256)/2
            bottom = (img.height + 256)/2

            # Crop image to 256x256, from the center
            cropped_img = img.crop((left, top, right, bottom))

            # Save cropped image, overwriting the original
            cropped_img.save(path)
