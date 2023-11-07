import os
from PIL import Image


def get_img_size(path):
    width, height = Image.open(path).size
    return width, height  # Return both width and height


main_folder = 'C:\\Users\\chris\\Documents\\MIT\\Data\\data\\dtd_torch\\dtd\\dtd\\images'

# Use os.walk() to get paths to all image files in all subfolders
the_paths = []
small_images = 0  # Initialize counter for small images
for root, dirs, files in os.walk(main_folder):
    for file in files:
        if file.endswith('.jpg'):  # Or whatever file extension your images have
            path = os.path.join(root, file)
            the_paths.append(path)

            # Get image size
            width, height = get_img_size(path)

            # Check if either dimension is smaller than 512
            if width < 256 or height < 256:
                small_images += 1
                print(path)# If true, increment the counter

total_images = len(the_paths)  # Number of total images

smallest = min(the_paths, key=get_img_size)  # smallest will hold the path to the smallest image

print("Total number of images:", total_images)
print("Number of images smaller than 512 in either dimension:", small_images)


