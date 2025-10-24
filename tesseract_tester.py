import pytesseract
from PIL import Image, ImageDraw

# 1. Tell pytesseract where the .exe is (this is the key step)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 2. Create a simple test image in memory
img = Image.new("RGB", (200, 100), color="white")
d = ImageDraw.Draw(img)
d.text((10, 10), "123m test", fill="black")

# 3. Run OCR on that image
text = pytesseract.image_to_string(img)

# 4. Print the result
print(text)
