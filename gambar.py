from PIL import Image, ImageDraw, ImageFont
import os

def create_synthetic_shopping_list(items, output_path):
    img = Image.new('RGB', (400, 300), color='white')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 20)
    y = 20
    for item in items:
        draw.text((20, y), item, fill='black', font=font)
        y += 30
    img.save(output_path)

items = ["Susu UHT 1L", "Beras 5kg", "Sabun Cair 500ml"]
os.makedirs("data/images", exist_ok=True)
create_synthetic_shopping_list(items, "data/images/synthetic_list_1.png")