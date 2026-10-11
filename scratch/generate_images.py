import os
from PIL import Image, ImageDraw, ImageFont

def create_placeholder(text, path, color=(50, 50, 50)):
    # Create a 300x300 image
    img = Image.new('RGB', (300, 300), color=color)
    d = ImageDraw.Draw(img)
    
    # Try to load a generic font, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except IOError:
        font = ImageFont.load_default()
    
    # Simple text wrapping / centering
    try:
        text_bbox = d.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
    except AttributeError:
        # Fallback for older Pillow
        text_width, text_height = d.textsize(text, font=font)
        
    x = (300 - text_width) / 2
    y = (300 - text_height) / 2
    
    d.text((x, y), text, fill=(255, 255, 255), font=font)
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
    print(f"Created: {path}")

def main():
    base_dir = "data/sign_images"
    languages = ["ASL", "ISL"]
    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [
        "PHRASE_HELP", "PHRASE_THANK_YOU", "PHRASE_WATER", 
        "PHRASE_WASHROOM", "PHRASE_YES", "PHRASE_NO"
    ]
    
    for lang in languages:
        for label in labels:
            path = os.path.join(base_dir, lang, f"{label}.png")
            create_placeholder(f"[{lang}]\n{label}", path, color=(30, 100, 60))

if __name__ == "__main__":
    main()
