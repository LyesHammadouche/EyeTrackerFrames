from PIL import Image
import os

src = "interface elements/Eye-fb-round.png"
dst = "app_icon.ico"

if os.path.exists(src):
    img = Image.open(src)
    img.save(dst, format='ICO', sizes=[(256, 256)])
    print(f"Created {dst}")
else:
    print(f"Source image not found: {src}")
