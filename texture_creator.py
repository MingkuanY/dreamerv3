from PIL import Image

size = (16, 16)

yellow_color = (129, 34, 141)
img = Image.new("RGB", size, yellow_color)

img.save("crafter-main/crafter/assets/purple.png")
