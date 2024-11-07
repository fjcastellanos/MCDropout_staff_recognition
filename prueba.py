from PIL import Image, ImageDraw  # Para crear y manipular imágenes
import matplotlib.pyplot as plt   # Para crear la notación matemática como imagen




animal_img = Image.new('RGB', (200, 200), color=(200, 180, 150))
draw = ImageDraw.Draw(animal_img)
draw.text((20, 90), "Animal Image", fill="black")

# Create classification type placeholder (right side)
classification_img = Image.new('RGB', (200, 200), color=(180, 200, 150))
draw = ImageDraw.Draw(classification_img)
draw.text((20, 90), "Type: Animal", fill="black")

# Update notation to include example features below
fig, ax = plt.subplots(figsize=(5, 3))
ax.axis('off')
notation_text = (r"$f : \mathcal{X} \rightarrow \mathcal{Y}$" + "\n" +
                 r"$f : \mathcal{X} \rightarrow \mathcal{F} \rightarrow \mathcal{Y}$" + "\n\n" +
                 "Features: Color, Ear Shape, Size")
ax.text(0.5, 0.5, notation_text, ha='center', va='center', fontsize=15)

# Save the updated notation as an image
notation_image_path = "pruebas/updated_notation_image_with_features.png"
plt.savefig(notation_image_path, dpi=300, bbox_inches='tight')
plt.close(fig)

# Load the updated notation image
notation_img = Image.open(notation_image_path)

# Combine the images: Animal image (left), Notation with features (center), Classification type (right)
total_width = animal_img.width + notation_img.width + classification_img.width + 40  # adding padding between images
max_height = max(animal_img.height, notation_img.height, classification_img.height)

# Create a blank canvas for combined image
final_img = Image.new('RGB', (total_width, max_height), color=(255, 255, 255))

# Paste images on the canvas
final_img.paste(animal_img, (0, (max_height - animal_img.height) // 2))
final_img.paste(notation_img, (animal_img.width + 20, (max_height - notation_img.height) // 2))
final_img.paste(classification_img, (animal_img.width + notation_img.width + 40, (max_height - classification_img.height) // 2))

# Save the final image
final_img_path = "pruebas/final_classification_with_features_image.png"
final_img.save(final_img_path)

final_img.show()