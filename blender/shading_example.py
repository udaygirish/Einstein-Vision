import bpy

BASE_PATH = "/home/udaygirish/Projects/WPI/computer_vision/project3/"
DATA_PATH = BASE_PATH + "P3Data/"
ASSETS_PATH = DATA_PATH + "Assets/"
# Load the .blend file containing the object
blend_file_path = ASSETS_PATH+ "StopSign.blend" # Replace this with the path to your .blend file
object_name = "StopSign"  # Replace this with the name of the object you want to add texture to

# Append the object from the .blend file
with bpy.data.libraries.load(blend_file_path, link=False) as (data_from, data_to):
    data_to.objects = [name for name in data_from.objects if name.startswith(object_name)]

# Link the object to the scene
for obj in data_to.objects:
    if obj is not None:
        bpy.context.collection.objects.link(obj)

# Assign a material to the object
if len(obj.data.materials) == 0:
    mat = bpy.data.materials.new(name="Material")
    obj.data.materials.append(mat)
else:
    mat = obj.data.materials[0]

# Load the image
image_path = ASSETS_PATH+ "StopSignImage.png"  # Replace this with the path to your image
image = bpy.data.images.load(image_path)

# Create a new texture
texture = bpy.data.textures.new(name="Texture", type='IMAGE')
texture.image = image

# Create a new texture slot
tex_slot = mat.texture_slots.add()
tex_slot.texture = texture

# Set texture coordinates (optional)
tex_slot.texture_coords = 'UV'

# Set mapping (optional)
tex_slot.mapping = 'FLAT'

# Adjust other properties as needed
tex_slot.scale = (1.0, 1.0, 1.0)  # Scale of the texture

# Apply the changes
bpy.context.view_layer.update()
