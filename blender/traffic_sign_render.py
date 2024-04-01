import bpy


base_path = "/home/udaygirish/Projects/WPI/computer_vision/project3/P3Data/Assets_Mod/traffic_signals/"
# Path to UV edited image
image_path = base_path + "green.png"  # Change this to your image path

# Create or select image
image = bpy.data.images.get("UV_Edit_Colormap")
if image is None:
    image = bpy.data.images.load(image_path)
    image.name = "UV_Edit_Colormap"

# Apply the image texture to the active object's material
obj = bpy.context.active_object
if obj is not None and obj.type == 'MESH':
    mat = obj.active_material
    if mat is None:
        mat = bpy.data.materials.new(name="Material")
        obj.active_material = mat

    # Create a new principled shader node
    principled_bsdf = mat.node_tree.nodes.get('Principled BSDF')
    if principled_bsdf is None:
        principled_bsdf = mat.node_tree.nodes.new('ShaderNodeBsdfPrincipled')
    
    # Create an image texture node
    texture_node = mat.node_tree.nodes.new('ShaderNodeTexImage')
    texture_node.image = image
    
    # Connect the image texture node to the base color input of the principled shader node
    mat.node_tree.links.new(texture_node.outputs['Color'], principled_bsdf.inputs['Base Color'])