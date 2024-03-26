import bpy
import random 

class BlenderObj:
    def __init__(self):
        pass 
    
    def create_road_lanes(self):
        bpy.ops.mesh.primitive_plane_add(size=100, enter_editmode=False, align='WORLD', location=(0, 0, 0))
        road_surface = bpy.context.object
        road_surface.name = "Road_Surface"
        
#         Assign material to road surface
        road_material = bpy.data.materials.new(name="Road_Material")
        road_material.diffuse_color = (0, 0, 0, 1)  # Black color
        road_surface.data.materials.append(road_material)
        
#         Create lane markings
        bpy.ops.mesh.primitive_cube_add(size=0.1, enter_editmode=False, align='WORLD', location=(0, 0, 0))
        lane_marking = bpy.context.object
        lane_marking.name = "Lane_Marking"
        lane_marking.scale = (1, 4, 0.6)
        
#        Assign material to lane markings
        lane_material = bpy.data.materials.new(name="Lane_Material")
        lane_material.diffuse_color = (1, 1, 1, 1)  # White color
        lane_marking.data.materials.append(lane_material)
        
#         Create multiple lane markings
        for i in range(-999, 1000):
            new_lane = lane_marking.copy()
            new_lane.data = lane_marking.data.copy()
            bpy.context.collection.objects.link(new_lane)
            new_lane.location = (0, i * 1.0, 0)  # Adjust the spacing between lanes
       
    def setup_camera_frontal(self):
#        Create camera
        bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(0, 0, 0))
        camera = bpy.context.object
        camera.name = "Camera_F"
        
#        Position camera inside the car's dashboard
#      if view == 'Frontal':
        camera.location = (0, -5, 0.5)  # Adjust position as needed
        camera.rotation_euler = (1.57, 0, 0)  # Adjust rotation to face the road
    
    def setup_camera_oblique(self):
        bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(0, 0, 0))
        camera = bpy.context.object
        camera.name = "Camera_O"
        camera.location = (3, -4, 1)  # Adjust position as needed
        camera.rotation_euler = (1.57,0, 0.75) 
    
            
    def SpawnObject(self, file_path, object_names, location, scale):
        
        with bpy.data.libraries.load(file_path) as (data_from, data_to):
            data_to.objects = data_from.objects
        for i, obj in enumerate(data_to.objects):
            bpy.context.collection.objects.link(obj)
            obj.location = location
            obj.scale = scale
        return data_to.objects
    
    def switch_camera(self, camera1, camera2, frame_start, frame_end, view):
        # Set the output file path and format
        bpy.context.scene.frame_start = 1
        bpy.context.scene.frame_end = 100
        output_path = "//output_"  # Output file path without extension
        file_format = 'PNG'  # Output file format
        if view == "Frontal":
            for frame in range(frame_start, frame_end+1):
                bpy.context.scene.camera = camera1
        elif view == "Oblique":
            for frame in range(frame_start, frame_end+1):
                bpy.context.scene.camera = camera2
                
            bpy.context.scene.frame_set(frame)
            
            # Set the output file path for the current frame
            bpy.context.scene.render.filepath = f"{output_path}{frame:04d}.{file_format.lower()}"
            
            # Render the current frame
            bpy.ops.render.render(write_still=True)


bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

blender_obj = BlenderObj()

blender_obj.create_road_lanes()

blender_obj.setup_camera_frontal()
blender_obj.setup_camera_oblique()

camera1 = bpy.data.objects.get("Camera_F")
camera2 = bpy.data.objects.get("Camera_O")



vehicles = ["Bicycle", "PickupTruck", "SedanAndHatchback", "SUV", "Truck"]
entities = ["Dustbin", "Pedestrain", "SpeedLimitSign", "StopSign", "TrafficAssets", "TrafficConeAndCylinder", "TrafficSignal"]

#pedestrians = blender_obj.SpawnObject("/home/pradnya/P3_Scripts/P3Data/Assets/Pedestrain.blend", ["Pedestrian"], (10, 4, 0), (1, 1, 1))

# Import traffic lights from .blend file
#traffic_assets = blender_obj.import_objects_from_blend("./P3Data/Assets/TrafficAssets.blend", ["TrafficAssets"], (0, 0, 0), (1, 1, 1))

# Import vehicles from .blend file
vehicles = blender_obj.SpawnObject("/home/pradnya/P3_Scripts/P3Data/Assets/Vehicles/SUV.blend", ["SUV"], (-2, 3, 0), (1, 1, 1))


frame_start = 1 
frame_end = 100
view = "Frontal"
blender_obj.switch_camera(camera1, camera2, start, end, view)
