"""
Loops through ai2thor floorplans and saves out the metadata from each floorplan
for use in task creation.
"""

import os
import json

from ai2thor.controller import Controller

c = Controller()

kitchens = [f"FloorPlan{i}" for i in range(1, 31)]
living_rooms = [f"FloorPlan{200 + i}" for i in range(1, 31)]
bedrooms = [f"FloorPlan{300 + i}" for i in range(1, 31)]
bathrooms = [f"FloorPlan{400 + i}" for i in range(1, 31)]

# scenes = kitchens + living_rooms + bedrooms + bathrooms
scenes = kitchens

# create folder for saving metadata
scene_obj_metadata_folder = "ai2thor_metadata/scene_object_metadata"
if not os.path.exists(scene_obj_metadata_folder):
    os.makedirs(scene_obj_metadata_folder)

for scene in scenes:
    event = c.reset(scene=scene)
    obj_mdata = event.metadata["objects"]
    obj_list = [x["objectId"]+"\n" for x in obj_mdata]
    save_path_obj_metadata = os.path.join(scene_obj_metadata_folder, scene+"_obj_metadata.json")
    with open(save_path_obj_metadata, "w") as file:
        json.dump(obj_mdata, file, indent=3)
    save_path_obj_list = os.path.join(scene_obj_metadata_folder, scene+"_obj_list.txt")
    with open(save_path_obj_list, "w") as file:
        file.writelines(obj_list)