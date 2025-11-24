from typing import List, Dict, Optional, Any

def check_by_obj_id_and_attribute(obj_metadata_list: List[Dict],
                                  ai2thor_obj_id: str, 
                                  attribute: str) -> bool:
    """Checks if an attribute of a specific object specified by ai2thor
    object id is true or not. Returns True if True, False if False."""
    for obj_metadata in obj_metadata_list:
        if obj_metadata['objectId'] == ai2thor_obj_id:
            return True if obj_metadata[attribute] else False
    assert False, "Error: ai2thor object id must be wrong. Check your function."

def check_by_obj_type_and_attribute(obj_metadata_list: List[Dict],
                                    obj_type: str,
                                    attribute: str,
                                    attribute_value: Optional[Any] = None) -> bool:
    """Checks if an attribute of any object with a specific object type is True.
    If any object with the specified object type has the attribute as True, return
    True. Otherwise, return False."""
    for obj_metadata in obj_metadata_list:
        if obj_type in obj_metadata['objectType'].lower():
            if attribute_value is not None:
                if obj_metadata[attribute] == attribute_value:
                    return True
            elif obj_metadata[attribute]:
                return True
    return False

def check_if_obj_type_in_obj_type(obj_metadata_list: List[Dict],
                                  obj_type: str,
                                  obj_type_receptacle: str) -> bool:
    """Checks if there are any object types inside the object receptacle type."""
    for obj_metadata in obj_metadata_list:
        if obj_type in obj_metadata['objectType'].lower():
            parent_receptacles = obj_metadata['parentReceptacles']
            if parent_receptacles is None:
                continue
            for receptacle in parent_receptacles:
                if obj_type_receptacle.lower() in receptacle.lower():
                    return True
    return False

def check_if_obj_id_is_empty(obj_metadata_list: List[Dict],
                             ai2thor_obj_id: str) -> bool:
    for obj_metadata in obj_metadata_list:
        if obj_metadata['objectId'] == ai2thor_obj_id:
            receptacle_obj_ids = obj_metadata['receptacleObjectIds']
            if receptacle_obj_ids is None or len(receptacle_obj_ids) == 0:
                return True
            else:
                return False
    assert False, "Error: Your ai2thor object id must be wrong. Please check it."

#############

# VERIFICATION FUNCTIONS FOR DIFFICULT TASKS

#############

# def task_000_verification(obj_mdata: list[dict]) -> bool:
#     check_by_obj_type_and_attribute(obj_mdata, "egg", "temperature", "Hot")

# def task_001_verification(obj_mdata: list[dict]) -> bool:
#     pass

def hard_task_004_verification(obj_mdata: list[dict]) -> bool:
    check_1 = check_by_obj_type_and_attribute(obj_mdata, "tomato", "temperature", "Cold")
    return check_1

def hard_task_012_verification(obj_mdata: list[dict]) -> bool:
    check_1 = check_by_obj_type_and_attribute(obj_mdata, "mug", "fillLiquid", "coffee")
    return check_1