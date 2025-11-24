import asyncio
import warnings
from abc import ABC
from typing import Dict, List, Set

lock = asyncio.Lock()


class ObjectDatabase(ABC):
    """Class for keeping track of objects in an environment.

    There are two types of object ids:
        (llm)
            format: f"{object_type}_{integer_id}"
            example: "Apple_1"
        (env)
            format: f"specific to the environment"
            example: "AlarmClock|-02.08|+00.94|-03.62" (for the env environment)

    I want to be able to map an llm id to it's corresponding
    env id and vice versa, so this class will take care of
    several common operations relating the two types of ids.
    """

    def __init__(self):
        self.reset()

    @property
    def llm_ids(self) -> Set[str]:
        return set(self.llm_to_env_dict.keys())

    @property
    def env_ids(self) -> Set[str]:
        return set(self.env_to_llm_dict.keys())

    def reset(self) -> None:
        # maps llm to corresponding env ids
        self.llm_to_env_dict = dict()
        # maps env to corresponding llm ids
        self.env_to_llm_dict = dict()
        # maps env object id to env object metadata dict
        self.obj_metadatas = dict()

    def ensure_llm_id(self, obj_id: str) -> str:
        """Takes in an LLM facing or environment facing object id and
        return the corresponding llm facing object id."""
        if obj_id in self.llm_ids:
            return obj_id
        elif obj_id in self.env_ids:
            return self.env_to_llm(obj_id)
        # check for lowercase and format
        elif obj_id.lower() in [x.lower() for x in self.llm_ids]:
            obj_id = [x for x in self.llm_ids if x.lower() == obj_id.lower()][0]
            return obj_id
        elif obj_id.lower() in [x.lower() for x in self.env_ids]:
            obj_id = [x for x in self.env_ids if x.lower() == obj_id.lower()][0]
            return self.env_to_llm(obj_id)
        else:
            raise KeyError(f"{obj_id} not a recorded object id.")

    def ensure_env_id(self, obj_id: str) -> str:
        """Takes in an LLM facing or environment facing object id and
        return the corresponding env facing object id."""
        if obj_id in self.llm_ids:
            return self.llm_to_env(obj_id)
        elif obj_id in self.env_ids:
            return obj_id
        # check for lowercase and format
        elif obj_id.lower() in [x.lower() for x in self.llm_ids]:
            obj_id = [x for x in self.llm_ids if x.lower() == obj_id.lower()][0]
            return self.llm_to_env(obj_id)
        elif obj_id.lower() in [x.lower() for x in self.env_ids]:
            obj_id = [x for x in self.env_ids if x.lower() == obj_id.lower()][0]
            return obj_id
        else:
            raise KeyError(f"{obj_id} not a recorded object id.")

    def env_to_llm(self, env_id: str) -> str:
        return self.env_to_llm_dict[env_id]

    def llm_to_env(self, llm_id: str) -> str:
        return self.llm_to_env_dict[llm_id]

    def get_env_from_llm(self, llm_id: str) -> str:
        if llm_id not in self.llm_ids:
            raise KeyError(f"{llm_id} not found in list of object ids.")
        return self.llm_to_env_dict[llm_id]

    def get_llm_from_env(self, env_id: str) -> str:
        if env_id not in self.env_ids:
            raise KeyError(f"{env_id} not found in list of object ids.")
        return self.env_to_llm_dict[env_id]

    def check_if_llm_id_exists(self, llm_id: str) -> bool:
        return True if llm_id in self.llm_to_env_dict else False

    def check_if_env_id_exists(self, env_id: str) -> bool:
        return True if env_id in self.env_to_llm_dict else False
    
    def check_if_id_exists(self, id: str) -> bool:
        if id in self.llm_ids or id in self.env_ids:
            return True
        return False

    def get_llm_ids(self) -> List[str]:
        return [x for x, _ in self.llm_to_env_dict.items()]

    def get_env_ids(self) -> List[str]:
        return [x for x, _ in self.env_to_llm_dict.items()]

    def get_obj_type_count(self, obj_type: str) -> int:
        """Checks to see the number of objects of a certain type
        already present in the list of objects."""
        count = len([x for x in self.llm_to_env_dict if obj_type == x.split("_")[0]])
        return count

    def get_obj_metadata(self, obj_id: str) -> Dict:
        """Given an object id in either env or llm format, return
        the dictionary of object metadata.

        Args:
            obj_id (str): object id in llm or env format

        Returns:
            object_metadata: env object metadata corresponding to
                query object id
        """
        if obj_id in self.llm_to_env_dict:
            env_id = self.llm_to_env_dict[obj_id]
            return self.obj_metadatas[env_id]
        if obj_id in self.env_to_llm_dict:
            return self.obj_metadatas[obj_id]
        raise KeyError(f"{obj_id} not found in list of object ids.")

    def get_obj_metadata_list(self) -> List[Dict]:
        """Return a list of dictionaries representing the object metadata
        of every discovered object contained in the object database.

        Returns:
            object_metadata_list (List[Dict]): a list of dictionaries
                representing the metadata of all discovered objects in the
                object state
        """
        return [v for _, v in self.obj_metadatas.items()]

    def add_obj(self, env_obj_metadata: Dict) -> None:
        env_id = env_obj_metadata["objectId"]
        if env_id in self.env_to_llm_dict:
            warnings.warn(f"{env_id} already present in list of object ids.")
            return
        obj_type = env_obj_metadata["objectType"]
        obj_type_cnt = self.get_obj_type_count(obj_type)
        llm_id = f"{obj_type}_{obj_type_cnt+1}"  # idx starts at 1
        # add to dictionaries
        self.llm_to_env_dict[llm_id] = env_id
        self.env_to_llm_dict[env_id] = llm_id
        self.obj_metadatas[env_id] = env_obj_metadata

    def update_obj_metadata_list(self, obj_mdata_list: List[Dict]) -> None:
        """Given an env controller, update the metadata of all objects
        currently in the list of objects."""
        for obj_mdata in obj_mdata_list:
            env_id = obj_mdata["objectId"]
            if env_id not in self.env_ids:
                continue
            self.obj_metadatas[env_id] = obj_mdata

    def update_obj_mdata(
        self, obj_id: str, obj_attr: str, *args
    ) -> None:
        """Takes in an object name, an object attribute, and a target value. If the
        attribute is already present in the metadata of the specified object, the
        attribute's value will be overwritten with the new object attribute value. If
        the attribute is not present, the attribute will be added with the specified
        object attribute value.

        Args:
            obj_id (str): the name of the object
            obj_attr (str): the name of the object attribute
            *args : values for arguments for the pddl predicate

        Returns:
            None
        """
        obj_id = self.ensure_env_id(obj_id)
        obj_mdata = self.get_obj_metadata(obj_id)
        args = list(args)
        if len(args) == 1 and isinstance(args[0], bool):
            obj_mdata[obj_attr] = args[0]
        else:
            obj_mdata[obj_attr] = args
        self.obj_metadatas[obj_id] = obj_mdata
