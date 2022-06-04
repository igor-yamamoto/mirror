import pandas as pd
from difflib import SequenceMatcher
from uuid import uuid4
from datetime import datetime

class Ground():
    """
    Ground main class.

    Parameters:
        |- ground: Pandas Dataframe of the ground truth dataset. Type: `pandas.DataFrame`
        |- keys: Columns to be considered as keys. Type: `list`
    """
    def __init__(self, ground: pd.DataFrame, keys: list):
        self.ground = ground
        self.keys = keys

        self.columns = list(self.ground.columns)
        
        attributes = list(self.ground.columns)
        for col in self.columns:
            if col in self.keys:
                attributes.remove(col)

        self.attributes = attributes

        self.mirror_instances = {}

    def add_mirror(self, mirror: pd.DataFrame, label=''):
        """
        Method to add new mirror instances into the `mirror_instance` attribute.

        Parameters: 
            |- mirror: Pandas Dataframe of the mirror dataset. Type: `pandas.Dataframe`
            |- label (optional, default `''`): Label for the new mirror instance. Type: `string`
        """

        mirror_uuid = str(uuid4())
        creation_datetime = datetime.strftime(datetime.now(), "%Y-%M-%dT%H:%m:%S")

        mirror_instances = self.mirror_instances 

        if not label:
            instances_keys = mirror_instances.keys()
            label = "mirror_{}".format(len(instances_keys)+1)

        new_mirror_instance = {
            "createdAt": creation_datetime,
            "updatedAt": creation_datetime,
            "uuid": mirror_uuid,
            "dataframe": mirror
        }

        self.mirror_instances[label] = new_mirror_instance
