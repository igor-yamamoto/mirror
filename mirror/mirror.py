import pandas as pd
from uuid import uuid4
from datetime import datetime
from mirror.ground import Ground
import mirror.functions as f
import numpy as np


class Mirror(Ground):
    """
    Mirror main class.

    Parameters:
        |- ground: Ground object.
        |- mirror: Pandas Dataframe of the mirror dataset. Type: `pandas.DataFrame`
        |- score (optional, default `abs`): Scoring method for assertivity evaluation. The following methods are suported:
            a) "abs": binary comparison (0 for divergence, 1 for assertion)
            b) "sequence-matcher":  implementation of `difflib.SequenceMatcher` algorithm
    """
    def __init__(self, ground, mirror: pd.DataFrame, score="abs"):
        self.ground = ground.ground

        self.mirror = mirror

        self.keys = []
        self.columns = []
        self.attributes = []

        mirror_columns = self.mirror.columns

        for column in mirror_columns:
            if column in ground.columns:
                self.columns.append(column)
                if column in ground.keys:
                    self.keys.append(column)
                elif column in ground.attributes:
                    self.attributes.append(column)

        self.extract_map()
        self.extract_stats(score)

    def extract_map(self, score):
        ground_df = self.ground
        mirror_df = self.mirror

        join = pd.merge(ground_df, mirror_df, on=self.keys, how="outer", suffixes=(
            "_ground", "_mirror"), indicator=True)

        raw_map_order = []

        for coll in self.attributes:
            if score == "abs":
                join[coll] = f.apply_abs_eval(
                    join, coll+"_ground", coll+"_mirror"
                )

            elif score == "sequence-matcher":
                join[coll] = f.apply_sequence_matcher_eval(
                    join, coll+"_ground", coll+"_mirror"
                )

            raw_map_order += [coll+"_ground", coll+"_mirror", coll]

        join["_mean"] = join[self.attributes].mean(axis=1)
        self.raw_map = join[self.keys+["_mean", "_merge"]+raw_map_order]
        self.map = join[self.keys+["_mean", "_merge"]+self.columns]
        self.scoring_method = score

    def extract_stats(self):
        stats_dict = {
            "volume": {
                "ground": len(self.ground),
                "mirror": len(self.mirror)
            },
            "key_matching": {
                "matched": len(self.raw_map[self.raw_map["_merge"] == "both"]),
                "unmatched_ground": len(self.raw_map[self.raw_map["_merge"] == "left_only"]),
                "unmatched_mirror": len(self.raw_map[self.raw_map["_merge"] == "right_only"])
            },
            "field_assertivity": {}
        }

        if self.attributes:
            stats_info = (
                self.map[self.map["_merge"] == "both"]
            ).describe().loc[["count", "mean"]]

            for attribute in self.attributes:
                assertivity = 100*stats_info[attribute].loc[["mean"]][0]
                stats_dict["field_assertivity"][attribute] = assertivity

        self.stats = stats_dict


    def print_stats(self, assertivity_subset=["*"]):
        """
        Method that prints basic statistics over comparison.

        Parameters: 
            |- assertivity_subset (optional, default `["*"]`): array containing which attributes should be selected for printing statistics. Type: `array` 
        """
        print("VOLUMETRY:")
        print("\t|- Ground truth: {}".format( self.stats["volume"]["ground"] ))
        print("\t|- Mirror: {}".format( self.stats["volume"]["mirror"] ))

        print("\nKEY MATCHING:")
        print("\t|- Matched keys: {} ({}%)".format(
            self.stats["key_matching"]["matched"], 
            int(100*self.stats["key_matching"]["matched"]/self.stats["volume"]["ground"])
        ))
        print("\t|- Unmatched keys (ground): {} ({}%)".format(
            self.stats["key_matching"]["unmatched_ground"], 
            int(100*self.stats["key_matching"]["unmatched_ground"]/self.stats["volume"]["ground"])
        ))
        print("\t|- Unmatched keys (mirror): {} ({}%)".format(
            self.stats["key_matching"]["unmatched_mirror"], 
            int(100*self.stats["key_matching"]["unmatched_mirror"]/self.stats["volume"]["mirror"])
        ))

        if ((len(assertivity_subset) == 1) and (assertivity_subset[0]=="*")):
            assertivity_subset = self.attributes
        
        if ((self.attributes) and (assertivity_subset)):
            print("\nFIELD ASSERTIVITY: {}".format(self.scoring_method))
            for attribute in self.attributes:
                if attribute in assertivity_subset:
                    print("\t|- {}: {}".format(attribute, self.stats["field_assertivity"][attribute]))

    def map_field_error_frequency(self):
        errors = self.map[self.map["_merge"]=="both"][self.map["_mean"]<1]

        def map_cols_with_error(row):
            errors_col = ''
            for col in errors.columns:
                if (col not in self.keys) & ("_" != col[0:1]):
                    if row[col]<1:
                        errors_col = errors_col + col + " "
            errors_col = errors_col.strip()
            if not errors_col:
                errors_col = None
            return errors_col
        errors["_fields_with_error"] = errors.apply(map_cols_with_error, axis=1)
        errors = errors.groupby("_fields_with_error").count()[[self.keys]].sort_values(by=self.keys[0], ascending=False)
        self.error_frequency = errors

    def inspect_divergence_on_field(self, field: str, hide_keys=False, add_fields=[]):
        """
        Method that returns a dataframe listing all records that are divergent over an attribute.

        Parameters:
            |- field: Name of the field. Type: `string`
            |- hide_keys (optional, default `False`): Boolean specifying if keys should be hidden. Type: `boolean`
            |- add_fields (optional, default `[]`): Array containing all aditional columns to be selected. Type: `array`
        """
        if hide_keys:
            keys=[]
        else: 
            keys=self.keys

        display_fields = keys+add_fields+[field+"_ground",field+"_mirror",field]
        divergence_df = self.raw_map[self.raw_map["_merge"]=="both"][self.raw_map[field]<1][display_fields]
        return divergence_df