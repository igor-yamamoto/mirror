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
    def __init__(self, ground, mirror: pd.DataFrame, score={}):
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

        if not score:
            for attribute in self.attributes:
                score[attribute] = ["abs"]
        else:
            for attribute in self.attributes:
                if attribute not in score.keys():
                    score[attribute] = ["abs"]
        
        self.score = score

        self.extract_attribute_columns()
        self.extract_map()
        self.extract_stats()

    def extract_attribute_columns(self):
        attribute_columns = {}

        for attribute in self.attributes:
            attribute_columns[attribute] = {}
            attribute_columns[attribute]["ground"] = attribute+"__ground"
            attribute_columns[attribute]["mirror"] = attribute+"__mirror"
            score_cols = {}
            for score in self.score[attribute]:
                score_cols[score] = "_"+attribute+"__"+score 
            attribute_columns[attribute]["score"] = score_cols
        
        self.attribute_columns = attribute_columns

    def extract_map(self):
        ground_df = self.ground
        mirror_df = self.mirror

        join = pd.merge(ground_df, mirror_df, on=self.keys, how="outer", suffixes=(
            "__ground", "__mirror"), indicator=True)

        raw_map_order = []
        scoring_cols = []

        for coll in self.attributes:
            attribute_scorings = self.score[coll]
            attribute_score_cols = []

            ground_col = self.attribute_columns[coll]["ground"]
            mirror_col = self.attribute_columns[coll]["mirror"]

            for scoring in attribute_scorings:
                scored_col_name = self.attribute_columns[coll]["score"][scoring]#coll + "__s_" + scoring

                if scoring == "abs":
                    join[scored_col_name] = f.apply_abs_eval(
                        join, ground_col, mirror_col
                    )

                elif scoring == "sequence-matcher":
                    join[scored_col_name] = f.apply_sequence_matcher_eval(
                        join, ground_col, mirror_col
                    )

                attribute_score_cols.append(scored_col_name)

            raw_map_order = raw_map_order + [ground_col, mirror_col] + attribute_score_cols
            scoring_cols += attribute_score_cols

        join["_mean"] = join[scoring_cols].mean(axis=1)
        self.raw_map = join[self.keys+["_mean", "_merge"]+raw_map_order]
        self.map = join[self.keys+["_mean", "_merge"]+scoring_cols]

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
                stats_dict["field_assertivity"][attribute] = {}
                for score in self.score[attribute]:
                    score_col = self.attribute_columns[attribute]["score"][score]
                    score_val = 100*stats_info[score_col].loc[["mean"]][0]
                    stats_dict["field_assertivity"][attribute][score] = score_val

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
            print("\nFIELD ASSERTIVITY:")
            for attribute in self.attributes:
                if attribute in assertivity_subset:
                    print("\t|- {}:".format(attribute))
                    for score in self.score[attribute]:
                        print("\t\t|- {}: {}%".format(score, self.stats["field_assertivity"][attribute][score]))

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

    def inspect_divergence_on_field(self, field: str, score="", hide_keys=False, add_fields=[]):
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

        ground_col = self.attribute_columns[field]["ground"]
        mirror_col = self.attribute_columns[field]["mirror"]

        scores = self.score[field]
        score_fields = []
        for score_field in scores:
            score_fields.append(self.attribute_columns[field]["score"][score_field])

        if not score:
            if "abs" in scores:
                score = "abs"
            else:
                score = scores[0]

        score_col = self.attribute_columns[field]["score"][score]
        display_fields = keys+add_fields+[ground_col,mirror_col]+score_fields
        divergence_df = self.raw_map[self.raw_map["_merge"]=="both"][self.raw_map[score_col]<1][display_fields]
        return divergence_df