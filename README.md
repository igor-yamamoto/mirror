# Mirror

> *Mirror, mirror on the wall, who's the most accurate of them all?*

Mirror is a simple package written for the corparison of database samples. Its main purpose is to assert the assertivity between two samples by comparing each and individual value when a key value is matched. It also helps to find which values were divergent and what their keys were.

Mirror is built on top of `pandas.DataFrame` objects for calculating assertivity scores. 

## Usage

In Mirror, we assume that there is:

- A **ground thruth sample**, in which we are trying to mirror (`ground`)
- A **mirrored sample** (`mirror`), which inherits the same schema from `ground`

The `ground` and `mirror` dataframes are then compared by specifying which are the keys used to link the information. 

To create a `ground` object:
``` python
from mirror.ground import Ground
ground = Ground(ground=ground_df, keys=keys_name_array)
```

Then we can create a mirror instance:
``` python
from mirror.mirror import Mirror
mirror = Mirror(ground, mirror=mirror_df)
```

From that, a `raw_map` dataframe is created, which is a map of all the raw values from both the `ground` and `mirror` dataframes, and the scores computed by the method specified. To print an assertivity summary, the method `mirror.print_stats()` can be used.

Currently, there are only two scoring methods available:

- `abs`:  an absolute and binary comparison method (0 for divergence, 1 for assertion)
-  `sequence-mathcer`: an implementation of `difflib.SequenceMatcher` algorithm

The scoring methods for each attribute field are set by giving the `Mirror` class the `score` parameter, which is simply a `dict` object of arrays containing which method to use:

``` python
scoring = {
    "attribute_col_1": [
        "abs",
        "sequence-matcher"
    ],
    "attribute_col_2": [
        "abs"
    ],
    "attribute_col_3": [
        "sequence-matcher"
    ]
}

mirror = Mirror(ground, mirror=mirror_df, score=scoring)
```

## Installing

As of now, the package is not published on the PiPy repository. It can be mannually installed as a wheel package by:

``` sh
$ git clone https://github.com/igor-yamamoto/mirror.git
$ pip install ./mirror/dist/mirror-0.1.0-py3-none-any.whl 
```