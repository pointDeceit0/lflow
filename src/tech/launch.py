from collections.abc import Callable
from typing import List, Tuple, Union
import pandas as pd
import datetime
import numpy as np
from io import BytesIO


def launch_plot_functions(functions: List[Union[List[Callable], Callable]],
                          df: pd.DataFrame, metadata: pd.DataFrame, trainings: pd.DataFrame, **kwargs
                          ) -> List[Tuple[BytesIO, str]]:
    """Sequentially launches all plots making functions

    Args:
        functions (List[Callable]): function that is need to be called
        df (pd.DataFrame): master dataframe with all data
        metadata (pd.DataFrame): metadata of master dataframe that neccesary must contain metatype and feature
        trainings (pd.DataFrame): trainings dataframe, which must containt date column
        **kwargs: additional parameters to functions

    Returns:
        List[Tuple[BytesIO, str]]: list of tuples where each one contains BytesIO view of plot firstly, and
                str message to this plot secondly
    """
    # assignment of current week, previous week and previous period. Current week is counted from now and 7 days before,
    # previous week 7 days before current accordingly. Is needed for many plots.
    df['period'] = np.where(
        df['date'] > (datetime.datetime.now() - datetime.timedelta(days=8)).date(),
        'Current week',
        np.where(
            (df['date'] > (datetime.datetime.now() - datetime.timedelta(days=15)).date()) &
            (df['date'] <= (datetime.datetime.now() - datetime.timedelta(days=8)).date()),
            'Previous week',
            'Previous period'
        )
    )

    ret = []
    for f in functions:
        # case when multiple plot must be union in one list
        if isinstance(f, list):
            ret.append([
                subf(df, metadata, **x)
                if kwargs is not None and (x := kwargs.get(subf.__name__, False)) else subf(df, metadata)
                for subf in f
            ])
        elif kwargs.get(f.__name__, None) is not None and (x := kwargs.get(f.__name__, False)):
            ret.append(f(df, metadata, **x))
        else:
            ret.append(f(df, metadata))
    return ret
