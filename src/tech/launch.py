from collections.abc import Callable
from typing import List, Tuple, Union
import pandas as pd
import datetime
import numpy as np
from io import BytesIO


def launch_plot_functions(functions: List[Union[List[Callable], Callable]],
                          df: pd.DataFrame, metadata: pd.DataFrame, trainings: pd.DataFrame,
                          expenses: pd.DataFrame, income: pd.DataFrame, **kwargs
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
            tmp = []
            for subf in f:
                if kwargs is not None and (x := kwargs.get(func_name := subf.__name__, False)) and\
                        (func_name.startswith('expenses_') or func_name.startswith('income_')):
                    tmp.append(subf(expenses, income, **x))
                elif func_name.startswith('expenses_') or func_name.startswith('income_'):
                    tmp.append(subf(expenses, income))
                elif kwargs is not None and x:
                    tmp.append(subf(df, metadata, **x))
                else:
                    tmp.append(subf(df, metadata))
            ret.append(tmp)
        elif kwargs.get(func_name := f.__name__, None) is not None and (x := kwargs.get(func_name, False)) and\
                (func_name.startswith('expenses_') or func_name.startswith('income_')):
            ret.append(f(expenses, income, **x))
        elif func_name.startswith('expenses_') or func_name.startswith('income_'):
            ret.append(f(expenses, income))
        elif kwargs.get(func_name, None) is not None and x:
            ret.append(f(df, metadata, **x))
        else:
            ret.append(f(df, metadata))
    return ret
