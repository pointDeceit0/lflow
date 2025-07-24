import pandas as pd
from apiclient.discovery import Resource
import datetime

from typing import List


# mapping for frequencies featuries
d = {'Y': 1, 'N': 0}


def data_metadata_consistensy_check(df: pd.DataFrame, metadata: pd.DataFrame) -> None | str:
    """Function checks consistency of metadata to master dataframe.

    NOTE: NEEDS TO BE IMPROVED

    Args:
        df (pd.DataFrame): master dataframe
        metadata (pd.DataFrame): metadata dataframe

    Returns:
        None | str: None if all is good, otherwise error
    """
    if len(metadata['feature'].sort_values()) != len(df.columns.sort_values()):
        raise Exception('Error! Different number of columns in "master table" and "metadata"!')
    if not (metadata['feature'].sort_values() == df.columns.sort_values()).all():
        raise Exception('Error! Different attributive composition of "master table" and "metadata"!')


def get_data(resource: Resource, sheet_name: str, SPREADSHEET_ID: str) -> pd.DataFrame:
    """Gets data from google API datatype Resource by sheet names and spreadsheet ids

    Args:
        resource (Resource): resource from above description itself
        sheet_name (str): google sheets sheet name
        SPREADSHEET_ID (str): standard spreadsheet id from google API

    Returns:
        pd.DataFrame: recieved table in pandas dataframe
    """
    return pd.DataFrame(
        resource.get(
            spreadsheetId=SPREADSHEET_ID,
            range=sheet_name,
            majorDimension='ROWS'
        ).execute().get('values', [])
    )


def init_transform_condition_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """Inition transformation of condition sheet, for following transformation transformed metadata sheet is needed.
        Get data from condition sheet and transform it, merge head, fill empties with nulls, converts date.

    Args:
        df (pd.DataFrame): dataframe with data

    Returns:
        pd.DataFrame: transformed dataframe
    """

    df.columns = df.iloc[df.index[df.iloc[:, 0].str.strip() == 'date'][0]].str.replace(' ', '_').values
    df = df.iloc[df.index[df.iloc[:, 0].str.strip() == 'date'][0] + 1:]

    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y').dt.date
    df = df[df['date'] < datetime.datetime.now().date()]
    df = df.replace('-', None).replace('', None)
    df = df.reset_index(drop=True)

    # TODO: add time convertation

    return df


def init_transform_metadata_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """Get data from manual condition sheet and transform it, merge head, fill empties with nulls, converts date.

    Args:
        df (pd.DataFrame): dataframe with data

    Returns:
        pd.DataFrame: getted, transformed dataframe
    """

    cols = df.iloc[1, :4].values
    df = df.iloc[2:, :4]
    df.columns = cols

    df = df.replace('-', None).replace('', None)
    df = df.reset_index(drop=True)
    df['feature'] = df['feature'].str.replace(' ', '_')

    return df


def init_transform_train_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """Get data from trainings sheet and transform it, merge head, fill empties with nulls, converts date.

    Args:
        df (pd.DataFrame): dataframe with data

    Returns:
        pd.DataFrame: getted, transformed dataframe
    """

    if df.shape[1] == 1:
        df = df.iloc[df.index[df.iloc[:, 0].str.strip() == 'date'][0] + 1:]
        df.columns = ['date']
        return df

    # df = df[df['date'] < datetime.datetime.now().date()]
    df = df.replace('-', None).replace('', None)
    df = df.reset_index(drop=True)

    df.iloc[0] = df.iloc[0].ffill()
    df.iloc[1] = df.iloc[1].ffill()
    df.iloc[:3] = df.iloc[:3].fillna('').map(lambda x: x.replace(' ', '_'))

    df.columns = ['->'.join([s for s in df[v].iloc[:3].values if s != '']) for v in df]
    df.columns = df.columns.str.replace('->_->', '->')
    df = df.iloc[3:]
    df = df.reset_index(drop=True)

    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y').dt.date

    return df


def fill_condition_sheet(df: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Making empties filling according to metatype from metadata sheet

    Args:
        df (pd.DataFrame): master dataframe
        metadata (pd.DataFrame): metadata dataframe

    Returns:
        pd.DataFrame: filled master dataframe

    TODO: filling using only previous values not for the all period
    """
    for c in df.columns:
        match metadata.loc[metadata['feature'] == c, 'metatype'].iloc[0]:
            case 'key_feature':
                df[c] = pd.to_numeric(df[c])
                df[c] = df[c].fillna(df[c].mean())
            case 'feature':
                df[c] = pd.to_numeric(df[c])
                df[c] = df[c].fillna(df[c].mean())
            case 'frequency_tracking':
                mode = df[~pd.isna(df[c])][c].mode()
                df[c] = df[c].fillna(mode.values[0])
                df[c] = df[c].map(d)
                df[c] = pd.to_numeric(df[c])
            case 'important_meetings':
                mode = df[~pd.isna(df[c])][c].mode()
                df[c] = df[c].fillna(mode.values[0])
                df[c] = df[c].map(d)
                df[c] = pd.to_numeric(df[c])
            case 'nutrition_tracking':
                df[c] = pd.to_numeric(df[c])
                df[c] = df[c].fillna(df[c].mean())
    return df


def transform_enrichment_data(resource: Resource, SPREADSHEET_ID: str) -> List[pd.DataFrame]:
    """Gets all three sheets, makes: initial transforming, filling misses, merging trains df to master

    Args:
        resource (Resource): google API datatype Resource. Could be recieved from src/tech/connection/extract_data.py
        SPREADSHEET_ID (str): standard spreadsheet id from google API

    Returns:
        pd.DataFrame: master df, metadata df, trains df
    """
    condition = init_transform_condition_sheet(get_data(resource, 'condition', SPREADSHEET_ID))
    metadata = init_transform_metadata_sheet(get_data(resource, 'manual condition', SPREADSHEET_ID))
    trains = init_transform_train_sheet(get_data(resource, 'trainings', SPREADSHEET_ID))
    data_metadata_consistensy_check(condition, metadata)

    df = fill_condition_sheet(condition, metadata)

    trains['train'] = 1
    df = df.merge(trains[['date', 'train']], 'left', 'date')
    df.loc[:, 'train'] = df.loc[:, 'train'].fillna(0)

    metadata = pd.concat([
        metadata,
        pd.DataFrame({
            c: [v]
            for v, c in zip(
                ['train', 'binary', None, 'frequency_tracking'],
                ['feature', 'measure unit', 'measure method', 'metatype']
            )
        })
    ])

    return df, metadata, trains
