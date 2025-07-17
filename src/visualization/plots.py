import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
from io import BytesIO
from typing import Tuple

# import matplotlib.colors as mcolors
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


plt.style.use('seaborn-v0_8-whitegrid')


def _convert_to_bytes(fig) -> BytesIO:
    # TODO: arbitrary dpi
    plot_file = BytesIO()
    fig = fig.get_figure()
    fig.savefig(plot_file, format='png', dpi=200)
    plot_file.seek(0)
    return plot_file


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` Axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding Axes.

    was taken from https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')
            # Убираем отступ, растягивая оси до края патча
            self.set_ylim(0, 1)  # Убедитесь, что верхний предел совпадает с радиусом патча (0.5)

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)  # Радиус 0.5 (половина размера Axes)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                spine = Spine(
                    axes=self,
                    spine_type='circle',
                    path=Path.unit_regular_polygon(num_vars)
                )
                spine.set_transform(
                    Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes
                )
                return {'polar': spine}

    register_projection(RadarAxes)
    return theta


def habbits_radar(df: pd.DataFrame, metadata: pd.DataFrame, **kwargs) -> Tuple[BytesIO, str]:
    """Makes radar plot and of frequency features and visually comparison them

    Args:
        df (pd.DataFrame): master dataframe
        metadata (pd.DataFrame): metadata of master dataframe. Columns 'feature' and 'metatype' are neccesary

        ** exclude (tuple): the tuple of excluding frequency features
        ** TODO: support of arbitrary messages

    Returns:
        Tuple[BytesIO, str]: BytesIO view of plot and message with plot
    """
    df = df.loc[
        :, ['date', 'period'] + metadata.loc[metadata['metatype'].isin(['frequency_tracking']), 'feature'].to_list()
    ]
    if len(exclude := kwargs.get('exclude', 0)) == 1:
        df = df.loc[:, ~df.columns.isin(exclude)]
    elif len(exclude) > 1:
        df = df.loc[:, ~df.columns.isin(list(exclude))]

    current, previous = df[df['period'] == 'Current week'], df[df['period'] == 'Previous week']

    # initializing
    current = current.iloc[:, 2:].sum().sort_index().reset_index()
    current.columns = ['index', 'value']
    previous = previous.iloc[:, 2:].sum().sort_index().reset_index()
    previous.columns = ['index', 'value']
    N = previous.shape[0]
    theta = radar_factory(N, frame='polygon')
    labels = current['index'].map(lambda x: ' '.join(x.split('_')).capitalize())

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection='radar'))
    ax.set_varlabels(labels)

    ax.plot(theta, current['value'], 'xkcd:magenta', alpha=0.8)
    ax.fill(theta, current['value'], facecolor='xkcd:magenta', alpha=0.35, label='_nolegend_')
    ax.plot(theta, previous['value'], 'xkcd:sky blue', alpha=0.8)
    ax.fill(theta, previous['value'], facecolor='xkcd:sky blue', alpha=0.35, label='_nolegend_')

    ax.set_rgrids(np.arange(8))

    # add legend relative to top-left plot
    ax.legend(('Current week', 'Previous week'), loc=(0.9, .95), labelspacing=0.1, fontsize='small')
    ax.set_title('Frequencies comparison for last 7 days and 7 days before them', fontweight='bold')

    # TODO:
    message = '_comming soon..._'
    # message = '_' + ' — '.join(
    #     df.loc[df['period'] == 'Previous week', 'date'].iloc[[0, -1]].map(lambda x: x.strftime('%d.%m.%Y'))
    # ) + ' : ' +\
    #     ' — '.join(
    #         df.loc[df['period'] == 'Current week', 'date'].iloc[[0, -1]].map(lambda x: x.strftime('%d.%m.%Y'))
    #     ) + ' period._' +\
    #         '\n\n_Comparison of Current week and Previous week frequency features_'

    return _convert_to_bytes(fig), message


def habbits_linear(df: pd.DataFrame, metadata: pd.DataFrame, **kwargs) -> Tuple[BytesIO, str]:
    """Makes bar plot for frequency features

    Args:
        df (pd.DataFrame): master dataframe
        metadata (pd.DataFrame): metadata of master dataframe. Columns 'feature' and 'metatype' are neccesary

        ** exclude (tuple): the tuple of excluding frequency features
        ** number_of_weeks (int): how last weeks to take. Defaults to 10
        ** base_width (float): base_width of each group of bars, width of each separate bar is calculated from that.
                                    Defaults to 1
        ** TODO: list of tracking parameters and theirs trend and trend analysis
        ** TODO: support of arbitrary messages

    Returns:
        Tuple[BytesIO, str]: BytesIO view of plot and message with plot

    TODO: more meaningfull message
    """
    df = df.loc[
        :, ['date', 'period'] + metadata.loc[metadata['metatype'].isin(['frequency_tracking']), 'feature'].to_list()
    ]
    if len(exclude := kwargs.get('exclude', 0)) == 1:
        df = df.loc[:, ~df.columns.isin(exclude)]
    elif len(exclude) > 1:
        df = df.loc[:, ~df.columns.isin(list(exclude))]

    # forming dataframe of sequentially following dates and left join main dataframe to it to get sequential weeks
    df = pd.DataFrame(
        {
            'date': pd.date_range(
                start=df['date'].min(), end=df['date'].max() + datetime.timedelta(6 - df['date'].max().weekday()),
                freq='D'
            ).date
        }
    ).merge(df, 'left', 'date')

    # create week start date for each date
    df['week_start'] = (df['date'] - (df['date'].map(lambda x: datetime.timedelta(x.weekday()))))

    # create groups by start of week and summing them, where all values for week are nulls then null is setted
    groups = df.iloc[:, 2:].groupby('week_start').apply(
        lambda x: x.sum() if ~x.isna().any().all() else None
    ).iloc[-kwargs.get('number_of_weeks', 10):]

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    n_bars = groups.shape[0]
    bar_width = kwargs.get('base_width', 0.7) / n_bars

    fig, ax = plt.subplots(figsize=(16, 12))
    bars = []
    for i, (_, values) in enumerate(groups.items()):
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2
        for x, y in enumerate(values.values):
            bar = ax.bar(x + x_offset, y, width=bar_width, color=colors[i % len(colors)])
        bars.append(bar[0])

    ax.legend(bars, [' '.join(c.split('_')).capitalize() for c in groups])
    ax.set_xticks(np.arange(n_bars), labels=groups.index)

    ax.set_xlabel('t', loc='right')
    ax.set_ylabel('frequency', loc='top', rotation=0)
    fig.tight_layout()

    ax.set_title(f'Frequencies for last {kwargs.get('number_of_weeks', 10)} weeks', fontweight='bold')

    # TODO:
    message = '_comming soon..._'

    return _convert_to_bytes(fig), message
