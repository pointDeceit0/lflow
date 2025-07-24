import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import datetime
from io import BytesIO
from typing import Tuple

from scipy.stats import ttest_ind

# import matplotlib.colors as mcolors
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from matplotlib.patches import Patch


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

    ax.legend(bars, [' '.join(c.split('_')).capitalize() for c in groups], loc='upper right')
    ax.set_xticks(np.arange(n_bars), labels=groups.index)

    ax.set_xlabel('t', loc='right')
    ax.set_ylabel('frequency', loc='top', rotation=0)
    fig.tight_layout()

    ax.set_title(f'Frequencies for last {kwargs.get('number_of_weeks', 10)} weeks', fontweight='bold')

    # TODO:
    message = '_comming soon..._'

    return _convert_to_bytes(fig), message


def CPFC_violins(df: pd.DataFrame, metadata: pd.DataFrame, **kwargs) -> Tuple[BytesIO, str]:
    """Makes 4 violin plots for callories, proteins, fats, carbohydrates compare theirs consumption by periods

    Args:
        df (pd.DataFrame): master dataframe
        metadata (pd.DataFrame): metadata of master dataframe. Columns 'feature' and 'metatype' are neccesary
        TODO: **kwargs

    Returns:
        Tuple[BytesIO, str]: Tuple[BytesIO, str]: BytesIO view of plot and message with plot

    TODO TODO TODO: observation of such a specific few features isn't seemed like a general approach.
                    However, they are considered together in most cases.
                    But more general approach should be considered.

    TODO TODO TODO: also it isn't seemed good that there's no option of choosing a period for comparison.
                    The approach where specific dates periods are considered seemed more attractive.
                    And it is not so complicated in realisation: it could be done over existing realisation replacing
                    existing values of period column with new values recieved from input. But the problem of getting
                    starts and ends of period are opened due to the fact that we cannot send it directly.
                    Maybe another approaches of launching functions with variables should be considered.
                    Realisation with kwargs doesn't seemed to me elegant, but again, it could be implemented fast.
    """
    df = df.loc[
        :, ['date', 'period'] + metadata.loc[metadata['metatype'].isin(['nutrition_tracking']), 'feature'].to_list()
    ]
    fig, ax = plt.subplots(1, df.shape[1] - 2, figsize=(20, 10))
    # TODO: add to kwargs
    fig.suptitle('CPFC periods densities', fontsize=16)

    # TODO: add to kwargs bellow variables
    RED_FLAG = '🟥'
    GREEN_FLAG = '🟩'
    YELLOW_FLAG = '🟨'
    prev_per_col, prev_week_col, curr_week_col = "#4d88b3", "#1a5a77", "#f7e64c"
    gap = 0.03
    alpha = 0.5

    message = 'AVG magnitudes\\*:\n'
    for k, feature in enumerate(df.columns[2:]):
        current_data = df[df['period'] == 'Current week'].copy()
        previous_data = df[df['period'] != 'Current week'].copy()
        sns.violinplot(
            data=previous_data, y=feature, ax=ax[k],
            hue="period", split=True, gap=gap, cut=0,
            inner='quart',
            palette='mako', legend=None, alpha=alpha, linewidth=1.2,
            color=[prev_per_col, prev_week_col]
        )

        violin_parts = ax[k].violinplot([current_data[feature]],
                                        positions=[gap],
                                        vert=True,
                                        showmeans=False, showmedians=False, showextrema=False,
                                        widths=0.5)
        # TODO: the problem of displaying quartiles of current week period isn't solved due to lack of
        # TODO: due to lack of ideas how to do that. It should be done
        for pc in violin_parts['bodies']:
            pc.set_facecolor(curr_week_col)
            pc.set_edgecolor('black')
            pc.set_alpha(alpha)

            # Get vertices and modify to keep only right side
            # + gap / 4 provides aligning with other half of main violin
            vertices = pc.get_paths()[0].vertices
            vertices[:, 0] = np.maximum(vertices[:, 0], 0) + gap / 4

        legend_patches = [
            Patch(facecolor=prev_per_col, alpha=0.5, label='Previous period'),
            Patch(facecolor=prev_week_col, alpha=0.5, label='Previous week'),
            Patch(facecolor=curr_week_col, alpha=0.5, label='Current Week')
        ]

        # Add legend only to the last plot
        if k == 3:
            ax[k].legend(handles=legend_patches)

        # message with sinificant changes
        # NOTE: not sure that it is the most correct way to compare significance with all previous period
        # 		maybe it would be better to compare with only previous week or specific period
        tmp = ttest_ind(current_data[feature].mean(), previous_data[feature])[1]
        change = round(
            current_data[feature].mean() - previous_data[feature].mean(), 1
        )
        change = '+' + str(change) if change > 0 else change
        message += (f'    {RED_FLAG if tmp <= .01 else YELLOW_FLAG if tmp <= .05 else GREEN_FLAG} '
                    f'{feature}={round(current_data[feature].mean(), 1)} '
                    f'({change})\n')
    fig.tight_layout()

    message += ('\n🟥 — _changes are significant on 1% level of confidence_\n🟨 — '
                '_changes are significant on 5% level of confidence_\n🟩 — _changes are not significant_\n'
                '\n\\* _changes are shown relative to the mean of previous weeks._')

    return _convert_to_bytes(fig), message


def CPFC_linear(df: pd.DataFrame, metadata: pd.DataFrame, **kwargs) -> Tuple[BytesIO, str]:
    """Makes 3 linear plot (due to different scale) of callories | carbohydrates | proteins and fats.
       Also general condition line is reflected regarding to max value of feature

    Args:
        df (pd.DataFrame): master dataframe
        metadata (pd.DataFrame): metadata of master dataframe. Columns 'feature' and 'metatype' are neccesary

        ** grannulation(Any['day', 'week', 'month']): grannulation of grouping. Defaults to "week".
        ** displayed_times(int): how many grouped values to show. Defaults to 10.
        ** show_gen_cond(bool): True if to show general condition. Defaults to True.

    Returns:
        Tuple[BytesIO, str]: Tuple[BytesIO, str]: BytesIO view of plot and message with plot
    """
    # kwargs argument
    if (grannulation := kwargs.get('grannulation', 'week')) not in ('day', 'week', 'month'):
        return None

    # kwargs argument
    show_gen_cond = kwargs.get('show_gen_cond', False)
    df = df[
        ['date', 'period'] + (['general_condition'] if show_gen_cond else []) +
        metadata.loc[metadata['metatype'] == 'nutrition_tracking', 'feature'].to_list()
    ]

    # filling dates consequently
    df = pd.DataFrame(
        {
            'date': pd.date_range(
                start=df['date'].min(), end=df['date'].max() + datetime.timedelta(6 - df['date'].max().weekday()),
                freq='D'
            ).date
        }
    ).merge(df, 'left', 'date')

    # defining start of week, month. If day leave the same
    match grannulation:
        case 'month':
            df['grannulation'] = df['date'].map(lambda x: x.replace(day=1))
        case 'week':
            df['grannulation'] = (df['date'] - (df['date'].map(lambda x: datetime.timedelta(x.weekday()))))
        case 'day':
            df['grannulation'] = df['date']

    groups = df.iloc[:, 2:].groupby('grannulation').apply(
            lambda x: x.mean() if ~x.isna().any().all() else None
    )

    # defining x axis. If month -> days from date are deleted
    #                     weeks -> years from date are deleted, showed starts of weeks
    #                     days  -> only days are showed, according monthes and years reflected in the title
    match grannulation:
        case 'month':
            x = groups.index.map(lambda x: f'{str(x).split('-')[0]}-{str(x).split('-')[1]}')
        case 'week':
            x = groups.index.map(lambda x: f'{str(x).split('-')[1]}-{str(x).split('-')[2]}')
        case 'day':
            x = groups.index.map(lambda x: str(x).split('-')[2])

    groups = groups.iloc[-kwargs.get('displayed_times', 10):]
    x = x.values[-kwargs.get('displayed_times', 10):]

    # TODO: add to kwargs bellow variables
    color_protein, color_fats = "#6A9700", "#4d88b3"
    color_carbs = "#1a5a77"
    color_callories = "#f7e64c"
    color_gen_cond = "#e64040"

    fig, ax = plt.subplots(1, 3, figsize=(20, 10))
    xna = x[~groups['protein'].isna()]

    ax[2].plot(x, groups['protein'], color=color_protein)
    ax[2].plot(x, groups['fats'], color=color_fats)
    ax[1].plot(x, groups['carbohydrates'], color=color_carbs)
    ax[0].plot(x, groups['callories'], color=color_callories)

    if show_gen_cond:  # adding general condition line
        gen_cond_share = groups['general_condition'] / 100
        # second line is a upper bound to show relative share of general condition, lower bound is zero
        ax[2].plot(x, gen_cond_share * groups[['protein', 'fats']].max().max(), linestyle='--', color=color_gen_cond)
        ax[2].plot(xna, np.ones(xna.shape[0]) * groups[['protein', 'fats']].max().max(), linestyle='--',
                   color=color_gen_cond, alpha=0.3)
        ax[1].plot(x, gen_cond_share * groups['carbohydrates'].max(), linestyle='--', color=color_gen_cond)
        ax[1].plot(xna, np.ones(xna.shape[0]) * groups['carbohydrates'].max(), linestyle='--',
                   color=color_gen_cond, alpha=0.3)
        ax[0].plot(x, gen_cond_share * groups['callories'].max(), linestyle='--', color=color_gen_cond)
        ax[0].plot(xna, np.ones(xna.shape[0]) * groups['callories'].max(), linestyle='--',
                   color=color_gen_cond, alpha=0.3)

    ax[2].legend(['protein', 'fats'] + ['general condition share'] if show_gen_cond else [], loc='upper right')
    ax[2].set_xlabel(kwargs.get('grannulation', None), loc='right')
    ax[2].set_ylabel('g', loc='top', rotation=0)

    ax[1].legend(['carbohydrates'] + ['general condition share'] if show_gen_cond else [], loc='upper right')
    ax[1].set_xlabel(kwargs.get('grannulation', None), loc='right')
    ax[1].set_ylabel('g', loc='top', rotation=0)

    ax[0].legend(['callories'] + ['general condition share'] if show_gen_cond else [], loc='upper right')
    ax[0].set_xlabel(kwargs.get('grannulation', None), loc='right')
    ax[0].set_ylabel('kcall', loc='top', rotation=0)

    # title depends on grannutlation, monthes and years assigning
    match grannulation:
        case 'month':
            title = ', '.join(np.unique(groups.index.map(lambda x: str(x.year)))) +\
                    (' years' if len(np.unique(groups.index.map(lambda x: str(x.year)))) > 1 else ' year')
        case 'week':
            title = ', '.join(np.unique(groups.index.map(lambda x: str(x.year)))) +\
                    (' years' if len(np.unique(groups.index.map(lambda x: str(x.year)))) > 1 else ' year')
        case 'day':
            title = ', '.join(np.unique(groups.index.map(lambda x: str(x.year)))) +\
                    (' years' if len(np.unique(groups.index.map(lambda x: str(x.year)))) > 1 else ' year') + ' for ' +\
                    ', '.join(np.unique(groups.index.map(lambda x: str(x.month)))) +\
                    (' monthes' if len(np.unique(groups.index.map(lambda x: str(x.month)))) > 1 else ' month')

    fig.suptitle('CPFC for ' + title, fontsize=16)
    fig.tight_layout()

    # TODO:
    message = '_comming soon..._'

    return _convert_to_bytes(fig), message
