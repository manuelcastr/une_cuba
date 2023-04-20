import pathlib
from datetime import datetime, timedelta

import numpy as np
from matplotlib import pyplot as plt, patches


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def plot_daily_generation_and_demand(headers, dates, data, suptitle='', **kwargs):
    generation, demand, outage_estimate, real_outage = data.T
    date_step = kwargs.get('date_step', 1)
    n_days = len(dates)

    fig: plt.Figure = plt.figure(suptitle, (13, 6), dpi=175)
    axes: plt.Axes = fig.gca()
    plt.suptitle(suptitle)

    x = np.arange(n_days)
    axes.fill_between(x, np.zeros(n_days), generation,
                      color='tab:green', edgecolor='k', lw=.8, zorder=100, label=headers[0])
    axes.plot(x, demand,
              color='tab:red', lw='3', zorder=200, label=headers[1])
    axes.fill_between(x, np.zeros(n_days), outage_estimate,
                      color='tab:brown', edgecolor='k', lw=1.2, zorder=300, label=headers[2])
    axes.plot(x, real_outage,
              color='firebrick', lw='2', ls=(0, (5, 1)), zorder=400, label=headers[3])

    x_labels = np.array([''] * len(dates), dtype=object)
    x_labels[::date_step] = np.array([d.strftime("%d/%m") for d in dates], dtype=object)[::date_step]
    axes.set_xticks(range(n_days), x_labels, rotation=-70, fontsize='x-small')

    axes.set_xlabel('Fecha')
    axes.set_ylabel('MWh', rotation=0)
    axes.yaxis.set_label_coords(-0.015, 1)

    axes.grid(which='major', axis='y', ls=':', lw=.5, c='gray')
    axes.grid(which='minor', axis='y', ls=':', lw=.3, c='gray')
    axes.set_xlim(0, n_days)
    axes.set_ylim(0, max(generation.max(), demand.max()) * 1.2)

    axes.legend(loc='upper right')

    for k, spine in axes.spines.items():  # ax.spines is a dictionary
        spine.set_zorder(1000)
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)

    plt.tight_layout(pad=.5)

    if kwargs.get('show', False):
        plt.show()

    return fig


def report_une_stats(data_filename, output_path, show=False):
    with open(data_filename, encoding='utf-8') as f:
        lines = np.array([line.strip().split('\t') for line in f], dtype=object)
    headers = lines[0]
    dates = np.array([datetime.strptime(day, '%d-%m-%Y').date() for day in lines[1:, 0]])
    une_data = lines[1:, 1:].astype(int)

    output_path = pathlib.Path(output_path)

    fig_generation = plot_daily_generation_and_demand(
        headers[[1,2,4,5]],
        dates,
        une_data[:, [0, 1, 3, 4]],
        suptitle='Estimado de Generación vs Demanda en horario pico',
        date_step=3, show=show)

    fig_generation.savefig(str((output_path / 'fig_01_generation_vs_demand.png').absolute()))


if __name__ == '__main__':
    report_une_stats('une_cuba_diario_plus.txt',
                     'c:/Users/User/',
                     show=True)

