import pathlib
from datetime import datetime, timedelta

import numpy as np
from matplotlib import pyplot as plt, patches


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def plot_daily_generation_and_demand(headers, dates, data, suptitle=''):
    generation, demand = data.T

    fig: plt.Figure = plt.figure(suptitle, (13, 6), dpi=175)
    axes: plt.Axes = fig.gca()
    plt.suptitle(suptitle)

    n_days = (dates[-1] - dates[0]).days
    available_dates = set(dates)

    # over_lines = list()
    for i in range(n_days):
        if dates[0] + timedelta(i) in available_dates:
            axes.bar(i, generation[i], 1.0, color='tab:green', edgecolor='k', lw=.6, zorder=100)
            axes.bar(i, demand[i], 1.0, color='tab:red', edgecolor='k', lw=.6, zorder=200)


    #         over_lines.append([[i*2, generation[i]], [i*2 + 1, generation[i]]])
    #     else:
    #         a = over_lines[-1][1]
    #         if dates[0] + timedelta(i+1) in available_dates:
    #             b = [i*2 + 1, generation[i+1]]
    #         else:
    #             b = [i*2 + 1, a[1]]
    #         over_lines.append([a, b])
    # over_lines = np.array(over_lines)
    # for line in over_lines:
    #     plt.plot(*line.T, c='k', ls=':', lw=.6, zorder=100)

    x_labels = np.array([d.strftime("%d/%m") for d in dates], dtype=object)
    x_labels[1::2] = ''
    axes.set_xticks(range(n_days + 1), x_labels, rotation=-60)

    axes.grid(which='major', axis='y', ls=':', lw=.5, c='gray')
    axes.grid(which='minor', axis='y', ls=':', lw=.3, c='gray')
    axes.set_xlim(-0.75, n_days - .25)
    axes.set_ylim(0, max(generation.max(), demand.max()) * 1.2)
    handles, labels = axes.get_legend_handles_labels()
    patch_gen = patches.Patch(color='tab:green', edgecolor='k', label=headers[0])
    patch_dem = patches.Patch(color='tab:red', edgecolor='k', label=headers[1])
    handles.extend([patch_gen, patch_dem])
    axes.legend(handles=handles, loc='upper right')
    plt.tight_layout(pad=.5)
    plt.show()

    #
    #
    # for i in range(len(dates) - 1):
    #     if une_data[i].max() == 0 or une_data[i + 1].max() == 0:
    #         continue
    #     axes.plot([i, i + 1], une_data[[i, i + 1], 0], c='k', lw=.6, zorder=100)
    #     axes.fill_between([i, i + 1], une_data[[i, i + 1], 0], color='tab:green', zorder=90)
    #
    #     axes.plot([i, i + 1], une_data[[i, i + 1], 1], c='tab:red', ls='--', lw=2, zorder=200)
    #
    # for i in range(len(dates)):
    #     a0 = (i == 0) or une_data[i - 1].max() == 0
    #     b0 = (i == len(dates) - 1) or une_data[i + 1].max() == 0
    #     if a0 and b0 and une_data[i].max() > 0:
    #         axes.plot([max(.0, i - 0.5), min(len(dates) - 1.0, i + 0.5)], [une_data[i, 0], une_data[i, 0]],
    #                   c='k', lw=.6, zorder=100)
    #         axes.fill_between([max(.0, i - 0.5), min(len(dates) - 1.0, i + 0.5)], [une_data[i, 0], une_data[i, 0]],
    #                           color='tab:green', zorder=90)
    #         axes.plot([max(.0, i - 0.5), min(len(dates) - 1.0, i + 0.5)], [une_data[i, 0], une_data[i, 0]],
    #                   c='tab:red', lw=2, ls='--', zorder=200)
    #
    # x_labels = np.array([d.strftime("%d/%m") for d in dates], dtype=object)
    # step = 2
    # x_labels[1::step] = ''
    # axes.set_xticks(range(len(dates)), x_labels, rotation=45)

    # from matplotlib.lines import Line2D
    # import matplotlib.patches as mpatches

    # handles, labels = axes.get_legend_handles_labels()
    #
    # patch = patches.Patch(color='tab:green', label='Disponibilidad')
    # line = Line2D([0], [0], label='Demanda', color='tab:red', ls='--')
    #
    # handles.extend([patch, line])
    #
    # axes.legend(handles=handles, loc='upper right')

    # x_labels = [d.strftime("%d/%m") for d in dates]
    # for i in range(n_days - 1):
    #     if i % span:
    #         x_labels[n_days-1 - i] = ''
    # if first_date != 0:
    #     x_labels[0] = ''
    #     x_labels[first_date] = dates[first_date].strftime("%d/%m")
    #     x_labels[first_date+1] = ''
    #     x_labels[first_date+2] = ''
    # use_labels = np.array(x_labels)
    # no_labels = [1, 2]
    # if first_date in no_labels:
    #     no_labels.remove(first_date)
    # use_labels[no_labels] = ''
    # tick_positions = np.where(x_labels)[0]
    # axes.set_xticks(tick_positions, use_labels[tick_positions], rotation=45)
    # # axes.set_xticks(range(n_days), minor=True)
    # y_minor = AutoMinorLocator(5)
    # axes.yaxis.set_minor_locator(y_minor)
    # if format_func:
    #     axes.get_yaxis().set_major_formatter(FuncFormatter(format_func))
    #
    # axes.set_xlim(left=0, right=len(dates))
    # axes.set_ylim(bottom=0, top=une_data.max() * 1.15)
    #
    # plt.xlabel('Fecha')
    # plt.ylabel('MWh', rotation=0)
    # axes.yaxis.set_label_coords(-0.015, 1)
    #
    # axes.spines['right'].set_visible(False)
    # axes.spines['top'].set_visible(False)
    #
    # axes.xaxis.set_tick_params(which='minor', width=.6)
    # axes.xaxis.set_tick_params(which='major', pad=-2, labelsize=8)
    #

    # axes.legend(loc='best').set_zorder(500)

    # fig.tight_layout()

    # n_days = len(dates)
    # xs = range(n_days)
    # fig.gca().fill_between(xs, une_data[:, 0], label=headers[0],
    #                        color='tab:green', zorder=10, edgecolor='k', lw=.6)
    # fig.gca().fill_between(xs, numbers[:, 2], label=labels[1],
    #                        color='tab:green', zorder=11, edgecolor='k', lw=.3)
    # fig.gca().fill_between(xs, numbers[:, 3], label=labels[2],
    #                        color='tab:orange', zorder=12, edgecolor='k', lw=.3)
    # fig.gca().fill_between(xs, numbers[:, 4], label=labels[3],
    #                        color=(.88, .39, .0), zorder=11, edgecolor='k', lw=.6)  # color=(.88, .39, .0)
    # fig.gca().fill_between(xs, numbers[:, 5], label=labels[4],
    #                        color='tab:red', zorder=13, edgecolor='k', lw=.6)
    #
    # common_style_settings(fig, dates, n_days, numbers[:, 1].max(), 'Millones',
    #                       format_func=lambda x, p: f'{x*1e-6:1.1f} M' if x > 0 else '',
    #                       span=14)

    # plt.gca().spines['bottom'].set_zorder(100)
    # plt.gca().spines['left'].set_zorder(100)

    # plt.show()


    return fig


def report_une_stats(data_filename, output_path):
    with open(data_filename) as f:
        lines = np.array([line.strip().split('\t') for line in f], dtype=object)
    headers = lines[0]
    dates = np.array([datetime.strptime(day, '%d-%m-%Y').date() for day in lines[1:, 0]])
    une_data = lines[1:, 1:].astype(int)

    output_path = pathlib.Path(output_path)

    fig_generation = plot_daily_generation_and_demand(headers[1:3], dates, une_data[:, [0, 1]], 'Generación vs Demanda')



if __name__ == '__main__':
    report_une_stats('une_cuba_diario.txt',
                     'c:/Users/User/')

