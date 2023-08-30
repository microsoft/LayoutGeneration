import argparse
import datetime
import errno
import os
import re

import matplotlib.pyplot as plt
import tqdm
from file_utils import read_json

TYPE2COLOR = {
    'web': {
        'text': 'black',
        'link': 'silver',
        'button': 'red',
        'title': 'tan',
        'description': 'orange',
        'submit': 'pink',
        'image': 'green',
        'background': 'blue',
        'icon': 'violet',
        'logo': 'pink',
        'input': 'red'
    },
    'rico': {
        'icon': 'tan',
        'list item': 'orange',
        'text button': 'orange',
        'toolbar': 'green',
        'web view': 'blue',
        'input': 'violet',
        'card': 'c',
        'advertisement': 'goldenrod',
        'background image': 'teal',
        'drawer': 'skyblue',
        'radio button': 'pink',
        'checkbox': 'black',
        'multi-tab': 'silver',
        'pager indicator': 'red',
        'modal': 'tan',
        'on/off switch': 'orange',
        'slider': 'orange',
        'map view': 'green',
        'button bar': 'blue',
        'video': 'violet',
        'bottom navigation': 'c',
        'number stepper': 'goldenrod',
        'date picker': 'teal',
        'text': 'silver',
        'image': 'red',
    }
}


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def draw_layouts(args, RE_PATTERN):
    if args.is_draw_ground_true:
        dirname = os.path.join(args.output_base, args.dirname, 'real')
    else:
        dirname = os.path.join(args.output_base, args.dirname, 'pred')
    mkdir_if_missing(dirname)
    predictions = read_json(args.prediction_filename)
    for prediction in tqdm.tqdm(predictions):
        line_id = prediction['line_id']
        layouts = [prediction['gold_layout_seq']] if args.is_draw_ground_true \
                    else prediction['prediction']
        for i, layout in enumerate(layouts):
            elements = re.findall(RE_PATTERN, layout)
            show_sample(elements, line_id, i, dirname, args.dataset)
        # exit()


def show_sample(elements, line_id, image_idx, dirname, dataset):
    fig = plt.figure()
    ax1 = fig.add_subplot(111, aspect='equal')

    for i in range(len(elements)):
        label = elements[i][0]
        color = TYPE2COLOR[dataset][label]
        x = int(elements[i][1])
        y = int(elements[i][2])
        w = int(elements[i][3])
        h = int(elements[i][4])
        ax1.add_patch(
            plt.Rectangle(
                (x / 100, y / 100),
                w / 100, h / 100,
                facecolor=color, alpha=0.2
            )
        )
        if dataset == 'rico':
            ax1.add_patch(
                plt.Rectangle(
                    (x / 100, y / 100),
                    w / 100, h / 100,
                    linewidth=1.5,
                    edgecolor=color,
                    fill=False
                )
            )
            ax1.text(
                x / 100 + 0.03,
                y / 100 + 0.15,
                label,
                fontsize=6,
                color=color
            )
        else:
            ax1.add_patch(
                plt.Rectangle(
                    (x / 100, y / 100),
                    w / 100, h / 100,
                    linewidth=0.7,
                    edgecolor=color,
                    fill=False
                )
            )
            ax1.text(
                x / 100 + 0.025,
                y / 100 + 0.035,
                label,
                fontsize=4,
                color=color
            )

    if dataset == 'web':
        ax1.set_xlim(0, 1.200)
        ax1.set_ylim(2.133, 0)
    else:
        ax1.set_xlim(0, 1.440)
        ax1.set_ylim(2.560, 0)
    ax1.set_xticks([])
    ax1.set_yticks([])
    if dataset == 'web':
        plt.axis('off')
    plt.show()
    plt.savefig(
        os.path.join(dirname, f'{line_id}_{image_idx}'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['web', 'rico'])
    parser.add_argument('--prediction_filename', type=str)
    parser.add_argument('--is_draw_ground_true', action='store_true')
    parser.add_argument('--output_base', type=str, default=os.path.join(os.getcwd(), '../visualization_results'))
    parser.add_argument('--dirname', type=str)
    args = parser.parse_args()
    if args.dirname is None:
        args.dirname = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    labels = TYPE2COLOR[args.dataset].keys()
    RE_PATTERN = f'({"|".join(list(labels))}) (\d+) (\d+) (\d+) (\d+)'
    draw_layouts(args, RE_PATTERN)
