#!/usr/bin/env python

import os
import argparse
from collections import OrderedDict


import json
import autopep8

from jinja2 import Environment, FileSystemLoader

PATH = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_ENVIRONMENT = Environment(
    autoescape=True,
    loader=FileSystemLoader(os.path.join(PATH, 'template')),
    trim_blocks=True,
    lstrip_blocks=True)


def render_template(template_filename, context):
    return TEMPLATE_ENVIRONMENT.get_template(template_filename).render(context)


def create_trainer_py(config, out):
    context = {
        "meta": config["meta"],
        "data": config["data"],
        "network": config["network"],
        "loss": config["loss"],
        "optimizer": config["optimizer"]
    }

    with open(out, 'w') as f:
        py = render_template('trainer.tpl', context)
        py = autopep8.fix_code(py, options={'aggressive': 1})
        f.write(py)


########################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--trainer', '-t', type=str, default="",
                        help='The JSON definition of the network')
    parser.add_argument('--out', '-o', type=str, default="",
                        help='The generated network')
    args = parser.parse_args()
    # Load Json
    with open(args.trainer, 'r') as f:
        config = json.load(f, object_pairs_hook=OrderedDict)

    # Generate Network for the lolz
    create_trainer_py(config, args.out)
