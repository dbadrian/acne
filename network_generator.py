#!/usr/bin/env python

import os
import argparse
from collections import OrderedDict

import json
import autopep8

from jinja2 import Environment, FileSystemLoader

PATH = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_ENVIRONMENT = Environment(
    autoescape=False,
    loader=FileSystemLoader(os.path.join(PATH, 'template')),
    trim_blocks=True,
    lstrip_blocks=True)


def render_template(template_filename, context):
    return TEMPLATE_ENVIRONMENT.get_template(template_filename).render(context)


def create_network_py(config):
    context = {
        "meta": config["meta"],
        "data": config["data"],
        "network": config["network"],
        "loss": config["loss"],
    }

    with open(os.path.join(config["path"], config["name"]), 'w') as f:
        py = render_template('network.tpl', context)
        py = autopep8.fix_code(py, options={'aggressive': 1})
        f.write(py)


########################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--network', '-n', type=str, default="",
                        help='The JSON definition of the network')
    args = parser.parse_args()
    # Load Json
    with open(args.network, 'r') as f:
        nn = json.load(f, object_pairs_hook=OrderedDict)

    # Generate Network for the lolz
    create_network_py(nn, args.out)
