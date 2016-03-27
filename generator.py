#!/usr/bin/env python

import os
import argparse
from collections import OrderedDict


import ujson
import json

from jinja2 import Environment, FileSystemLoader

PATH = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_ENVIRONMENT = Environment(
    autoescape=False,
    loader=FileSystemLoader(os.path.join(PATH, 'template')),
    trim_blocks=True,
    lstrip_blocks=True)


def render_template(template_filename, context):
    return TEMPLATE_ENVIRONMENT.get_template(template_filename).render(context)


def create_network_py(network, out):
    context = {
        "meta": network["meta"],
        "data": network["data"],
        "network": network["network"],
        "loss": network["loss"],
    }
    #
    with open(out, 'w') as f:
        py = render_template('network.tpl', context)
        f.write(py)


########################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--network', '-n', type=str, default="",
                        help='The JSON definition of the network')
    parser.add_argument('--out', '-o', type=str, default="",
                        help='The generated network')
    args = parser.parse_args()
    # Load Json
    with open(args.network, 'r') as f:
        nn = json.load(f, object_pairs_hook=OrderedDict)

    # Generate Network for the lolz
    create_network_py(nn, args.out)
