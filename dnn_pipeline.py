#!/usr/bin/env python
from __future__ import print_function
import os
import argparse
import json
import shutil
from datetime import datetime
import imp

import autopep8
from jinja2 import Environment, FileSystemLoader
from cursesmenu import SelectionMenu

import pinterest.dataset as ds
import pinterest.utils as utils
import pinterest.fs as fs

PATH = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_ENVIRONMENT = Environment(
    autoescape=True,
    loader=FileSystemLoader(os.path.join(PATH, 'template')),
    trim_blocks=True,
    lstrip_blocks=True)


def render_template(template_filename, context):
    return TEMPLATE_ENVIRONMENT.get_template(template_filename).render(context)


def create_trainer_py(config):
    context = {
        "meta": config["meta"],
        "data": config["data"],
        "network": config["network"],
        "loss": config["loss"],
        "optimizer": config["loss"]
    }

    with open(os.path.join(config["path"], "trainer.py"), 'w') as f:
        py = render_template('trainer.tpl', context)
        py = autopep8.fix_code(py, options={'aggressive': 1})
        f.write(py)


def create_network_py(config):
    context = {
        "path": config["path"],
        "meta": config["meta"],
        "data": config["data"],
        "network": config["network"],
        "loss": config["loss"],
    }

    with open(os.path.join(config["path"], config["name"] + ".py"), 'w') as f:
        py = render_template('network.tpl', context)
        py = autopep8.fix_code(py, options={'aggressive': 1})
        f.write(py)

########################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='',
                        help='Base folder of the crawled pinterest dataset',
                        required=True)
    args = parser.parse_args()

    #####################################################
    #####################################################
    print ("1. Loading Dataset")
    dataset = ds.PinterestDataset(args.dataset)
    dataset.init_from_file()

    cfg_folder = os.path.join(dataset.dnn_dir, "_configs")
    config_list = [f for f in os.listdir(cfg_folder)
                   if os.path.isfile(os.path.join(cfg_folder, f))]

    selection = \
        SelectionMenu.get_selection(config_list, title='DNN - Trainer',
                                    subtitle="Please select configuration file\
                                              from list", exit_option=True)
    if selection == len(config_list):
        print ("No choice made. Exit...")
        exit()

    # Some General Config
    with open(os.path.join(cfg_folder, config_list[selection]), 'r') \
            as outfile:
        config = json.load(outfile)

    print ("Selected Config File:", config_list[selection])

    for key, val in config["meta"].iteritems():
        print ('\t{0:<20} :: {1:<50}'.format(key, val))

    if not utils.yes_or_no("Proceed with these settings?"):
        exit()

    #####################################################
    #####################################################
    print ("2. Loaded Configuration for:", config["tag"])
    print ("Description", config["description"])
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    config["ts_start"] = timestamp
    config["ts_end"] = None
    config["path"] = os.path.join(dataset.dnn_dir, timestamp)
    fs.mkdir_p(os.path.join(config["path"], "_model")) # contains trained model and optimizer state
    fs.mkdir_p(os.path.join(config["path"], "_html")) # for up to date live stats visualized
    with open(os.path.join(config["path"], "config.json"), 'w') as outfile:
        json.dump(eval_run, outfile, indent=4)
    # Make copy of the whole source code
    shutil.copytree(os.path.dirname(os.path.realpath(__file__)),
                    os.path.join(config["path"], "_source"))

    #  TODO Setup logging file????

    #####################################################
    #####################################################

    print ("3. Generating Network from Config")
    create_network_py(config)

    print ("4. Generating Trainer from Config")
    create_trainer_py(config)

    print ("5. Starting Generated Trainer Routine - Please Stand By")
    dnn_trainer = imp.load_source('module.name',
                                  os.path.join(config["path"], "trainer.py"))
    trainer = dnn_trainer.Trainer()
    trainer.start()
