#!/usr/bin/env python

from __future__ import print_function
import argparse
import datetime
import json
import multiprocessing
import os
import random
import sys
import threading
import time

import numpy as np
from PIL import Image
import six
import six.moves.cPickle as pickle
from six.moves import queue

import chainer
from chainer import computational_graph
from chainer import cuda
import chainer.cuda.cupy as xp
from chainer import optimizers
from chainer import serializers

from cursesmenu import SelectionMenu

import pinterest.dataset as ds
import pinterest.containers as containers
import pinterest.features as features
import pinterest.utils as utils
from pinterest.evaluation import classifier

import network_generator

from {{meta["name"]}} import {{meta["name"]}}

class {{meta.name}}Trainer(object):
    def __init__(self, data, config, gpu_list, model=None, optimizer=None,
                 test=false):
        # General Config
        self.init = False
        self.config = config
        self.gpu_list = gpu_list
        self.model_state = model
        self.optimizer_state = optimizer

        # Data to process
        self.data = data
        self.data_q = queue.Queue(maxsize=1)
        self.res_q = queue.Queue()

        # Threading Configuration
        self.feeder = threading.Thread(target=feed_data)
        self.feeder.daemon = True
        self.logger = threading.Thread(target=log_result)
        self.logger.daemon = True

        assert 50000 % args.val_batchsize == 0
        if args.test:
            denominator = 1
        else:
            denominator = 100000

    def start(self):
        # Setup Models/Optimizer and Push to GPU
        self._setup_model()

        # Invoke threads
        self.feeder.start()
        self.logger.start()
        self.train_loop()
        self.feeder.join()
        self.logger.join()

    def _setup_model(self):
        print ("Initializing Model and GPUs")
        cuda.check_cuda_available()

        # Create Models
        for idx, gpu_id in gpu_list:
            if idx == 0:
                self.models = [{{meta["name"]}}()]  # Master Model
            else:
                self.models.append(self.models[0].copy())  # Shallow Model

            self.models[idx].to_gpu(gpu_id)

        # Setup optimizer
        optimizer = optimizers.{{optimizer["type"]}}(
        {%- for param, value in optimizer["params"].iteritems() -%}
            {{ param }}={{ value }},
        {%- endfor %})
        optimizer.setup(self.models[0])

        # Init/Resume
        if self.model_state:
            print('Load model from', self.model_state)
            serializers.HDF5VERSION TODO(self.model_state, model)
        if self.optimizer_state:
            print('Load optimizer state from', self.optimizer_state)
            serializers.HDF5VERSION TODO(self.optimizer_state, optimizer)

    def _setup_data(self):
        print ("Loading Data Containers")
        {% for name, desc in data.iteritems() %}
        {{ name }} = containers.LMDBContainerCached("{{desc["data_path"]}}")
        {{ name }}.make_key_map()
        {% endfor %}

    def train_loop():
        # Trainer
        graph_generated = False
        while True:
            while data_q.empty():
                time.sleep(0.1)
            inp = data_q.get()
            if inp == 'end':  # quit
                res_q.put('end')
                # Save final model
                serializers.save_npz(args.out, model)
                serializers.save_npz(args.outstate, optimizer)
                break
            elif inp == 'train':  # restart training
                res_q.put('train')
                model.train = True
                continue
            elif inp == 'val':  # start validation
                res_q.put('val')
                serializers.save_npz(args.out, model)
                serializers.save_npz(args.outstate, optimizer)
                model.train = False
                continue

            volatile = 'off' if model.train else 'on'
            x = chainer.Variable(xp.asarray(inp[0]), volatile=volatile)
            t = chainer.Variable(xp.asarray(inp[1]), volatile=volatile)

            if model.train:
                optimizer.update(model, x, t)
                if not graph_generated:
                    with open('graph.dot', 'w') as o:
                        o.write(computational_graph.build_computational_graph(
                            (model.loss,)).dump())
                    print('generated graph', file=sys.stderr)
                    graph_generated = True
            else:
                model(x, t)

            res_q.put((float(model.loss.data), float(model.accuracy.data)))
            del x, t

    def feed_data(self):
        # Data feeder
        i = 0
        count = 0

        x_batch = np.ndarray(
            (args.batchsize, 3, model.insize, model.insize), dtype=np.float32)
        y_batch = np.ndarray((args.batchsize,), dtype=np.int32)
        val_x_batch = np.ndarray(
            (args.val_batchsize, 3, model.insize, model.insize), dtype=np.float32)
        val_y_batch = np.ndarray((args.val_batchsize,), dtype=np.int32)

        batch_pool = [None] * args.batchsize
        val_batch_pool = [None] * args.val_batchsize
        pool = multiprocessing.Pool(args.loaderjob)
        data_q.put('train')
        for epoch in six.moves.range(1, 1 + args.epoch):
            print('epoch', epoch, file=sys.stderr)
            print('learning rate', optimizer.lr, file=sys.stderr)
            perm = np.random.permutation(len(train_list))
            for idx in perm:
                path, label = train_list[idx]
                batch_pool[i] = pool.apply_async(read_image, (path, False, True))
                y_batch[i] = label
                i += 1

                if i == args.batchsize:
                    for j, x in enumerate(batch_pool):
                        x_batch[j] = x.get()
                    data_q.put((x_batch.copy(), y_batch.copy()))
                    i = 0

                count += 1
                if count % denominator == 0:
                    data_q.put('val')
                    j = 0
                    for path, label in val_list:
                        val_batch_pool[j] = pool.apply_async(
                            read_image, (path, True, False))
                        val_y_batch[j] = label
                        j += 1

                        if j == args.val_batchsize:
                            for k, x in enumerate(val_batch_pool):
                                val_x_batch[k] = x.get()
                            data_q.put((val_x_batch.copy(), val_y_batch.copy()))
                            j = 0
                    data_q.put('train')

            optimizer.lr *= 0.97
        pool.close()
        pool.join()
        data_q.put('end')

    def log_result(self):
        # Logger
        train_count = 0
        train_cur_loss = 0
        train_cur_accuracy = 0
        begin_at = time.time()
        val_begin_at = None
        while True:
            result = res_q.get()
            if result == 'end':
                print(file=sys.stderr)
                break
            elif result == 'train':
                print(file=sys.stderr)
                train = True
                if val_begin_at is not None:
                    begin_at += time.time() - val_begin_at
                    val_begin_at = None
                continue
            elif result == 'val':
                print(file=sys.stderr)
                train = False
                val_count = val_loss = val_accuracy = 0
                val_begin_at = time.time()
                continue

            loss, accuracy = result
            if train:
                train_count += 1
                duration = time.time() - begin_at
                throughput = train_count * args.batchsize / duration
                sys.stderr.write(
                    '\rtrain {} updates ({} samples) time: {} ({} images/sec)'
                    .format(train_count, train_count * args.batchsize,
                            datetime.timedelta(seconds=duration), throughput))

                train_cur_loss += loss
                train_cur_accuracy += accuracy
                if train_count % 1000 == 0:
                    mean_loss = train_cur_loss / 1000
                    mean_error = 1 - train_cur_accuracy / 1000
                    print(file=sys.stderr)
                    print(json.dumps({'type': 'train', 'iteration': train_count,
                                      'error': mean_error, 'loss': mean_loss}))
                    sys.stdout.flush()
                    train_cur_loss = 0
                    train_cur_accuracy = 0
            else:
                val_count += args.val_batchsize
                duration = time.time() - val_begin_at
                throughput = val_count / duration
                sys.stderr.write(
                    '\rval   {} batches ({} samples) time: {} ({} images/sec)'
                    .format(val_count / args.val_batchsize, val_count,
                            datetime.timedelta(seconds=duration), throughput))

                val_loss += loss
                val_accuracy += accuracy
                if val_count == 50000:
                    mean_loss = val_loss * args.val_batchsize / 50000
                    mean_error = 1 - val_accuracy * args.val_batchsize / 50000
                    print(file=sys.stderr)
                    print(json.dumps({'type': 'val', 'iteration': train_count,
                                      'error': mean_error, 'loss': mean_loss}))
                    sys.stdout.flush()
