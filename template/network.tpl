#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import OrderedDict

import chainer
import chainer.links as L
import chainer.functions as F

# import acne.methods

class {{ meta.name }}(chainer.Chain):

    def __init__(self):
        super({{ meta.name }}, self).__init__()

        # We add links outside the initializer
        # This allows more complex manipulations in __call__
        # and forward
        self.chain = OrderedDict()
        {% for layer_name, layer_def in network["__layers__"].iteritems() %}
        chain['{{ layer_name }}'] = {% if layer_def["type"].startswith('L')  %} {{ layer_def["type"] }}({% for param, value in layer_def["params"].iteritems() %}{{ param }}={{ value }},{% endfor %}){% else %}lambda self, {% for input in layer_def["input"].keys() %}{{ input }}{% endfor %}: {{ layer_def["type"] }}({% for input, value in layer_def["input"].iteritems() %}{{ input }}={{ value }},{% endfor %}{% for param, value in layer_def["params"].iteritems() %}{% if param in network["__states__"] %}{{ param }}=self.{{ value }},{% else %}{{ param }}={{ value }},{% endif %}{% endfor %}){% endif %}

        {% endfor %}

        self.variables = {}
        {% for layer_name, layer_def in network["__layers__"].iteritems() %}{% if layer_def["is_variable"] %}
        self.variables["{{ layer_name }}"] = None
        {% endif %}{% endfor %}

        # Generate Additional State Variables
        {% for state, value in network['__states__'].iteritems() %}
        self.{{ state }} = value
        {% endfor %}

        # Create Dummy Entries for Later
        {% for loss_name in loss.keys() %}
        self.variables["{{ loss_name }}"] = None
        {% endfor %}

        # Register Links with the Chainer library
        for link_name, link in self.chain.iteritems():
            if llink_name.startswith('L')::
                self.add_link(link)

    def __call__(self, {% for var in data %}{{ var }}, {% endfor %}):
        {% for output in network["__output__"] %}{{ output }}, {% endfor %}= forward({% for input in network["__input__"] %}{{ input }}, {% endfor %})

        # Determine our losses *sniff*
        {% for loss_name, loss_def in loss.iteritems() if loss_name != "__loss__" %}
        {% if "func" in loss_def %}
        {{ loss_name }} = {{loss_def["func"]}}
        {% else %}
        {{ loss_name }} = {{ loss_def["type"] }}({% for input, value in loss_def["input"].iteritems() %}{{ input }}={{ value }}, {% endfor %}{% if "params" in los_def %}{% for param, value in loss_def["params"].iteritems() %}{{ param }}={{ value }}, {% endfor %}{% endif %})
        {% endif %}
        {% endfor %}

        # Store only the float value (should be smaller than the total loss)
        {% for loss_name, loss_def in loss.iteritems() if loss_name != "__loss__" %}
        self.variables["{{ loss_name }}"]
        {% endfor %}

        # Calculate cumulative loss
        self.loss = {% for loss_name, weight in loss["__loss__"] %}{{ weight }}*{{ loss_name }}{% if not loop.last %} + {% endif %}{% endfor %}

        # Return output
        return self.loss


    def forward(self, {% for input in network["__input__"] %}{{ input }}, {% endfor %}target_layer = None):

        {% for layer_name, layer_def in network["__layers__"].iteritems() %}
        {% if layer_def["type"].startswith('L')  %}
        {{ layer_name }} = chain[{{ layer_name }}]({% for input, value in layer_def["input"].iteritems() %}{{ input }}={{ value }}{% endfor %})
        {% else %}
        {{ layer_name }} = chain[{{ layer_name }}](self, {% for input, value in layer_def["input"].iteritems() %}{{ input }}={{ value }}{% endfor %})
        {% endif %}
        if "{{ layer_name }}" in self.variables:
            self.variables["{{ layer_name }}"] = {{ layer_name }}.data
        if "{{ layer_name }}" == target_layer:
            break

        {% endfor %}

        return{% for output in network["__output__"] %} {{ output }},{% endfor %}