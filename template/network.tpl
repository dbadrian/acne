#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import OrderedDict

import chainer
import chainer.links as L
import chainer.functions as F

# import acne.methods

{% import 'acne.tpl' as acne %}
class {{ meta.name }}(chainer.Chain):

    def __init__(self):
        super({{ meta.name }}, self).__init__()

        self.chain = OrderedDict()
        {% for layer_name, layer_def in network["__layers__"].iteritems() %}
        self.chain['{{ layer_name }}'] =
            {%- if layer_def["type"].startswith('L')  %}
                {{ acne.chainer_link(layer_def) }}
            {% else %}
                {{ acne.chainer_func(layer_def) }}
            {% endif %}
        {% endfor %}

        self.variables = {}
        {% for layer_name, layer_def in network["__layers__"].iteritems() %}{% if layer_def["is_variable"] %}
        self.variables["{{ layer_name }}"] = None
        {% endif %}{% endfor %}

        # Generate Additional State Variables
        {% for state, value in network['__states__'].iteritems() %}
        self.{{ state }} = {{ value }}
        {% endfor %}

        # Register Links with the Chainer library
        {% for layer_name, layer_def in network["__layers__"].iteritems() if layer_def["type"].startswith('L') %}
        self.add_link('{{ layer_name }}', self.chain['{{ layer_name }}'])
        {% endfor %}

    def __call__(self, {% for var in data %}{{ var }}, {% endfor %}):
        ({% for output in network["__output__"] %}{{ output }}, {% endfor %}) = self.forward({% for input in network["__input__"] %}{{ input }}, {% endfor %})

        # Determine our losses *sniff*
        {% for loss_name, loss_def in loss.iteritems() if loss_name != "__loss__" %}
        {% if "func" in loss_def %}
        {{ loss_name }} = {{loss_def["func"]}}
        {% else %}
        {{ loss_name }} = {{ acne.chainer_loss(loss_def) }}
        {% endif %}
        {% endfor %}

        # Store only the float value (should be smaller than the total loss)
        {% for loss_name, loss_def in loss.iteritems() if loss_name != "__loss__" %}
        self.{{ loss_name }} = {{ loss_name }}.data
        {% endfor %}

        # Calculate cumulative loss
        self.loss = {% for loss_name, weight in loss["__loss__"] %}{{ weight }}*{{ loss_name }}{% if not loop.last %} + {% endif %}{% endfor %}


        # Return output
        return self.loss

    def forward(self, {% for input in network["__input__"] %}{{ input }}, {% endfor %}target_layer=None):
        {% for layer_name, layer_def in network["__layers__"].iteritems() %}
        {% if layer_def["type"].startswith('L')  %}
        {{ layer_name }} = self.chain['{{ layer_name }}']({% for input, value in layer_def["input"].iteritems() %}{{ input }}={{ value }}{% endfor %})
        {% else %}
        {{ layer_name }} = self.chain['{{ layer_name }}'](self, {% for value in layer_def["input"].values() %}{{ value }}={{ value }}{% endfor %})
        {% endif %}
        if "{{ layer_name }}" in self.variables:
            self.variables["{{ layer_name }}"] = {{ layer_name }}.data
        if "{{ layer_name }}" == target_layer:
            return {{ layer_name }}

        {% endfor %}
        return{% for output in network["__output__"] %} {{ output }},
        {% endfor %}