{% macro chainer_link(layer_def) -%}
    {{ layer_def["type"] }}(
        {%- for param, value in layer_def["params"].iteritems() -%}
            {{ param }}={{ value }},
        {%- endfor %})
{%- endmacro %}

{% macro chainer_func(layer_def) -%}
    lambda self,
        {%- for input in layer_def["input"].values() -%}
            {{ input }}
        {%- endfor %}: {{ layer_def["type"] }}(
        {%- for input, value in layer_def["input"].iteritems() %}
            {{ input }}={{ value }},
        {%- endfor -%}
        {%- for param, value in layer_def["params"].iteritems() %}
            {%- if param in network["__states__"] %}
                {{ param }}=self.{{ value }},
            {%- else -%}
                {{ param }}={{ value }},
            {%- endif -%}
        {% endfor %})
{%- endmacro %}

{% macro chainer_loss(loss_def) -%}
{{ loss_def["type"] }}(
{%- for input, value in loss_def["input"].iteritems() -%}
    {{ input }}={{ value }},
{%- endfor -%}
{%- if "params" in los_def -%}
    {%- for param, value in loss_def["params"].iteritems() -%}
        {{ param }}={{ value }}, 
    {%- endfor -%}
{% endif %})
{%- endmacro %}

{% macro link_up(network, chain) %}
{% for layer_name, layer_def in network["__layers__"].iteritems() if layer_def["type"].startswith('L') %}
self.add_link('{{ layer_name }}', self.chain['{{ layer_name }}'])
{% endfor %}
{% endmacro %}

