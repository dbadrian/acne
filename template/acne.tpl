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
