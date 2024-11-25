.. _{{ fullname }}:

{{ name }}
{{ underline }}

.. automodule:: {{ fullname }}
   {% if classes + functions -%}
   :exclude-members: {% for member in classes + functions %}{{ member }}{% if not loop.last %}, {% endif %}{% endfor %}
   {% endif -%}


{% if classes %}
.. rubric:: Classes

.. autosummary::
   :toctree: generated/
   :nosignatures:

   {% for cls in classes -%}
   {{ cls }}
   {% endfor -%}
{% endif -%}

{% if functions %}
.. rubric:: Functions

.. autosummary::
   :toctree: generated/
   :nosignatures:

   {% for func in functions -%}
   {{ func }}
   {% endfor -%}
{% endif -%}
