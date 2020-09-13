# Contribute
For new format, just subclass [Parser](https://github.com/RandySheriffH/tracer/blob/master/tracer/parsers.py#L10) and override methods:

- get_ops
- get_inputs_outputs
- get_op_name
- get_op_type
- get_attr
- load_graph
- get_type

Then add entry under [parse](https://github.com/RandySheriffH/tracer/blob/master/tracer/parsers.py#L518).
