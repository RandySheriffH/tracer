# Contribute
To add parser for new model format, just subclass [Parser](https://github.com/RandySheriffH/tracer/blob/master/tracer/parsers.py#L14) and override methods:

- get_ops
- get_inputs_outputs
- get_op_name
- get_op_type
- get_attr
- load_graph
- get_type

Finally, go to function [parse](https://github.com/RandySheriffH/tracer/blob/master/tracer/parsers.py#L590) to add new entry for the class.
