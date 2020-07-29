# Contribute
To add parser for new model format, just create a new class inherit from class [Parser](https://github.com/RandySheriffH/tracer/blob/master/tracer/parsers.py#L14) and define all unimplemented methods:

- get_ops(self, model_graph)
- get_inputs_outputs(self, operator)
- get_op_name(self, operator)
- get_op_type(self, operator)
- get_attr(self, operator)
- load_graph(self, model_file_path)
- get_type(self)

Finally, go to function [parse](https://github.com/RandySheriffH/tracer/blob/master/tracer/parsers.py#L572) in the same file to add new entry for the class.