# Licensed under the MIT license.
''' Parers to read models '''
#pylint: disable=no-member,import-outside-toplevel,too-many-locals,too-many-branches,too-many-statements,too-many-return-statements,protected-access

import os
import json
import time

from pathlib import Path
from graphviz import Digraph
from .utils import GetTemp, INT

class Parser:
    ''' base class of all interfaces'''

    def __init__(self):
        self.count = 0

    def get_ops(self, model_graph):
        '''return list of ops'''
        raise NotImplementedError('Not implemented!')

    def get_inputs_outputs(self, operator):
        '''return lists of input, output, shape and type'''
        raise NotImplementedError('Not implemented!')

    def get_op_name(self, operator):
        '''return name of op'''
        raise NotImplementedError('Not implemented!')

    def get_op_type(self, operator):
        '''return type of op'''
        raise NotImplementedError('Not implemented!')

    def get_attr(self, operator):
        '''return attrs and all referred subgraphs'''
        raise NotImplementedError('Not implemented!')

    def load_graph(self, model_file_path):
        '''return graph and total number of ops'''
        raise NotImplementedError('Not implemented!')

    def get_type(self):
        '''return model type in string'''
        raise NotImplementedError('Not implemented!')

    @staticmethod
    def fill_output_map(graph):
        '''extract all output and map them to ops'''

        def fill_output(graph, output_map):
            for vertice in graph['vertices']:
                for output in graph['vertices'][vertice]['outputs']:
                    output_map[output] = {'graph':graph, 'from': vertice, 'to': {}}
            for subgraph in graph['subgraphs']:
                fill_output(graph['subgraphs'][subgraph], output_map)

        def fill_input(graph, input_map):
            for vertice in graph['vertices']:
                for iter_i in graph['vertices'][vertice]['inputs']:
                    if iter_i in input_map:
                        input_map[iter_i]['to'][vertice] = graph
            for subgraph in graph['subgraphs']:
                fill_input(graph['subgraphs'][subgraph], input_map)

        def write_graph(graph, fixed_map):
            graph['map'] = fixed_map
            for subgraph in graph['subgraphs']:
                write_graph(graph['subgraphs'][subgraph], fixed_map)

        global_map = {}
        fill_output(graph, global_map)
        fill_input(graph, global_map)
        write_graph(graph, global_map)
        return graph

    def parse(self, model_file_path,
              init_progress_callback,
              updage_progress_callback,
              max_node_per_graph=600):
        '''parse graph and return parsed'''

        model_graph, total_ops = self.load_graph(model_file_path)
        init_progress_callback(total_ops)
        ret, _ = updage_progress_callback(0)
        if ret:
            return Parser.fill_output_map(self.parse_graph(model_graph,
                                                           Path(model_file_path).stem,
                                                           updage_progress_callback,
                                                           max_node_per_graph))
        return None

    @staticmethod
    def empty_graph():
        '''empty graph in json'''
        return {'name': '', 'type': '', 'vertices': {}, 'shapes': {},\
               'types': {}, 'edges': {}, 'selected':'', 'subgraphs':{}, 'map':{}}

    def parse_graph(self,
                    model_graph,
                    graph_name,
                    updage_progress_callback,
                    max_node_per_graph):
        '''parse graph and all embedded'''

        graph = Parser.empty_graph()
        graph['name'] = graph_name
        graph['type'] = self.get_type()
        ops = self.get_ops(model_graph)
        num_ops = len(ops)

        if num_ops > max_node_per_graph:
            all_inputs = {}
            all_outputs = {}

            for iter_i in range(0, num_ops, max_node_per_graph):
                sub_graph_name = graph_name + '_vertices_' + str(iter_i/max_node_per_graph + 1)
                sub_graph = Parser.empty_graph()
                sub_graph['name'] = sub_graph_name
                sub_graph['type'] = self.get_type()

                for operator in ops[iter_i: min(iter_i+max_node_per_graph, num_ops)]:
                    attrs, subgraphs = self.get_attr(operator)
                    for subgraph in subgraphs:
                        sub_graph['subgraphs'][subgraph] =\
                            self.parse_graph(subgraphs[subgraph], subgraphs,
                                             updage_progress_callback,
                                             max_node_per_graph)

                    inputs, outputs, output_shapes, output_types = self.get_inputs_outputs(operator)

                    vertice = {'type': self.get_op_type(operator),
                               'attrs': attrs,
                               'inputs': inputs,
                               'outputs': outputs,
                               'edges': []}

                    for iter_ii, output in enumerate(outputs):
                        if output_shapes[iter_ii] is not None:
                            sub_graph['shapes'][output] = output_shapes[iter_ii]
                        if output_types[iter_ii] is not None:
                            sub_graph['types'][output] = output_types[iter_ii]

                    op_name = self.get_op_name(operator)
                    sub_graph['vertices'][op_name] = vertice
                    for iter_ii in inputs:
                        all_inputs[iter_ii] = sub_graph_name
                    for output in outputs:
                        all_outputs[output] = sub_graph_name
                    self.count += 1
                    ret, _ = updage_progress_callback(self.count)
                    if ret is False:
                        return None

                Parser.render(sub_graph)
                graph['vertices'][sub_graph_name] =\
                    {'type': '+',
                     'attrs': {'graph part':{'type': 'subgraph', 'value': sub_graph_name}},
                     'inputs': [],
                     'outputs': [],
                     'edges': []
                    }
                graph['subgraphs'][sub_graph_name] = sub_graph

                for iter_ii in all_inputs:

                    if iter_ii in all_outputs and all_inputs[iter_ii] != all_outputs[iter_ii]:

                        edge_name = all_outputs[iter_ii] + '_to_' + all_inputs[iter_ii]
                        from_vertice = graph['vertices'][all_outputs[iter_ii]]

                        if edge_name not in from_vertice['outputs']:
                            from_vertice['outputs'].append(edge_name)

                        to_vertice = graph['vertices'][all_inputs[iter_ii]]
                        if edge_name not in to_vertice['inputs']:
                            to_vertice['inputs'].append(edge_name)

        else:
            for iter_i, operator in enumerate(ops):
                attrs, subgraphs = self.get_attr(operator)

                for subgraph in subgraphs:
                    graph['subgraphs'][subgraph] = self.parse_graph(subgraphs[subgraph], subgraph,\
                                                            updage_progress_callback,\
                                                            max_node_per_graph)

                inputs, outputs, output_shapes, output_types = self.get_inputs_outputs(operator)
                vertice = {'type': self.get_op_type(operator),
                           'attrs': attrs,
                           'inputs': inputs,
                           'outputs': outputs,
                           'edges': []}

                for iter_ii, output in enumerate(outputs):
                    if output_shapes[iter_ii] is not None:
                        graph['shapes'][output] = output_shapes[iter_ii]
                    if output_types[iter_ii] is not None:
                        graph['types'][output] = output_types[iter_ii]

                op_name = self.get_op_name(operator)
                graph['vertices'][op_name] = vertice
                self.count += 1
                ret, _ = updage_progress_callback(self.count)
                if ret is False:
                    return None

        Parser.render(graph)
        return graph

    @staticmethod
    def render(graph):
        '''collect rendering info'''

        temp_name = str(time.time())
        dot = Digraph(format='json', name=temp_name)
        dot.attr('graph')
        dot.attr('node', shape='box')
        output_from = {}

        for vertice_name in graph['vertices']:
            vertice = graph['vertices'][vertice_name]
            dot.node(vertice_name, vertice['type'])
            for output in vertice['outputs']:
                output_from[output] = vertice_name

        edge_labels = []
        for vertice_name in graph['vertices']:
            vertice = graph['vertices'][vertice_name]
            for iter_i in vertice['inputs']:
                if iter_i in output_from:
                    dot.edge(output_from[iter_i], vertice_name, str(len(edge_labels)))
                    edge_labels.append(output_from[iter_i] + '~' + vertice_name)

        output_path = GetTemp() + temp_name
        dot.render(output_path)

        with open(output_path + '.json', 'r') as render_f:
            jobj = json.load(render_f)
            graph['size'] = (int(float(jobj['bb'].split(',')[2])),
                             int(float(jobj['bb'].split(',')[3])))

            for vertice in jobj['objects']:
                points = vertice['_draw_'][1]['points']
                lefttop = points[2]
                rightbtm = points[0]
                rect = lefttop + [rightbtm[0]-lefttop[0], rightbtm[1]-lefttop[1]]
                graph['vertices'][vertice['name']]['rect'] = INT(rect)
                label = vertice['_ldraw_'][-1]['pt']
                label[0] -= vertice['_ldraw_'][-1]['width']/2
                graph['vertices'][vertice['name']]['label'] = label

            edges = jobj['edges'] if 'edges' in jobj else []

            for edge in edges:
                label = edge_labels[int(edge['label'])]
                graph['edges'][label] =\
                    {'spline': [INT(point) for point in edge['_draw_'][-1]['points']],\
                     'arrow': [INT(point) for point in edge['_hdraw_'][-1]['points']]}
                [input_v, output_v] = label.split('~')
                graph['vertices'][input_v]['edges'].append(label)
                graph['vertices'][output_v]['edges'].append(label)

            graph['selected'] = jobj['objects'][-1]['name']


class OnnxParser(Parser):
    '''parser for onnx models'''

    def __init__(self):
        Parser.__init__(self)
        self.shapes = {}
        self.types = {}

    def get_ops(self, model_graph):
        return model_graph.node

    def get_type(self):
        return 'onnx'

    def get_inputs_outputs(self, operator):
        return operator.input, operator.output,\
               [self.shapes[output] if output in self.shapes\
                else None for output in operator.output],\
               [self.types[output] if output in self.types\
                else None for output in operator.output]

    def get_op_name(self, operator):
        return operator.op_type + '_' + str(self.count)\
               if len(operator.name) == 0 else operator.name

    def get_op_type(self, operator):
        return operator.op_type

    def get_attr(self, operator):
        from onnx import numpy_helper
        from onnx import AttributeProto as AP
        sgs = {}
        attrs = {}
        for attr in operator.attribute:
            if attr.type == AP.FLOAT:
                attrs[attr.name] = {'type': 'string', 'value': str(attr.f)}
            elif attr.type == AP.INT:
                attrs[attr.name] = {'type': 'string', 'value': str(attr.i)}
            elif attr.type == AP.STRING:
                attrs[attr.name] = {'type': 'string', 'value': str(attr.s)}
            elif attr.type == AP.FLOATS:
                attrs[attr.name] = {'type': 'string',
                                    'value': ','.join([str(f) for f in attr.floats])}
            elif attr.type == AP.INTS:
                attrs[attr.name] = {'type': 'string',
                                    'value': ','.join([str(ii) for ii in attr.ints])}
            elif attr.type == AP.STRINGS:
                attrs[attr.name] = {'type': 'string', 'value': ','.join(attr.strings)}
            elif attr.type == AP.GRAPH:
                attrs[attr.name] = {'type': 'subgraph', 'value': attr.g.name}
                sgs[attr.g.name] = attr.g
                self.fill_type_shape(attr.g)
            elif attr.type == AP.GRAPHS:
                for embedded_graph in attr.graphs:
                    attrs[embedded_graph.name] = {'type': 'subgraph', 'value': embedded_graph.name}
                    sgs[embedded_graph.name] = embedded_graph
                    self.fill_type_shape(embedded_graph)
            elif attr.type == AP.TENSOR:
                attrs[attr.name] = {'type': 'tensor', 'value': numpy_helper.to_array(attr.t)}
            elif attr.type == AP.TENSORS:
                for tensor in attr.tensors:
                    attrs[tensor.name] = {'type': 'tensor', 'value': numpy_helper.to_array(tensor)}
            elif attr.type == AP.Sparse_TENSOR:
                attrs[attr.name] = {'type': 'sparse_tensor',
                                    'value': numpy_helper.to_array(attr.sparse_tensor)}
            elif attr.type == AP.Sparse_TENSORS:
                for sparse_tensor in attr.sparse_tensors:
                    attrs[sparse_tensor.name] =\
                        {'type': 'tensor', 'value': numpy_helper.to_array(sparse_tensor)}
            else: raise TypeError("Unknow onnx attribute type")
        if operator.op_type == 'Cast':
            attrs['to']['value'] = OnnxParser.onnx_type_to_str(int(attrs['to']['value']))
        return attrs, sgs

    @staticmethod
    def onnx_type_to_str(elem_type):
        '''convert onnx data type to string'''
        types = ['undefined', 'float', 'uint8_t', 'int8_t', 'uint16_t', 'int16_t',\
                 'int32_t', 'int64_t', 'string', 'bool', 'float16', 'double', 'uint32_t',\
                 'uint64_t', 'complex64', 'complex128', 'bfloat16']
        return types[elem_type]

    def fill_type_shape(self, graph):
        '''attach type shape info to tensor'''
        for i in range(len(graph.value_info)):
            name = graph.value_info[i].name
            type_proto = graph.value_info[i].type
            self.types[name] = OnnxParser.onnx_type_to_str(type_proto.tensor_type.elem_type)
            dims = [d.dim_value for d in type_proto.tensor_type.shape.dim]
            if len(dims) > 0:
                self.shapes[name] = str(dims)

    def load_graph(self, model_file_path):
        import onnx
        from onnx import shape_inference
        model = shape_inference.infer_shapes(onnx.load(model_file_path))
        model = onnx.load(model_file_path)
        self.fill_type_shape(model.graph)
        # print ('model version:', model.model_version)
        return model.graph, self.count_ops(model.graph)

    def count_ops(self, graph):
        '''return num of all operators'''
        from onnx import AttributeProto as AP
        num = len(graph.node)
        for node in graph.node:
            for attr in node.attribute:
                if attr.type == AP.GRAPH:
                    num += self.count_ops(attr.g)
                elif attr.type == AP.GRAPHS:
                    for embedded_graph in attr.graphs:
                        num += self.count_ops(embedded_graph)
        return num


class TFParser(Parser):
    '''parser for tensorflow graph def'''

    def __init__(self):
        Parser.__init__(self)
        self.functions = {}

    def get_ops(self, model_graph):
        return model_graph.get_operations()

    def get_type(self):
        return 'tensorflow'

    def get_inputs_outputs(self, operator):

        def shape(tensor):
            try:
                return tensor.get_shape().as_list()
            except ValueError:
                return 'unknown'

        return [ii.name for ii in operator.inputs],\
               [o.name for o in operator.outputs],\
               [shape(o) for o in operator.outputs],\
               [self.tf_type_to_str(o.dtype) for o in operator.outputs]

    def get_op_name(self, operator):
        return operator.name

    def get_op_type(self, operator):
        return operator.type

    def get_attr(self, operator):
        import tensorflow as tf

        sgs = {}
        attrs = {}
        for key in operator.node_def.attr:
            value = operator.get_attr(key)
            dtype = type(value)
            if dtype is tf.python.framework.dtypes.DType:
                attrs[key] = {'type': 'string', 'value': TFParser.tf_type_to_str(value)}
            elif dtype is tf.core.framework.attr_value_pb2.NameAttrList:
                if key in ['body', 'cond', 'then_branch', 'else_branch']:
                    attrs[key] = {'type': 'subgraph', 'value': value.name}
                    sgs[value.name] = self.get_subgraph(value.name)
                # else: raise TypeError('Unknow TF NameAttrList')
            elif dtype is list:
                if len(value) == 0:
                    pass
                elif isinstance(value[0], tf.python.framework.dtypes.DType):
                    value = ','.join([TFParser.tf_type_to_str(e) for e in value])
                    attrs[key] = {'type': 'string', 'value': value}
                elif isinstance(value[0], tf.core.framework.tensor_shape_pb2.TensorShapeProto):
                    value = ','.join(['[' + '.'.join([str(d.size) for d in e.dim]) + ']'\
                                     for e in value])
                    attrs[key] = {'type': 'string', 'value': value}
                elif isinstance(value[0], int):
                    value = ','.join([str(e) for e in value])
                    attrs[key] = {'type': 'string', 'value': value}
                # else: raise TypeError('unknown tf list element type:', type(value[0]))
            elif dtype is bool:
                attrs[key] = {'type': 'string', 'value': str(value)}
            elif dtype is bytes:
                attrs[key] = {'type': 'string', 'value': str(value)}
            elif dtype is int:
                attrs[key] = {'type': 'string', 'value': str(value)}
            elif dtype is float:
                attrs[key] = {'type': 'string', 'value': str(value)}
            elif dtype is tf.core.framework.tensor_pb2.TensorProto:
                attrs[key] = {'type': 'tensor', 'value': tf.make_ndarray(value)}
            elif dtype is tf.core.framework.tensor_shape_pb2.TensorShapeProto:
                value = '.'.join([str(d.size) for d in value.dim])
                attrs[key] = {'type': 'string', 'value': value}
            # else: raise TypeError('unknown tf attr type:', t)
        return attrs, sgs

    @staticmethod
    def tf_type_to_str(value):
        '''map tf type to string'''

        import tensorflow as tf

        if value is tf.float16:
            return 'float16'
        if value is tf.float32:
            return 'float32'
        if value is tf.float64:
            return 'float64'
        if value is tf.bfloat16:
            return 'bfloat16'
        if value is tf.complex64:
            return 'complex64'
        if value is tf.complex128:
            return 'complex128'
        if value is tf.int8:
            return 'int8'
        if value is tf.uint8:
            return 'uint8'
        if value is tf.uint16:
            return 'uint16'
        if value is tf.uint32:
            return 'uint32'
        if value is tf.uint64:
            return 'uint64'
        if value is tf.int16:
            return 'int16'
        if value is tf.int32:
            return 'int32'
        if value is tf.int64:
            return 'int64'
        if value is tf.bool:
            return 'bool'
        if value is tf.string:
            return 'string'
        if value is tf.qint8:
            return 'qint8'
        if value is tf.quint8:
            return 'quint8'
        if value is tf.qint16:
            return 'qint16'
        if value is tf.quint16:
            return 'quint16'
        if value is tf.qint32:
            return 'qint32'
        if value is tf.resource:
            return 'resource'
        if value is tf.variant:
            return 'variant'
        return str(value)

    def load_graph(self, model_file_path):
        import tensorflow as tf
        tf.compat.v1.reset_default_graph()
        path_stem = os.path.dirname(model_file_path)
        if path_stem.endswith('saved_model'):
            imported = tf.saved_model.load(path_stem)
            from tensorflow.python.framework.convert_to_constants\
                import convert_variables_to_constants_v2
            all_sigs = imported.signatures.keys()
            signatures = [s for s in all_sigs if not s.startswith("_")]
            func = imported.signatures[signatures[0]]
            frozen_func = convert_variables_to_constants_v2(func, lower_control_flow=False)
            graph_def = frozen_func.graph.as_graph_def(add_shapes=True)
            with tf.compat.v1.Session() as sess:
                tf.import_graph_def(graph_def, name='')
                return sess.graph, self.count_ops(sess.graph)
        else:
            with tf.compat.v1.Session() as sess:
                graph_def = tf.compat.v1.GraphDef()
                # print ("graph_def version:", graph_def.version)
                with tf.io.gfile.GFile(model_file_path, 'rb') as model_f:
                    graph_def.ParseFromString(model_f.read())
                    tf.import_graph_def(graph_def, name='')
                    return sess.graph, self.count_ops(sess.graph)

    def get_subgraph(self, subgraph_name):
        '''return embedded graph'''
        return self.functions[subgraph_name]

    def count_ops(self, graph):
        '''return num of all operators'''
        from tensorflow.python.framework.function_def_to_graph import function_def_to_graph
        num = len(graph.get_operations())
        for key, fdef in graph._functions.items():
            sub_tf_graph = function_def_to_graph(fdef.definition)
            self.functions[key] = sub_tf_graph
            num += self.count_ops(sub_tf_graph)
        return num


class TFCKParser(TFParser):
    '''parser for tensorflow checkpoint'''

    def __init__(self):
        TFParser.__init__(self)

    def load_graph(self, model_file_path):
        import tensorflow as tf
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            saver = tf.compat.v1.train.import_meta_graph(model_file_path, clear_devices=True)
            saver.restore(sess, model_file_path[:-5])
            return sess.graph, self.count_ops(sess.graph)


def parse(model_file_path, init_progress_callback, updage_progress_callback):
    '''parse model from file and return graph'''
    suffix = Path(model_file_path).suffix
    if suffix == '.onnx':
        parser = OnnxParser()
    elif suffix == '.pb':
        parser = TFParser()
    elif suffix == '.meta':
        parser = TFCKParser()
    else: raise TypeError('Unkown model type!')
    return parser.parse(model_file_path, init_progress_callback, updage_progress_callback)
