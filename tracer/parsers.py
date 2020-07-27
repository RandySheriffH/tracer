import os
from pathlib import Path
from graphviz import Digraph
import time
import json
from utils import *

class Parser():

    def __init__(self):
        self.count = 0

    def GetOps(self, model_graph): # return list of ops
        raise NotImplementedError('Not implemented!')

    def GetInputsOutputs(self, op): # return lists of input, output, shape and type
        raise NotImplementedError('Not implemented!')

    def GetOpName(self, op): # return name of op
        raise NotImplementedError('Not implemented!')

    def GetOpType(self, op): # return type of op
        raise NotImplementedError('Not implemented!')

    def GetAttr(self, op): # return attrs and all referred subgraphs
        raise NotImplementedError('Not implemented!')

    def LoadGraph(self, model_file_path): # return graph and total number of ops
        raise NotImplementedError('Not implemented!')

    def GetType(self):
        raise NotImplementedError('Not implemented!')

    def Parse(self, model_file_path, init_progress_callback, updage_progress_callback, max_node_per_graph=600):
        model_graph, total_ops = self.LoadGraph(model_file_path)
        init_progress_callback(total_ops)
        ret, _ = updage_progress_callback(0)
        if ret: return self.ParseGraph(model_graph, Path(model_file_path).stem, updage_progress_callback, max_node_per_graph)
        else: return None

    def ParseGraph(self, model_graph, name, updage_progress_callback, max_node_per_graph):
        graph = {'name': name, 'type': self.GetType(), 'vertices': {}, 'shapes': {}, 'types': {}, 'edges': {}, 'selected':'', 'subgraphs':{}}
        ops = self.GetOps(model_graph)
        num_ops = len(ops)

        if num_ops > max_node_per_graph:
            all_inputs = {}
            all_outputs = {}

            for i in range(0, num_ops, max_node_per_graph):
                sub_graph_name = name + '_' + 'vertice_set_' + str(i/max_node_per_graph + 1)
                sub_graph = {'name': sub_graph_name, 'type': self.GetType(), 'vertices': {}, 'shapes': {}, 'types': {}, 'edges': {}, 'selected':'', 'subgraphs':{}}

                for op in ops[i: min(i+max_node_per_graph, num_ops)]:
                    attrs, sg = self.GetAttr(op)
                    for g in sg:
                        sub_graph['subgraphs'][g] = self.ParseGraph(sg[g], sg,\
                                                                    updage_progress_callback,\
                                                                    max_node_per_graph)

                    inputs, outputs, output_shapes, output_types = self.GetInputsOutputs(op)

                    vertice = {'type': self.GetOpType(op),
                               'attrs': attrs,
                               'inputs': inputs,
                               'outputs': outputs,
                               'edges': []}

                    for ii, o in enumerate(outputs):
                        if output_shapes[ii] is not None: sub_graph['shapes'][o] = output_shapes[ii]
                        if output_types[ii] is not None: sub_graph['types'][o] = output_types[ii]

                    op_name = self.GetOpName(op)
                    sub_graph['vertices'][op_name] = vertice
                    for ii in inputs:
                        all_inputs[ii] = sub_graph_name
                    for o in outputs:
                        all_outputs[o] = sub_graph_name
                    self.count += 1
                    ret, _ = updage_progress_callback(self.count)
                    if ret is False:
                        return None

                self.Render(sub_graph)
                graph['vertices'][sub_graph_name] = {'type': '+',
                                                     'attrs': {'graph part': {'type': 'subgraph', 'value': sub_graph_name}},
                                                     'inputs': [],
                                                     'outputs': [],
                                                     'edges': []
                                                    }
                graph['subgraphs'][sub_graph_name] = sub_graph

                for ii in all_inputs:

                    if ii in all_outputs and all_inputs[ii] != all_outputs[ii]:
                        edge_name = all_outputs[ii] + '_to_' + all_inputs[ii]
                        from_vertice = graph['vertices'][all_outputs[ii]]

                        if edge_name not in from_vertice['outputs']:
                            from_vertice['outputs'].append(edge_name)

                        to_vertice = graph['vertices'][all_inputs[ii]]
                        if edge_name not in to_vertice['inputs']:
                            to_vertice['inputs'].append(edge_name)

        else:
            for i, op in enumerate(ops):
                attrs, sgs = self.GetAttr(op)
                for sg in sgs:
                    graph['subgraphs'][sg] = self.ParseGraph(sgs[sg], sg,\
                                                            updage_progress_callback,\
                                                            max_node_per_graph)

                inputs, outputs, output_shapes, output_types = self.GetInputsOutputs(op)
                vertice = {'type': self.GetOpType(op),
                           'attrs': attrs,
                           'inputs': inputs,
                           'outputs': outputs,
                           'edges': []}

                for ii, o in enumerate(outputs):
                    if output_shapes[ii] is not None: graph['shapes'][o] = output_shapes[ii]
                    if output_types[ii] is not None: graph['types'][o] = output_types[ii]

                op_name = self.GetOpName(op)
                graph['vertices'][op_name] = vertice
                self.count += 1
                ret, _ = updage_progress_callback(self.count)
                if ret is False: return None

        self.Render(graph)
        return graph

    def Render(self, graph):
        temp_name = str(time.time())
        dot = Digraph(format='json', name=temp_name)
        dot.attr('graph')
        dot.attr('node', shape='box')
        output_from = {}

        for v in graph['vertices']:
            vertice = graph['vertices'][v]
            dot.node(v, vertice['type'])
            for output in vertice['outputs']:
                output_from[output] = v

        edge_labels = []
        for v in graph['vertices']:
            vertice = graph['vertices'][v]
            for i in vertice['inputs']:
                if i in output_from:
                    dot.edge(output_from[i], v, str(len(edge_labels)))
                    edge_labels.append(output_from[i] + '~' + v)
        output_path = GetTemp() + temp_name
        dot.render(output_path)

        with open(output_path + '.json', 'r') as f:
            jobj = json.load(f)
            graph['size'] = (int(float(jobj['bb'].split(',')[2])), int(float(jobj['bb'].split(',')[3])))

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
                [iv, ov] = label.split('~')
                graph['vertices'][iv]['edges'].append(label)
                graph['vertices'][ov]['edges'].append(label)

            graph['selected'] = jobj['objects'][-1]['name']


class OnnxParser(Parser):

    def __init__(self):
        Parser.__init__(self)
        self.shapes = {}
        self.types = {}

    def GetOps(self, model_graph):
        return model_graph.node

    def GetType(self):
        return 'onnx'

    def GetInputsOutputs(self, op):
        return op.input, op.output,\
               [self.shapes[o] if o in self.shapes else None for o in op.output],\
               [self.types[o] if o in self.types else None for o in op.output]

    def GetOpName(self, op):
        return op.op_type + '_' + str(self.count) if len(op.name) == 0 else op.name

    def GetOpType(self, op):
        return op.op_type

    def GetAttr(self, op):
        from onnx import numpy_helper
        from onnx import AttributeProto as AP
        sgs = {}
        attrs = {}
        for attr in op.attribute:
            if attr.type == AP.FLOAT:
                attrs[attr.name] = {'type': 'string', 'value': str(attr.f)}
            elif attr.type == AP.INT:
                attrs[attr.name] = {'type': 'string', 'value': str(attr.i)}
            elif attr.type == AP.STRING:
                attrs[attr.name] = {'type': 'string', 'value': str(attr.s)}
            elif attr.type == AP.FLOATS:
                attrs[attr.name] = {'type': 'string', 'value': ','.join([str(f) for f in attr.floats])}
            elif attr.type == AP.INTS:
                attrs[attr.name] = {'type': 'string', 'value': ','.join([str(ii) for ii in attr.ints])}
            elif attr.type == AP.STRINGS:
                attrs[attr.name] = {'type': 'string', 'value': ','.join(attr.strings)}
            elif attr.type == AP.GRAPH:
                attrs[attr.name] = {'type': 'subgraph', 'value': attr.g.name}
                sgs[attr.g.name] = attr.g
                self.FillTypeShape(attr.g)
            elif attr.type == AP.GRAPHS:
                for g in attr.graphs:
                    attrs[g.name] = {'type': 'subgraph', 'value': g.name}
                    sgs[g.name] = g
                    self.FillTypeShape(g)
            elif attr.type == AP.TENSOR:
                attrs[attr.name] = {'type': 'tensor', 'value': numpy_helper.to_array(attr.t)}
            elif attr.type == AP.TENSORS:
                for t in attr.tensors:
                    attrs[t.name] = {'type': 'tensor', 'value': numpy_helper.to_array(t)}
            elif attr.type == AP.SPARSE_TENSOR:
                attrs[attr.name] = {'type': 'sparse_tensor', 'value': numpy_helper.to_array(attr.sparse_tensor)}
            elif attr.type == AP.SPARSE_TENSORS:
                for st in attr.sparse_tensors:
                    attrs[st.name] = {'type': 'tensor', 'value': numpy_helper.to_array(st)}
            else: raise TypeError("Unknow onnx attribute type")
        if op.op_type == 'Cast':
            attrs['to']['value'] = self.OnnxType2Str(int(attrs['to']['value']))
        return attrs, sgs

    def OnnxType2Str(self, elem_type):
        types = ['undefined', 'float', 'uint8_t', 'int8_t', 'uint16_t', 'int16_t',\
                 'int32_t', 'int64_t', 'string', 'bool', 'float16', 'double', 'uint32_t',\
                 'uint64_t', 'complex64', 'complex128', 'bfloat16']
        return types[elem_type]

    def FillTypeShape(self, graph):
        for i in range(len(graph.value_info)):
            name = graph.value_info[i].name
            type_proto = graph.value_info[i].type
            self.types[name] = self.OnnxType2Str(type_proto.tensor_type.elem_type)
            try:
                dims = [d.dim_value for d in type_proto.tensor_type.shape.dim]
                if len(dims) > 0: self.shapes[name] = str(dims)
            except Exception:
                pass

    def LoadGraph(self, model_file_path):
        import onnx
        from onnx import shape_inference
        try:
            model = shape_inference.infer_shapes(onnx.load(model_file_path))
        except:
            model = onnx.load(model_file_path)
        self.FillTypeShape(model.graph)
        return model.graph, self.CountOps(model.graph)

    def CountOps(self, graph):
        from onnx import AttributeProto as AP
        num = len(graph.node)
        for n in graph.node:
            for a in n.attribute:
                if a.type == AP.GRAPH:
                    num += self.CountOps(a.g)
                elif a.type == AP.GRAPHS:
                    for g in a.graphs:
                        num += CountOps(g)
        return num


class TFParser(Parser):

    def __init__(self):
        Parser.__init__(self)
        self.functions = {}

    def GetOps(self, model_graph):
        return model_graph.get_operations()

    def GetType(self):
        return 'tensorflow'

    def GetInputsOutputs(self, op):

        def Shape(tensor):
            try:
                return tensor.get_shape().as_list()
            except Exception:
                return None

        return [ii.name for ii in op.inputs],\
               [o.name for o in op.outputs],\
               [Shape(o) for o in op.outputs],\
               [self.TFType2Str(o.dtype) for o in op.outputs]

    def GetOpName(self, op):
        return op.name

    def GetOpType(self, op):
        return op.type

    def GetAttr(self, op):
        import tensorflow as tf

        sgs = {}
        attrs = {}
        for k in op.node_def.attr:
            v = op.get_attr(k)
            t = type(v)
            if t is tf.python.framework.dtypes.DType:
                attrs[k] = {'type': 'string', 'value': self.TFType2Str(v)}
            elif t is tf.core.framework.attr_value_pb2.NameAttrList:
                if k in ['body', 'cond', 'then_branch', 'else_branch']:
                    attrs[k] = {'type': 'subgraph', 'value': v.name}
                    sgs[v.name] = self.GetSubGraph(v.name)
                # else: raise TypeError('Unknow TF NameAttrList')
            elif t is list:
                if len(v) == 0: pass
                elif type(v[0]) is tf.python.framework.dtypes.DType:
                    value = ','.join([self.TFType2Str(e) for e in v])
                    attrs[k] = {'type': 'string', 'value': value}
                elif type(v[0]) is tf.core.framework.tensor_shape_pb2.TensorShapeProto:
                    value = ','.join(['[' + '.'.join([str(d.size) for d in e.dim]) + ']' for e in v])
                    attrs[k] = {'type': 'string', 'value': value}
                elif type(v[0]) is int:
                    value = ','.join([str(e) for e in v])
                    attrs[k] = {'type': 'string', 'value': value}
                # else: raise TypeError('unknown tf list element type:', type(v[0]))
            elif t is bool:
                attrs[k] = {'type': 'string', 'value': str(v)}
            elif t is bytes:
                attrs[k] = {'type': 'string', 'value': str(v)}
            elif t is int:
                attrs[k] = {'type': 'string', 'value': str(v)}
            elif t is float:
                attrs[k] = {'type': 'string', 'value': str(v)}
            elif t is tf.core.framework.tensor_pb2.TensorProto:
                attrs[k] = {'type': 'tensor', 'value': tf.make_ndarray(v)}
            elif t is tf.core.framework.tensor_shape_pb2.TensorShapeProto:
                value = '.'.join([str(d.size) for d in v.dim])
                attrs[k] = {'type': 'string', 'value': value}
            # else: raise TypeError('unknown tf attr type:', t)
        return attrs, sgs

    def TFType2Str(self, v):
        import tensorflow as tf

        if v is tf.float16: return 'float16'
        elif v is tf.float32: return 'float32'
        elif v is tf.float64: return 'float64'
        elif v is tf.bfloat16: return 'bfloat16'
        elif v is tf.complex64: return 'complex64'
        elif v is tf.complex128: return 'complex128'
        elif v is tf.int8: return 'int8'
        elif v is tf.uint8: return 'uint8'
        elif v is tf.uint16: return 'uint16'
        elif v is tf.uint32: return 'uint32'
        elif v is tf.uint64: return 'uint64'
        elif v is tf.int16: return 'int16'
        elif v is tf.int32: return 'int32'
        elif v is tf.int64: return 'int64'
        elif v is tf.bool: return 'bool'
        elif v is tf.string: return 'string'
        elif v is tf.qint8: return 'qint8'
        elif v is tf.quint8: return 'quint8'
        elif v is tf.qint16: return 'qint16'
        elif v is tf.quint16: return 'quint16'
        elif v is tf.qint32: return 'qint32'
        elif v is tf.resource: return 'resource'
        elif v is tf.variant: return 'variant'
        else: return str(v)

    def LoadGraph(self, model_file_path):
        import tensorflow as tf
        tf.compat.v1.reset_default_graph()
        path_stem = os.path.dirname(model_file_path)
        if path_stem.endswith('saved_model'):
            imported = tf.saved_model.load(path_stem)
            convert_variables_to_constants = tf.compat.v1.graph_util.convert_variables_to_constants
            from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
            all_sigs = imported.signatures.keys()
            signatures = [s for s in all_sigs if not s.startswith("_")]
            func = imported.signatures[signatures[0]]
            frozen_func = convert_variables_to_constants_v2(func, lower_control_flow=False)
            graph_def = frozen_func.graph.as_graph_def(add_shapes=True)
            with tf.compat.v1.Session() as sess:
                tf.import_graph_def(graph_def, name='')
                return sess.graph, self.CountOps(sess.graph)
        else:
            with tf.compat.v1.Session() as sess:
                graph_def = tf.compat.v1.GraphDef()
                with tf.io.gfile.GFile(model_file_path, 'rb') as f:
                    graph_def.ParseFromString(f.read())
                    tf.import_graph_def(graph_def, name='')
                    return sess.graph, self.CountOps(sess.graph)

    def GetSubGraph(self, subgraph_name):
        return self.functions[subgraph_name]

    def CountOps(self, graph):
        from tensorflow.python.framework.function_def_to_graph import function_def_to_graph
        num = len(graph.get_operations())
        for k, fdef in graph._functions.items():
            sub_tf_graph = function_def_to_graph(fdef.definition)
            self.functions[k] = sub_tf_graph
            num += self.CountOps(sub_tf_graph)
        return num


class TFCKParser(TFParser):

    def __init__(self):
        TFParser.__init__(self)

    def LoadGraph(self, model_file_path):
        import tensorflow as tf
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            saver = tf.compat.v1.train.import_meta_graph(model_file_path, clear_devices=True)
            saver.restore(sess, model_file_path[:-5])
            return sess.graph, self.CountOps(sess.graph)


def Parse(model_file_path, init_progress_callback, updage_progress_callback):
    suffix = Path(model_file_path).suffix
    if suffix == '.onnx':
        parser = OnnxParser()
    elif suffix == '.pb':
        parser = TFParser()
    elif suffix == '.meta':
        parser = TFCKParser()
    else: raise TypeError('Unkown model type!')
    return parser.Parse(model_file_path, init_progress_callback, updage_progress_callback)