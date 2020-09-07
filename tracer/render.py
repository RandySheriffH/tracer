# Licensed under the MIT license.
'''Render a model graph'''
#pylint: disable=no-member,import-outside-toplevel,too-many-locals,too-many-branches,too-many-statements,too-many-return-statements,protected-access,anomalous-backslash-in-string

import wx, math
from parsers import parse

style = {
    'direction': 'topdown',#downtop, leftright, rightleft
    'inner_padding': [5, 5],
    'outter_padding': [30, 50],
    'edge_padding': 10,
    'point_space': 5,
    'arrow_length': 8,
    'arrow_width': 3
    }


def init_callback(n):
    pass


def update_callback(n):
    return True, True


class ChildFrame(wx.MDIChildFrame):
    '''child frame to show a graph'''

    def __init__(self, parent, graph):
        super(ChildFrame, self).__init__(parent, title="ChildFrame")
        self.graph = graph
        self.render_initialized = False
        self.canvas = wx.ScrolledCanvas(self, True)
        self.canvas.Bind(wx.EVT_PAINT, self.on_paint)
        '''
        x_units = 10
        y_units = 10
        x_steps = math.ceil(float(self.graph['size'][0])/x_units)
        y_steps = math.ceil(float(self.graph['size'][1])/y_units)
        self.canvas.SetScrollbars(x_units, y_units, x_steps, y_steps, 0, 0, True)
        '''
        self.canvas.SetScrollbars(10, 10, 200, 200, 0, 0, True)
        self.Maximize()
        self.Show(True)

    def on_paint(self, _):
        dc = wx.PaintDC(self.canvas)
        dc.Clear()
        self.canvas.PrepareDC(dc)

        if not self.render_initialized:
            self.initialize_render(dc)
            self.render_initialized = True
            '''
            canvas_size = self.canvas.GetSize()
            x_units = canvas_size[0]/2
            y_units = canvas_size[1]/2
            x_steps = math.ceil(float(self.graph['size'][0])/x_units)
            y_steps = math.ceil(float(self.graph['size'][1])/y_units)
            self.canvas.SetScrollbars(x_units, y_units, x_steps, y_steps, 0, 0, True)
            '''

        for vertice in self.graph['vertices']:
            dc.DrawRoundedRectangle(graph['vertices'][vertice]['rect'], 3)
            dc.DrawText(graph['vertices'][vertice]['type'], graph['vertices'][vertice]['label'])
            # dc.DrawCircle(graph['vertices'][vertice]['rect'][:2], 3)
            '''
            dc.DrawCircle((graph['vertices'][vertice]['rect'][0] + graph['vertices'][vertice]['rect'][2],
                           graph['vertices'][vertice]['rect'][1] + graph['vertices'][vertice]['rect'][3]), 3)
            '''
        for edge in self.graph['edges']:
            dc.DrawSpline(self.graph['edges'][edge]['spline'])
            dc.DrawPolygon(self.graph['edges'][edge]['arrow'])
            '''
            dc.DrawCircle(self.graph['edges'][edge]['arrow_center'], 2)
            dc.DrawCircle(self.graph['edges'][edge]['arrow_left'], 1)
            dc.DrawCircle(self.graph['edges'][edge]['arrow_right'], 1)
            '''
        '''
        for port in self.graph['in_ports'] + self.graph['out_ports']:
        # for port in self.graph['mid_ports']:
            dc.DrawCircle(port, 3)
        '''

    def initialize_render(self, dc):
        '''compute the rendering layout'''

        topology = {}
        edge_map = {}
        graph = self.graph
        for vertice in graph['vertices']:
            topology[vertice] = {'inputs':set(), 'outputs': set(), 'level': 0}
            for output in graph['vertices'][vertice]['outputs']:
                edge_map[output] = vertice
        for vertice in graph['vertices']:
            for input_ in graph['vertices'][vertice]['inputs']:
                if input_ in edge_map:
                    topology[vertice]['inputs'].add(edge_map[input_])
                    topology[edge_map[input_]]['outputs'].add(vertice)

        def DFS(vertice, visited, level):
            if vertice not in visited or level > topology[vertice]['level']:
                topology[vertice]['level'] = max(level, topology[vertice]['level'])
                visited.add(vertice)
                for downstream in topology[vertice]['outputs']:
                    DFS(downstream, visited, level + 1)

        for vertice in topology:
            if len(topology[vertice]['inputs']) == 0:
                DFS(vertice, set(), 0)

        def DFS2(vertice, visited):
            if vertice in visited:
                return topology[vertice]['level']
            levels = []
            visited.add(vertice)
            for downstream in topology[vertice]['outputs']:
                levels.append(max(0, DFS2(downstream, visited) - 1))
            topology[vertice]['level'] = min(levels) if levels else topology[vertice]['level']
            return topology[vertice]['level']

        visited = set()
        for vertice in topology:
            if vertice not in visited:
                DFS2(vertice, visited)

        vertices_per_level = {}
        width_per_level = {}

        for vertice in topology:
            level = topology[vertice]['level']

            if level in vertices_per_level:
                vertices_per_level[level] += 1

            else:
                vertices_per_level[level] = 1

            if level not in width_per_level:
                width_per_level[level] = 0

            width_per_level[level] += style['outter_padding'][0] * 2 +\
                                      dc.GetTextExtent(graph['vertices'][vertice]['type'])[0] +\
                                      style['inner_padding'][0] * 2

        offset_per_level = [0 for v in topology]
        max_width = max(width_per_level.values())

        for vertice in topology:
            level = topology[vertice]['level']
            offset_unit_width = int(max_width/vertices_per_level[level])
            offset_center = offset_unit_width * offset_per_level[level] + int(offset_unit_width/2)
            adjusted_offset = offset_center -\
                              int((dc.GetTextExtent(graph['vertices'][vertice]['type'])[0] + style['inner_padding'][0] * 2)/2)
            topology[vertice]['offset'] = adjusted_offset
            offset_per_level[level] += 1

        in_ports = {}
        out_ports = {}
        interval_per_level = [[] for v in topology]
        # mid_ports = []

        rect_height = 2 * style['inner_padding'][1] + dc.GetTextExtent('A')[1]
        for vertice in graph['vertices']:
            text = graph['vertices'][vertice]['type']
            text_width, text_height = dc.GetTextExtent(graph['vertices'][vertice]['type'])
            rect_width = text_width + 2 * style['inner_padding'][0]
            level = topology[vertice]['level']
            rect_height = 2 * style['inner_padding'][1] + text_height
            rect_y = rect_height * level + style['outter_padding'][1] * 2 * (level + 1)
            rect_x = topology[vertice]['offset']
            graph['vertices'][vertice]['rect'] = [rect_x, rect_y, rect_width, rect_height]
            graph['vertices'][vertice]['label'] = [rect_x + style['inner_padding'][0],
                                                   rect_y + style['inner_padding'][1]]
            in_ports[vertice] = [rect_x + int(rect_width/2), rect_y]
            #mid_ports.append((rect_x - style['outter_padding'][0], rect_y - style['outter_padding'][1]))
            out_ports[vertice] = [rect_x + int(rect_width/2), rect_y + rect_height]
            interval_per_level[level].append((rect_x, rect_x + rect_width))

        for vertice in graph['vertices']:
            from_level = topology[vertice]['level']
            from_point = out_ports[vertice]
            for to_vertice in topology[vertice]['outputs']:
                to_level = topology[to_vertice]['level']
                to_point = in_ports[to_vertice]
                edge = vertice + '>' + to_vertice
                if from_level + 1 == to_level:
                    mid_point = [to_point[0], int((from_point[1] + to_point[1]))/2]
                    graph['edges'][edge] = {'spline': [out_ports[vertice], mid_point, in_ports[to_vertice]]}
                else:
                    to_left = 1 if to_point[0] >= from_point[0] else -1
                    step = style['edge_padding'] 
                    tentative_xs = [from_point[0] + to_left * d_iter * step for d_iter in range(10)]
                    for tentative_x in tentative_xs:
                        if tentative_x < 0:
                            continue
                        overlap = False
                        for level_iter in range(from_level+1, to_level):
                            for interval in interval_per_level[level_iter]:
                                if tentative_x >= interval[0] and tentative_x <= interval[1]:
                                    overlap = True
                                    break
                            if overlap:
                                break
                        if not overlap:
                            mid_point_0 = [tentative_x, from_point[1] + style['outter_padding'][1]]
                            mid_point_1 = [tentative_x, to_point[1] - style['outter_padding'][1]]
                            graph['edges'][edge] = {'spline': [out_ports[vertice],
                                                               mid_point_0,
                                                               mid_point_1,
                                                               in_ports[to_vertice]]}
                            break

        points = {}
        for edge in graph['edges']:
            for p_iter in range(len(graph['edges'][edge]['spline'])):
                point = graph['edges'][edge]['spline'][p_iter]
                if tuple(point) not in points:
                    points[tuple(point)] = [edge]
                else:
                    points[tuple(point)].append(edge)

        for point in points:
            edges = points[point]
            edge_array = []
            for edge in edges:
                from_vertice, to_vertice = edge.split('>')
                if point == out_ports[from_vertice]:
                    edge_array.append((in_ports[to_vertice][0], edge))
                else:
                    edge_array.append((out_ports[from_vertice][0], edge))
            edge_array.sort()
            points[point] = [pair[1] for pair in edge_array]
        '''
        for point in points:
            edges = points[point]
            num_edges = len(edges)
            if num_edges > 5:
                edge_0 = edges[0]
                print (graph['edges'][edge_0])
                import copy
                next_spline = copy.deepcopy(graph['edges'][edge_0]['spline'])
                next_spline[0][0] -= 5
                graph['edges'][edge_0]['spline'] = next_spline
                print (graph['edges'][edge_0])
        '''
        for point in points:
            edges = points[point]
            num_edges = len(edges)
            if num_edges > 1:
                start_x = point[0] - int(num_edges/2) * style['point_space']
                splits = [start_x + ii * style['point_space'] for ii in range(num_edges)]
                # print ('splits:', splits, 'on', point)
                for e_iter in range(num_edges):
                    edge = edges[e_iter]
                    import copy
                    next_spline = copy.deepcopy(graph['edges'][edge]['spline'])
                    for p_iter in range(len(next_spline)):
                        if next_spline[p_iter][0] == point[0] and next_spline[p_iter][1] == point[1]:
                            next_spline[p_iter][0] = splits[e_iter]
                            break
                    graph['edges'][edge]['spline'] = next_spline

        for edge in graph['edges']:
            mid_point, end_point = graph['edges'][edge]['spline'][-2:]
            if end_point[0] != mid_point[0]:
                k = (end_point[1] - mid_point[1]) / (end_point[0] - mid_point[0])
                edge_x = abs(end_point[0] - mid_point[0])
                edge_y = abs(end_point[1] - mid_point[1])
                edge_z = math.sqrt(edge_x * edge_x + edge_y * edge_y)
                sin = edge_y / edge_z
                cos = edge_x / edge_z
                dis_x = style['arrow_length'] * cos
                dis_y = style['arrow_length'] * sin
                arrow_x = end_point[0] + (dis_x if mid_point[0] > end_point[0] else -dis_x)
                arrow_y = end_point[1] + (dis_y if mid_point[1] > end_point[1] else -dis_y)
                graph['edges'][edge]['arrow_center'] = [arrow_x, arrow_y]

                edge_axis_x = abs(style['arrow_width'] * sin)
                edge_axis_y = abs(style['arrow_width'] * cos)
                graph['edges'][edge]['arrow_left'] = [arrow_x - edge_axis_x, 0]
                graph['edges'][edge]['arrow_right'] = [arrow_x + edge_axis_x, 0]
                if arrow_x < end_point[0] and arrow_y < end_point[1] or\
                   arrow_x > end_point[0] and arrow_y > end_point[1]:
                    graph['edges'][edge]['arrow_left'][1] = arrow_y + edge_axis_y
                    graph['edges'][edge]['arrow_right'][1] = arrow_y - edge_axis_y
                else:
                    graph['edges'][edge]['arrow_left'][1] = arrow_y - edge_axis_y
                    graph['edges'][edge]['arrow_right'][1] = arrow_y + edge_axis_y

            else:
                edge_y = end_point[1] - style['arrow_length'] if mid_point[1] < end_point[1] else end_point[1] + style['arrow_length']
                graph['edges'][edge]['arrow_center'] = [end_point[0], edge_y]
                graph['edges'][edge]['arrow_left'] = [end_point[0] - style['arrow_width'], edge_y]
                graph['edges'][edge]['arrow_right'] = [end_point[0] + style['arrow_width'], edge_y]

            graph['edges'][edge]['arrow'] = [end_point,
                                             graph['edges'][edge]['arrow_left'],
                                             graph['edges'][edge]['arrow_right']]

        #graph['in_ports'] = in_ports
        #graph['mid_ports'] = mid_ports
        #graph['out_ports'] = out_ports
        graph['size'] = (max_width, (len(offset_per_level) + 1) * 2 * style['outter_padding'][1] + 
                                    len(offset_per_level) * rect_height)

class MainFrame(wx.MDIParentFrame):
    '''main frame'''
    def __init__(self, parent, graph):
        super(MainFrame, self).__init__(parent,
                                        title='Topology',
                                        size=(500, 300),
                                        style=wx.DEFAULT_FRAME_STYLE|wx.FRAME_NO_WINDOW_MENU)
        self.child_frame = ChildFrame(self, graph)
        self.Show(True)


if __name__ == '__main__':
    # graph = parse('.\\loop.pb', init_callback, update_callback, False)
    # graph = parse('C:\\Users\\rashuai\\OneDrive\\TheWork\\models\\cond.pb', init_callback, update_callback, False)
    graph = parse('C:\\Users\\rashuai\\OneDrive\\TheWork\\models\\part.onnx', init_callback, update_callback, False)
    ex = wx.App()
    MainFrame(None, graph)
    ex.MainLoop()


