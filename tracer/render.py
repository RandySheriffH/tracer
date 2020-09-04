# Licensed under the MIT license.
'''Render a model graph'''
#pylint: disable=no-member,import-outside-toplevel,too-many-locals,too-many-branches,too-many-statements,too-many-return-statements,protected-access,anomalous-backslash-in-string

import wx
from parsers import parse

style = {
    'direction': 'topdown',
    'char_height': 18, 'char_width': 6,
    'inner_padding': [5, 5], 'outter_padding': [30, 100]
    }


def fill_level(topology):

    def DFS(vertice, visited, level):
        if vertice in visited:
            if level > topology[vertice]['level']:
                topology[vertice]['level'] = level
                for downstream in topology[vertice]['outputs']:
                    DFS(downstream, visited, level + 1)
        else:
            topology[vertice]['level'] = level
            visited.add(vertice)
            for downstream in topology[vertice]['outputs']:
                DFS(downstream, visited, level + 1)

    for vertice in topology:
        if topology[vertice]['level'] == 0:
            DFS(vertice, set(), 0)


def fill_offset(graph, topology):

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
                                  len(graph['vertices'][vertice]['type']) * style['char_width'] +\
                                  style['inner_padding'][0] * 2

    offset_per_level = [0 for v in topology]
    max_width = max(width_per_level.values())
    for vertice in topology:
        level = topology[vertice]['level']
        offset_unit_width = int(max_width/vertices_per_level[level])
        offset_center = offset_unit_width * offset_per_level[level] + int(offset_unit_width/2)
        adjusted_offset = offset_center -\
                          int((len(graph['vertices'][vertice]['type']) * style['char_width'] +\
                               style['inner_padding'][0] * 2)/2)
        topology[vertice]['offset'] = adjusted_offset
        offset_per_level[level] += 1


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
        self.Maximize()
        self.Show(True)

    def on_paint(self, _):
        dc = wx.PaintDC(self.canvas)
        dc.Clear()
        if not self.render_initialized:
            self.initialize_render(dc)
            self.render_initialized = True
        for vertice in self.graph['vertices']:
            dc.DrawRoundedRectangle(graph['vertices'][vertice]['rect'], 3)
            dc.DrawText(graph['vertices'][vertice]['type'], graph['vertices'][vertice]['label'])

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
        fill_level(topology)

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

        rect_height = 2 * style['inner_padding'][1] + dc.GetTextExtent('A')[1]
        for vertice in graph['vertices']:
            text = graph['vertices'][vertice]['type']
            text_width, text_height = dc.GetTextExtent(graph['vertices'][vertice]['type'])
            rect_width = text_width + 2 * style['inner_padding'][0]
            level = topology[vertice]['level']
            rect_height = 2 * style['inner_padding'][1] + text_height
            rect_y = rect_height * level + style['outter_padding'][1] * (level + 1)
            rect_x = topology[vertice]['offset']
            graph['vertices'][vertice]['rect'] = [rect_x, rect_y, rect_width, rect_height]
            graph['vertices'][vertice]['label'] = [rect_x + style['inner_padding'][0],
                                                   rect_y + style['inner_padding'][1]]


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
    graph = parse('.\\loop.pb', init_callback, update_callback, False)
    # graph = parse('C:\\Users\\rashuai\\OneDrive\\TheWork\\models\\part.onnx', init_callback, update_callback, False)
    ex = wx.App()
    MainFrame(None, graph)
    ex.MainLoop()
