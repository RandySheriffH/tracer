# Licensed under the MIT license.
'''Render a model graph'''
#pylint: disable=no-member,import-outside-toplevel,too-many-locals,too-many-branches,too-many-statements,too-many-return-statements,protected-access,anomalous-backslash-in-string

import math

directions = ['topdown', 'toleft', 'bottomup', 'toright']

style = {
    'inner_padding': [10, 6],
    'outter_padding': [30, 50],
    'edge_padding': 10,
    'point_space': 5,
    'arrow_length': 8,
    'arrow_width': 3
    }


def render(dc, graph):

    def get_topology():

        topology = {}
        edge_map = {}

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

        return topology, edge_map

    topology, edge_map = get_topology()
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

        if directions[graph['direction']] in ['topdown', 'bottomup']:
            width_per_level[level] += style['outter_padding'][0] * 2 +\
                                      dc.GetTextExtent(graph['vertices'][vertice]['type'])[0] +\
                                      style['inner_padding'][0] * 2
        else:
            width_per_level[level] += style['outter_padding'][1] * 2 +\
                                      dc.GetTextExtent(graph['vertices'][vertice]['type'])[1] +\
                                      style['inner_padding'][1] * 2

    offset_per_level = [0 for v in topology]
    max_width = max(width_per_level.values())

    for vertice in topology:
        level = topology[vertice]['level']
        offset_unit_width = int(max_width/vertices_per_level[level])
        offset_center = offset_unit_width * offset_per_level[level] + int(offset_unit_width/2)
        adjusted_offset = offset_center -\
                          (int((dc.GetTextExtent(graph['vertices'][vertice]['type'])[0] + style['inner_padding'][0] * 2)/2)\
                           if directions[graph['direction']] in ['topdown', 'bottomup'] else\
                           int((dc.GetTextExtent(graph['vertices'][vertice]['type'])[1] + style['inner_padding'][1] * 2)/2))
        topology[vertice]['offset'] = adjusted_offset
        offset_per_level[level] += 1

    max_rect_width = max([dc.GetTextExtent(graph['vertices'][vertice]['type'])[0] for vertice in topology])
    in_ports = {}
    out_ports = {}
    interval_per_level = [[] for v in topology]
    rect_height = 2 * style['inner_padding'][1] + dc.GetTextExtent('A')[1]

    for vertice in graph['vertices']:
        text = graph['vertices'][vertice]['type']
        text_width, _ = dc.GetTextExtent(graph['vertices'][vertice]['type'])
        rect_width = text_width + 2 * style['inner_padding'][0]
        rect_x, rect_y = 0, 0
        if directions[graph['direction']] == 'topdown':
            level = topology[vertice]['level']
            rect_y = rect_height * level + style['outter_padding'][1] * 2 * (level + 1)
            rect_x = topology[vertice]['offset']
            in_ports[vertice] = [rect_x + int(rect_width/2), rect_y]
            out_ports[vertice] = [rect_x + int(rect_width/2), rect_y + rect_height]
            interval_per_level[level].append((rect_x, rect_x + rect_width))

        elif directions[graph['direction']] == 'bottomup':
            level = max(vertices_per_level.keys()) - topology[vertice]['level']
            rect_y = rect_height * level + style['outter_padding'][1] * 2 * (level + 1)
            rect_x = topology[vertice]['offset']
            out_ports[vertice] = [rect_x + int(rect_width/2), rect_y]
            in_ports[vertice] = [rect_x + int(rect_width/2), rect_y + rect_height]
            interval_per_level[topology[vertice]['level']].append((rect_x, rect_x + rect_width))

        elif directions[graph['direction']] == 'toright':
            level = topology[vertice]['level']
            rect_y = topology[vertice]['offset']
            rect_x = max_rect_width * level + style['outter_padding'][0] * 2 * (level + 1)
            in_ports[vertice] = [rect_x, rect_y + rect_height/2]
            out_ports[vertice] = [rect_x + rect_width, rect_y + rect_height/2]
            interval_per_level[level].append((rect_y, rect_y + rect_height))

        else:
            level = max(vertices_per_level.keys()) - topology[vertice]['level']
            rect_y = topology[vertice]['offset']
            rect_x = max_rect_width * level + style['outter_padding'][0] * 2 * (level + 1)
            out_ports[vertice] = [rect_x, rect_y + rect_height/2]
            in_ports[vertice] = [rect_x + rect_width, rect_y + rect_height/2]
            interval_per_level[topology[vertice]['level']].append((rect_y, rect_y + rect_height))

        graph['vertices'][vertice]['rect'] = [rect_x, rect_y, rect_width, rect_height]
        graph['vertices'][vertice]['label'] = [rect_x + style['inner_padding'][0],
                                               rect_y + style['inner_padding'][1]]
        graph['vertices'][vertice]['inport'] = in_ports[vertice]
        graph['vertices'][vertice]['outport'] = out_ports[vertice]

    for vertice in graph['vertices']:
        graph['vertices'][vertice]['edges'] = set()
        from_level = topology[vertice]['level']
        from_point = out_ports[vertice]
        for to_vertice in topology[vertice]['outputs']:
            to_level = topology[to_vertice]['level']
            to_point = in_ports[to_vertice]
            edge = vertice + '>' + to_vertice
            if from_level + 1 == to_level:
                if directions[graph['direction']] in ['topdown', 'bottomup']:
                    mid_point = [to_point[0], int((from_point[1] + to_point[1]))/2]
                else:
                    mid_point = [int((from_point[0] + to_point[0]))/2, to_point[1]]
                graph['edges'][edge] = {'spline': [out_ports[vertice], mid_point, in_ports[to_vertice]]}
            else:
                if directions[graph['direction']] in ['topdown', 'bottomup']:
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
                            if directions[graph['direction']] == 'topdown':
                                mid_point_0 = [tentative_x, from_point[1] + style['outter_padding'][1]]
                                mid_point_1 = [tentative_x, to_point[1] - style['outter_padding'][1]]
                            else:
                                mid_point_0 = [tentative_x, from_point[1] - style['outter_padding'][1]]
                                mid_point_1 = [tentative_x, to_point[1] + style['outter_padding'][1]]
                            graph['edges'][edge] = {'spline': [out_ports[vertice],
                                                               mid_point_0,
                                                               mid_point_1,
                                                               in_ports[to_vertice]]}
                            break
                else:
                    to_down = 1 if to_point[1] >= from_point[1] else -1
                    step = style['edge_padding'] 
                    tentative_ys = [from_point[1] + to_down * d_iter * step for d_iter in range(10)]
                    for tentative_y in tentative_ys:
                        if tentative_y < 0:
                            continue
                        overlap = False
                        for level_iter in range(from_level+1, to_level):
                            for interval in interval_per_level[level_iter]:
                                if tentative_y >= interval[0] and tentative_y <= interval[1]:
                                    overlap = True
                                    break
                            if overlap:
                                break
                        if not overlap:
                            if directions[graph['direction']] == 'toright':
                                mid_point_0 = [from_point[0] + style['outter_padding'][0], tentative_y]
                                mid_point_1 = [to_point[0] - style['outter_padding'][0], tentative_y]
                            else:
                                mid_point_0 = [from_point[0] - style['outter_padding'][0], tentative_y]
                                mid_point_1 = [to_point[0] + style['outter_padding'][0], tentative_y]
                            graph['edges'][edge] = {'spline': [out_ports[vertice],
                                                               mid_point_0,
                                                               mid_point_1,
                                                               in_ports[to_vertice]]}
                            break

    for edge in graph['edges']:
        from_vertice, to_vertice = edge.split('>')
        graph['vertices'][from_vertice]['edges'].add(edge)
        graph['vertices'][to_vertice]['edges'].add(edge)
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

    graph['size'] = (max_width, (len(offset_per_level) + 1) * 2 * style['outter_padding'][1] + 
                                len(offset_per_level) * rect_height)
    graph['selected'] = list(graph['vertices'].keys())[0]
    graph['rendered'] = True



