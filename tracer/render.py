# Licensed under the MIT license.
'''Render a model graph'''
#pylint: disable=no-member,import-outside-toplevel,too-many-locals,too-many-branches,too-many-statements,too-many-return-statements,protected-access,anomalous-backslash-in-string,line-too-long

import math

directions = ['topdown', 'toleft', 'bottomup', 'toright']

style = {
    'inner_padding': [10, 6],
    'outter_padding': [30, 50],
    'edge_padding': 10,
    'point_space': 5,
    'arrow_length': 8,
    'arrow_width': 3,
    'max_vertices_per_level': 200,
    'input_color': 'green',
    'output_color': 'yellow',
    }

def render(device_context, graph):
    '''render graph on to the device context'''

    graph['edges'] = {}

    def get_topology():

        topology = {}
        edge_map = {}

        for vertice in graph['vertices']:
            topology[vertice] = {'inputs': set(), 'outputs': set(), 'level': 0}

            for output in graph['vertices'][vertice]['outputs']:
                edge_map[output] = vertice

        for vertice in graph['vertices']:

            for input_ in graph['vertices'][vertice]['inputs']:

                if input_ in edge_map:
                    topology[vertice]['inputs'].add(edge_map[input_])
                    topology[edge_map[input_]]['outputs'].add(vertice)

        def fill_level():
            stack = []
            visited = set()
            for vertice in topology:
                if not topology[vertice]['inputs']:
                    stack.append(vertice)
                    visited.add(vertice)
            while stack:
                top = stack[-1]
                del stack[-1]
                for vertice in topology[top]['outputs']:
                    non_visited = sum([0 if input_ in visited else 1 for input_ in topology[vertice]['inputs']])
                    if not non_visited:
                        topology[vertice]['level'] = max([topology[input_]['level'] for input_ in topology[vertice]['inputs']]) + 1
                        stack.append(vertice)
                        visited.add(vertice)

        fill_level()

        def lift_vertices():

            for _ in range(1000):
                vertice_lifted = False
                for vertice in topology:
                    if not topology[vertice]['inputs']:
                        continue
                    max_upstream_level = max([topology[upstream]['level']\
                        for upstream in topology[vertice]['inputs']]) + 1
                    if max_upstream_level < topology[vertice]['level']:
                        topology[vertice]['level'] = max_upstream_level
                        vertice_lifted = True
                if not vertice_lifted:
                    break

        lift_vertices()

        def lower_vertices():

            for _ in range(1000):
                vertice_lowered = False
                for vertice in topology:
                    if not topology[vertice]['outputs']:
                        continue
                    min_downstream_level = min([topology[downstream]['level']\
                        for downstream in topology[vertice]['outputs']]) - 1
                    if min_downstream_level > topology[vertice]['level']:
                        topology[vertice]['level'] = min_downstream_level
                        vertice_lowered = True
                if not vertice_lowered:
                    break

        lower_vertices()

        def roll_level():
            level_map = {}
            for vertice in topology:
                level = topology[vertice]['level']
                if level not in level_map:
                    level_map[level] = []
                level_map[level].append(vertice)
            current_level = 0
            for level in sorted(level_map.keys()):
                from_offset = 0
                while from_offset < len(level_map[level]):
                    to_offset = min(from_offset + style['max_vertices_per_level'],
                                    len(level_map[level]))
                    for i_offset in range(from_offset, to_offset):
                        topology[level_map[level][i_offset]]['level'] = current_level
                    from_offset = to_offset
                    current_level += 1

        roll_level()
        return topology, edge_map

    topology, _ = get_topology()
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
                device_context.GetTextExtent(graph['vertices'][vertice]['type'])[0] +\
                style['inner_padding'][0] * 2
        else:
            width_per_level[level] += style['outter_padding'][1] * 2 +\
                device_context.GetTextExtent(graph['vertices'][vertice]['type'])[1] +\
                style['inner_padding'][1] * 2

    offset_per_level = [0] * (1 + max(list(width_per_level.keys())))
    max_width = max(width_per_level.values())

    for vertice in topology:
        level = topology[vertice]['level']
        offset_unit_width = int(max_width/vertices_per_level[level])
        offset_center = offset_unit_width * offset_per_level[level] + int(offset_unit_width/2)
        adjusted_offset = offset_center -\
            (int((device_context.GetTextExtent(graph['vertices'][vertice]['type'])[0] +\
                  style['inner_padding'][0] * 2)/2)\
             if directions[graph['direction']] in ['topdown', 'bottomup'] else\
             int((device_context.GetTextExtent(graph['vertices'][vertice]['type'])[1] +\
                  style['inner_padding'][1] * 2)/2))
        topology[vertice]['offset'] = adjusted_offset
        offset_per_level[level] += 1

    max_rect_width = max([device_context.GetTextExtent(graph['vertices'][vertice]['type'])[0]\
                          for vertice in topology])
    top_ports = {}
    btm_ports = {}
    lft_ports = {}
    rit_ports = {}
    interval_per_level = [[] for v in topology]
    rect_height = 2 * style['inner_padding'][1] + device_context.GetTextExtent('A')[1]

    for vertice in graph['vertices']:
        text_width, _ = device_context.GetTextExtent(graph['vertices'][vertice]['type'])
        rect_width = text_width + 2 * style['inner_padding'][0]
        rect_x, rect_y = 0, 0
        if directions[graph['direction']] == 'topdown':
            level = topology[vertice]['level']
            rect_y = rect_height * level + style['outter_padding'][1] * 2 * (level + 1)
            rect_x = topology[vertice]['offset']
            interval_per_level[level].append((rect_x, rect_x + rect_width))

        elif directions[graph['direction']] == 'bottomup':
            level = max(vertices_per_level.keys()) - topology[vertice]['level']
            rect_y = rect_height * level + style['outter_padding'][1] * 2 * (level + 1)
            rect_x = topology[vertice]['offset']
            interval_per_level[topology[vertice]['level']].append((rect_x, rect_x + rect_width))

        elif directions[graph['direction']] == 'toright':
            level = topology[vertice]['level']
            rect_y = topology[vertice]['offset']
            rect_x = max_rect_width * level + style['outter_padding'][0] * 2 * (level + 1)
            interval_per_level[level].append((rect_y, rect_y + rect_height))

        else:
            level = max(vertices_per_level.keys()) - topology[vertice]['level']
            rect_y = topology[vertice]['offset']
            rect_x = max_rect_width * level + style['outter_padding'][0] * 2 * (level + 1)
            interval_per_level[topology[vertice]['level']].append((rect_y, rect_y + rect_height))

        top_ports[vertice] = [rect_x + int(rect_width/2), rect_y]
        btm_ports[vertice] = [rect_x + int(rect_width/2), rect_y + rect_height]
        lft_ports[vertice] = [rect_x, rect_y + int(rect_height/2)]
        rit_ports[vertice] = [rect_x + rect_width, rect_y + int(rect_height/2)]

        graph['vertices'][vertice]['rect'] = [rect_x, rect_y, rect_width, rect_height]
        graph['vertices'][vertice]['label'] = [rect_x + style['inner_padding'][0],
                                               rect_y + style['inner_padding'][1]]

    for from_vertice in graph['vertices']:
        from_rect = graph['vertices'][from_vertice]['rect']

        for to_vertice in topology[from_vertice]['outputs']:
            to_rect = graph['vertices'][to_vertice]['rect']
            edge = from_vertice + '>' + to_vertice
            points = []

            if directions[graph['direction']] == 'topdown':
                if from_rect[1] < to_rect[1]:
                    points.append(btm_ports[from_vertice])
                    points.append([top_ports[to_vertice][0],
                                   btm_ports[from_vertice][1] + style['outter_padding'][1]])
                    points.append(top_ports[to_vertice])
                elif from_rect[1] > to_rect[1]:
                    points.append(top_ports[from_vertice])
                    points.append([btm_ports[to_vertice][0],
                                   top_ports[from_vertice][1] - style['outter_padding'][1]])
                    points.append(btm_ports[to_vertice])
                else:
                    if from_rect[0] < to_rect[0]:
                        points.append(rit_ports[from_vertice])
                        points.append([int((rit_ports[from_vertice][0] +\
                                            lft_ports[to_vertice][0])/2),
                                       rit_ports[from_vertice][1]])
                        points.append(lft_ports[to_vertice])
                    else:
                        points.append(lft_ports[from_vertice])
                        points.append([int((lft_ports[from_vertice][0] +\
                                            rit_ports[to_vertice][0])/2),
                                       lft_ports[from_vertice][1]])
                        points.append(rit_ports[to_vertice])

            elif directions[graph['direction']] == 'bottomup':
                if from_rect[1] > to_rect[1]:
                    points.append(top_ports[from_vertice])
                    points.append([btm_ports[to_vertice][0],
                                   top_ports[from_vertice][1] - style['outter_padding'][1]])
                    points.append(btm_ports[to_vertice])
                elif from_rect[1] < to_rect[1]:
                    points.append(btm_ports[from_vertice])
                    points.append([top_ports[to_vertice][0],
                                   btm_ports[from_vertice][1] + style['outter_padding'][1]])
                    points.append(top_ports[to_vertice])
                else:
                    if from_rect[0] < to_rect[0]:
                        points.append(rit_ports[from_vertice])
                        points.append([int((rit_ports[from_vertice][0] +\
                                            lft_ports[to_vertice][0])/2),
                                       rit_ports[from_vertice][1]])
                        points.append(lft_ports[to_vertice])
                    else:
                        points.append(lft_ports[from_vertice])
                        points.append([int((lft_ports[from_vertice][0] +\
                                            rit_ports[to_vertice][0])/2),
                                       lft_ports[from_vertice][1]])
                        points.append(rit_ports[to_vertice])

            elif directions[graph['direction']] == 'toright':
                if from_rect[0] < to_rect[0]:
                    points.append(rit_ports[from_vertice])
                    points.append([rit_ports[from_vertice][0] + style['outter_padding'][0],
                                   lft_ports[to_vertice][1]])
                    points.append(lft_ports[to_vertice])
                elif from_rect[0] > to_rect[0]:
                    points.append(lft_ports[from_vertice])
                    points.append([lft_ports[from_vertice][0] - style['outter_padding'][0],
                                   rit_ports[to_vertice][1]])
                    points.append(rit_ports[to_vertice])
                else:
                    if from_rect[1] < to_rect[1]:
                        points.append(btm_ports[from_vertice])
                        points.append([btm_ports[from_vertice][0],
                                       int((btm_ports[from_vertice][1] +\
                                            top_ports[to_vertice][1])/2)])
                        points.append(top_ports[to_vertice])
                    else:
                        points.append(top_ports[from_vertice])
                        points.append([top_ports[from_vertice][0],
                                       int((top_ports[from_vertice][1] +\
                                            btm_ports[to_vertice][1])/2)])
                        points.append(btm_ports[to_vertice])

            else:
                if from_rect[0] < to_rect[0]:
                    points.append(rit_ports[from_vertice])
                    points.append([rit_ports[from_vertice][0] + style['outter_padding'][0],
                                   lft_ports[to_vertice][1]])
                    points.append(lft_ports[to_vertice])
                elif from_rect[0] > to_rect[0]:
                    points.append(lft_ports[from_vertice])
                    points.append([lft_ports[from_vertice][0] - style['outter_padding'][0],
                                   rit_ports[to_vertice][1]])
                    points.append(rit_ports[to_vertice])
                else:
                    if from_rect[1] < to_rect[1]:
                        points.append(btm_ports[from_vertice])
                        points.append([btm_ports[from_vertice][0],
                                       int((btm_ports[from_vertice][1] +\
                                            top_ports[to_vertice][1])/2)])
                        points.append(top_ports[to_vertice])
                    else:
                        points.append(top_ports[from_vertice])
                        points.append([top_ports[from_vertice][0],
                                       int((top_ports[from_vertice][1] +\
                                            btm_ports[to_vertice][1])/2)])
                        points.append(btm_ports[to_vertice])

            graph['edges'][edge] = {'spline': points}


    for edge in graph['edges']:
        from_vertice, to_vertice = edge.split('>')
        graph['vertices'][from_vertice]['edges'].add(edge)
        graph['vertices'][to_vertice]['edges'].add(edge)
        mid_point, end_point = graph['edges'][edge]['spline'][-2:]
        if end_point[0] != mid_point[0]:
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
            edge_y = end_point[1] - style['arrow_length'] if mid_point[1] < end_point[1]\
                     else end_point[1] + style['arrow_length']
            graph['edges'][edge]['arrow_center'] = [end_point[0], edge_y]
            graph['edges'][edge]['arrow_left'] = [end_point[0] - style['arrow_width'], edge_y]
            graph['edges'][edge]['arrow_right'] = [end_point[0] + style['arrow_width'], edge_y]

        graph['edges'][edge]['arrow'] = [end_point,
                                         graph['edges'][edge]['arrow_left'],
                                         graph['edges'][edge]['arrow_right']]

    if directions[graph['direction']] in ['topdown', 'bottomup']:
        graph['size'] = (max_width,
                         len(offset_per_level) * 2 * style['outter_padding'][1] +
                         len(offset_per_level) * rect_height)
    else:
        graph['size'] = (len(offset_per_level) * 2 * style['outter_padding'][0] +
                         len(offset_per_level) * max_rect_width, max_width)

    for vertice in topology:
        if graph['vertices'][vertice]['is_const'] is True:
            graph['vertices'][vertice]['is_input'] = False
            graph['vertices'][vertice]['is_output'] = False
        else:
            input_count = sum([0 if graph['vertices'][v]['is_const'] else 1 for v in topology[vertice]['inputs']])
            graph['vertices'][vertice]['is_input'] = input_count == 0
            graph['vertices'][vertice]['is_output'] = len(topology[vertice]['outputs']) == 0

    graph['selected'] = list(graph['vertices'].keys())[0]
    graph['rendered'] = True
