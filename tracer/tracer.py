# Licensed under the MIT license.
'''tracer graphic user interface by wx'''
#pylint: disable=no-member,import-outside-toplevel,too-many-instance-attributes,too-many-locals,too-many-branches,too-many-statements,too-many-return-statements,protected-access,no-name-in-module,too-few-public-methods,invalid-name,chained-comparison

import os
import sys
import math
import wx
from wx import Point, propgrid
import graphviz
from .parsers import parse
from .utils import to_int, pwd, create_temp, remove_temp


class About(wx.Dialog):
    '''about info for tracer'''

    def __init__(self, parent):
        super(About, self).__init__(parent, size=(360, 120))
        self.info = wx.StaticText(self, pos=Point(80, 20))
        self.info.SetLabel("Tracer by Ran Shuai, MIT license")
        self.mail = wx.StaticText(self, pos=Point(90, 40))
        self.mail.SetLabel("randysheriff@hotmail.com")
        self.Center()
        self.Show()


class SubgraphProperty(wx.propgrid.LongStringProperty):
    '''subgraph property to allow for trace embedded graph'''

    def __init__(self, frame, label=propgrid.PG_LABEL, name=propgrid.PG_LABEL, value=''):
        super(SubgraphProperty, self).__init__(label, name, value)
        self.frame = frame

    def DisplayEditorDialog(self, _, label):
        '''respond to button click event'''
        graph = self.frame.get_subgraph(self.frame.graph, label)
        self.frame.GetParent().show_frame(graph)
        return (False, '')


class InputProperty(wx.propgrid.LongStringProperty):
    '''input property to allow for trace input'''

    def __init__(self, frame,
                 label=propgrid.PG_LABEL,
                 name=propgrid.PG_LABEL,
                 value=''):
        super(InputProperty, self).__init__(label, name, value)
        self.frame = frame

    def DisplayEditorDialog(self, _, label):
        '''respond to button click event'''
        if label in self.frame.graph['map']:
            record = self.frame.graph['map'][label]
            graph = record['graph']
            graph['selected'] = record['from']
            self.frame.GetParent().show_frame(graph)
        return (False, '')


class OutputProperty(wx.propgrid.LongStringProperty):
    '''output property to allow for trace output'''

    def __init__(self, frame, label=propgrid.PG_LABEL, name=propgrid.PG_LABEL, value=''):
        super(OutputProperty, self).__init__(label, name, value)
        self.frame = frame
        self.name = name
        self.value = value

    def DisplayEditorDialog(self, _, __):
        '''respond to button click event'''
        if self.name in self.frame.graph['map']:
            graph = self.frame.graph['map'][self.name]['to'][self.value]
            graph['selected'] = self.value
            self.frame.GetParent().show_frame(graph)
        return (False, '')


class ChildFrame(wx.MDIChildFrame):
    '''child frame to show a graph'''

    def __init__(self, parent, title, graph):
        super(ChildFrame, self).__init__(parent, title=title, size=graph['size'])
        self.graph = graph
        self.pen_color = wx.Colour('black')
        self.foreground_color = wx.Colour('gray')
        self.background_color = wx.Colour('white')
        self.dictionary = {}
        self.get_dict(self.graph, self.dictionary)
        icon = wx.Icon()
        if self.graph['type'] == 'onnx':
            icon.CopyFromBitmap(wx.Bitmap(os.path.join(pwd(), 'icons', 'onnx.png'),
                                          wx.BITMAP_TYPE_ANY))
        elif self.graph['type'] == 'tensorflow':
            icon.CopyFromBitmap(wx.Bitmap(os.path.join(pwd(), 'icons', 'tf.png'),
                                          wx.BITMAP_TYPE_ANY))
        else: icon.CopyFromBitmap(wx.Bitmap(os.path.join(pwd(), 'icons', 'model.png'),
                                            wx.BITMAP_TYPE_ANY))
        self.SetIcon(icon)
        self.canvas = wx.ScrolledCanvas(self, True)
        self.property = propgrid.PropertyGrid(self, size=(300, 500))
        self.box = wx.BoxSizer(wx.HORIZONTAL)
        self.box.Add(self.canvas, wx.ID_ANY, flag=wx.EXPAND|wx.ALL|wx.LEFT, border=1)
        self.box.Add(self.property, 0, flag=wx.EXPAND|wx.RIGHT, border=1)
        self.SetSizer(self.box)
        self.dc = None
        self.canvas.Bind(wx.EVT_PAINT, self.on_paint)
        self.canvas.Bind(wx.EVT_LEFT_DOWN, self.on_left_down)
        self.thumbnail = wx.Panel(self.canvas, style=wx.SIMPLE_BORDER)
        self.thumbnail.Bind(wx.EVT_PAINT, self.on_paint_thumb)
        self.thumbnail.Bind(wx.EVT_LEFT_DOWN, self.on_key_down)
        self.thumb_dc = None
        self.thumb_ratio = 20
        self.thumb_max_len = 300
        self.Maximize()
        self.Show(True)
        canvas_size = self.canvas.GetSize()
        self.x_units = canvas_size[0]/2
        self.y_units = canvas_size[1]/2
        x_steps = math.ceil(float(self.graph['size'][0])/self.x_units)
        y_steps = math.ceil(float(self.graph['size'][1])/self.y_units)
        self.canvas.SetScrollbars(self.x_units, self.y_units, x_steps, y_steps, 0, 0, True)
        self.GetParent().add_history([self.GetId(), self.graph['name'], self.graph['selected']])

    def get_dict(self, graph, dictionary):
        '''get keywors of graph for search'''

        def get_prefix(path):
            ps = path.split('/')
            paths = []
            for i in range(len(ps)):
                paths.append('/'.join(ps[i:]))
            return paths

        for v in graph['vertices']:
            vertice = graph['vertices'][v]
            for path in get_prefix(v):
                if path not in dictionary:
                    dictionary[path] = [v, graph]
            for o in vertice['outputs']:
                for path in get_prefix(o):
                    if path not in dictionary:
                        dictionary[path] = [v, graph]

        for sg in graph['subgraphs']:
            self.get_dict(graph['subgraphs'][sg], dictionary)

    def cacl_thumb_size(self):
        '''calculate size of thumbnail'''
        canvas_size = self.graph['size']
        target_size = (math.ceil(float(canvas_size[0])/self.thumb_ratio),
                       math.ceil(float(canvas_size[1])/self.thumb_ratio))

        if target_size[0] <= self.thumb_max_len and target_size[1] <= self.thumb_max_len:
            return target_size

        width, height = target_size[0], target_size[1]
        width_height_ratio = float(canvas_size[0]) / canvas_size[1]
        if width > self.thumb_max_len:
            width = self.thumb_max_len
            height = width / width_height_ratio

        if height > self.thumb_max_len:
            height = self.thumb_max_len
            width = height * width_height_ratio

        self.thumb_ratio = math.ceil(float(canvas_size[0])/width)
        return (math.ceil(width), math.ceil(height))

    def get_canvas_view(self):
        '''get rect of the view'''
        xy_strt = self.canvas.GetViewStart()
        xy_unit = self.canvas.GetScrollPixelsPerUnit()
        xy_size = self.canvas.GetSize()
        x_start = xy_strt[0] * xy_unit[0]
        y_start = xy_strt[1] * xy_unit[1]
        return (x_start, y_start, xy_size[0], xy_size[1])

    def on_key_down(self, e):
        '''handle left mouse key down event on thumbnail'''
        pos = e.GetLogicalPosition(self.thumb_dc)
        target_pos = (pos[0] * self.thumb_ratio, pos[1] * self.thumb_ratio)
        target_rect = self.get_canvas_view()
        if not ChildFrame.include(target_rect, target_pos):
            self.canvas.Scroll(target_pos[0]/self.x_units, target_pos[1]/self.y_units)

    def on_paint_thumb(self, _):
        '''paint thumbnail'''
        dc = wx.PaintDC(self.thumbnail)
        dc.Clear()
        self.thumb_dc = dc
        points = []
        for v in self.graph['vertices']:
            rect = self.graph['vertices'][v]['rect']
            points.append((int(rect[0]/self.thumb_ratio), int(rect[1]/self.thumb_ratio)))
        dc.DrawPointList(points, wx.Pen(self.background_color, 20))
        target_rect = self.get_canvas_view()
        thumb_rect = (math.floor(float(target_rect[0])/self.thumb_ratio),
                      math.floor(float(target_rect[1])/self.thumb_ratio),
                      math.ceil(float(target_rect[2])/self.thumb_ratio),
                      math.ceil(float(target_rect[3])/self.thumb_ratio))
        dc.SetPen(wx.Pen('red', 1))
        dc.SetBrush(wx.Brush('red', wx.BRUSHSTYLE_TRANSPARENT))
        dc.DrawRectangle(thumb_rect)
        rect = self.graph['vertices'][self.graph['selected']]['rect']
        dc.DrawCircle(Point((int(rect[0]/self.thumb_ratio), int(rect[1]/self.thumb_ratio))), 2)

    def get_subgraph(self, graph, name):
        '''get embedded graph'''
        if name in graph['subgraphs']:
            return graph['subgraphs'][name]
        for sg in graph['subgraphs']:
            ret = self.get_subgraph(graph['subgraphs'][sg], name)
            if ret is not None:
                return ret
        raise RuntimeError('Subgraph ', name, 'does not exist.')

    def on_click(self, e):
        '''open subgraph'''
        bid = e.GetId()
        name = self.op_info[bid].GetLabel().split('"')[-2]
        ChildFrame(self.GetParent(), name, self.graph['subgraphs'][name])

    @staticmethod
    def include(rect, pos):
        '''check if pos is in rect'''
        x, y = pos[0], pos[1]
        return x > rect[0] and\
               x < rect[0] + rect[2] and\
               y > rect[1] and\
               y < rect[1] + rect[3]

    @staticmethod
    def corners(rect):
        '''return four corners of a rect'''
        left = rect[0]
        top = rect[1]
        right = rect[0] + rect[2]
        btm = rect[1] + rect[3]
        return (left, top), (left, btm), (right, top), (right, btm)

    def on_left_down(self, event):
        '''handle left mouse key down event on main view'''
        pos = event.GetLogicalPosition(self.dc)
        need_refresh = False

        for vertice in self.graph['vertices']:
            if ChildFrame.include(self.graph['vertices'][vertice]['rect'], pos) and\
               vertice != self.graph['selected']:
                self.graph['selected'] = vertice
                self.GetParent().add_history([self.GetId(),
                                              self.graph['name'],
                                              self.graph['selected']])
                need_refresh = True
                break

        if need_refresh:
            self.canvas.Refresh()

    @staticmethod
    def intersect(rect_0, rect_1):
        '''check if two rects intersect'''
        lefttop = (rect_1[0], rect_1[1])
        leftbtm = (rect_1[0], rect_1[1] + rect_1[3])
        rihttop = (rect_1[0] + rect_1[2], rect_1[1])
        rihtbtm = (rect_1[0] + rect_1[2], rect_1[1] + rect_1[3])
        return ChildFrame.include(rect_0, lefttop) or ChildFrame.include(rect_0, leftbtm) or\
               ChildFrame.include(rect_0, rihttop) or ChildFrame.include(rect_0, rihtbtm)

    def adjust_canvas(self):
        '''move current view'''
        xy_strt = self.canvas.GetViewStart()
        xy_unit = self.canvas.GetScrollPixelsPerUnit()
        xy_size = self.canvas.GetSize()
        x_start = xy_strt[0] * xy_unit[0]
        y_start = xy_strt[1] * xy_unit[1]
        target_rect = (x_start, y_start, xy_size[0], xy_size[1])
        vertice = self.graph['vertices'][self.graph['selected']]

        if not ChildFrame.intersect(target_rect, vertice['rect']):
            self.canvas.Scroll(vertice['rect'][0]/self.x_units, vertice['rect'][1]/self.y_units)

    def set_property(self):
        '''set property panel'''
        v = self.graph['selected']
        name_property = self.property.GetPropertyByLabel('name')
        if name_property is not None and name_property.GetValue() == v:
            return
        self.property.Clear()
        vertice = self.graph['vertices'][v]
        self.property.Append(propgrid.PropertyCategory('Vertice Property',
                                                       'Vertice Property'))
        self.property.Append(propgrid.StringProperty('Name', 'Name', v))
        self.property.Append(propgrid.StringProperty('Type', 'Type', vertice['type']))
        self.property.Append(propgrid.PropertyCategory('Vertice inputs',
                                                       'Vertice inputs'))

        for i, n in enumerate(vertice['inputs']):
            input_name = 'input ' + str(i+1)
            prop = InputProperty(self, input_name, input_name, n)
            self.property.Append(prop)
            if n in self.graph['shapes']:
                subprop = propgrid.StringProperty('Shape', 'Shape', str(self.graph['shapes'][n]))
                self.property.AppendIn(propgrid.PGPropArgCls(prop), subprop)
            if n in self.graph['types']:
                subprop = propgrid.StringProperty('Type', 'Type', str(self.graph['types'][n]))
                self.property.AppendIn(propgrid.PGPropArgCls(prop), subprop)

        self.property.Append(propgrid.PropertyCategory("Vertice Outputs", "Vertice Outputs"))
        for i, n in enumerate(vertice['outputs']):
            output_name = 'Output ' + str(i+1)
            prop = propgrid.StringProperty(output_name, output_name, n)
            self.property.Append(prop)
            if n in self.graph['shapes']:
                subprop = propgrid.StringProperty('Shape', 'Shape', str(self.graph['shapes'][n]))
                self.property.AppendIn(propgrid.PGPropArgCls(prop), subprop)
            if n in self.graph['types']:
                subprop = propgrid.StringProperty('Type', 'Type', str(self.graph['types'][n]))
                self.property.AppendIn(propgrid.PGPropArgCls(prop), subprop)
            if n in self.graph['map']:
                for ii, to in enumerate(self.graph['map'][n]['to']):
                    consumer = 'Consumer ' + str(ii)
                    subprop = OutputProperty(self, consumer, n, to)
                    self.property.AppendIn(propgrid.PGPropArgCls(prop), subprop)

        self.property.Append(propgrid.PropertyCategory("Vertice Attrs", "Vertice Attrs"))
        for i, n in enumerate(vertice['attrs']):
            attr = vertice['attrs'][n]
            t = attr['type']
            if t == 'string':
                if len(attr['value']) < 30:
                    self.property.Append(propgrid.StringProperty(n, n, attr['value']))
                else:
                    self.property.Append(propgrid.LongStringProperty(n, n, attr['value']))
            elif t == 'subgraph':
                prop = SubgraphProperty(self, n, attr['value'], attr['value'])
                self.property.Append(prop)
            elif t in ['tensor', 'sparse_tensor']:
                shape_prop = propgrid.StringProperty('Shape', 'Shape', str(attr['value'].shape))
                self.property.Append(shape_prop)
                data_prop = propgrid.LongStringProperty('Data', 'Data', str(attr['value']))
                self.property.Append(data_prop)
        self.property.FitColumns()

    def on_paint(self, _):
        '''paint main view'''
        canvas_size = self.canvas.GetSize()
        dc = wx.PaintDC(self.canvas)
        dc.SetBackground(wx.Brush(self.background_color))
        self.dc = dc
        dc.Clear()
        dc.SetBrush(wx.Brush(self.foreground_color))
        dc.SetPen(wx.Pen(self.pen_color))
        dc.SetTextForeground(self.background_color)
        font = wx.Font(10, wx.FONTFAMILY_ROMAN, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, False)
        dc.SetFont(font)
        self.canvas.PrepareDC(dc)

        graph_size = self.graph['size']

        if canvas_size[0] < graph_size[0] and canvas_size[1] < graph_size[1]:
            self.thumbnail.SetPosition((0, 0))
            self.thumbnail.SetSize(self.cacl_thumb_size())
            self.thumbnail.SetBackgroundColour(self.foreground_color)
        else:
            self.thumbnail.Show(False)

        view = self.get_canvas_view()
        for edge in self.graph['edges']:
            spline = self.graph['edges'][edge]['spline']
            if ChildFrame.include(view, spline[0]) or ChildFrame.include(view, spline[-1]):
                dc.DrawSpline(spline)
                dc.DrawPolygon(self.graph['edges'][edge]['arrow'])

        for vertice in self.graph['vertices']:
            vertice_rect = self.graph['vertices'][vertice]['rect']
            corners = ChildFrame.corners(vertice_rect)
            for corner in corners:
                if ChildFrame.include(view, corner):
                    dc.DrawRoundedRectangle(self.graph['vertices'][vertice]['rect'], 3)
                    dc.DrawText(self.graph['vertices'][vertice]['type'],
                                to_int(self.graph['vertices'][vertice]['label']))
                    break

        vertice = self.graph['vertices'][self.graph['selected']]
        dc.SetPen(wx.Pen("red", 2))
        dc.DrawRoundedRectangle(vertice['rect'], 3)
        dc.DrawText(vertice['type'], to_int(vertice['label']))

        for edge in vertice['edges']:
            if edge in self.graph['edges']:
                dc.DrawSpline(self.graph['edges'][edge]['spline'])
                dc.DrawPolygon(self.graph['edges'][edge]['arrow'])

        self.set_property()
        self.thumbnail.Refresh()

    def select(self, key):
        '''highlight vertice on search'''
        if key in self.dictionary:
            v, sg = self.dictionary[key]
            if sg == self.graph['name']:
                need_refresh = False
                if self.graph['selected'] != v:
                    self.graph['selected'] = v
                    need_refresh = True
                if need_refresh:
                    self.canvas.Refresh()
                    self.adjust_canvas()
            else:
                sg['selected'] = v
                self.GetParent().show_frame(sg)


class MainFrame(wx.MDIParentFrame):
    '''main frame'''

    def __init__(self, parent, title):
        super(MainFrame, self).__init__(parent,
                                        title=title,
                                        size=(500, 300),
                                        style=wx.DEFAULT_FRAME_STYLE|wx.FRAME_NO_WINDOW_MENU)
        icon = wx.Icon()
        icon.CopyFromBitmap(wx.Bitmap(os.path.join(pwd(), 'icons', 'tracer.30.png'),
                                      wx.BITMAP_TYPE_ANY))
        self.SetIcon(icon)
        self.SetTitle('tracer')
        self.dc = None
        self.draw = False
        self.MenuBar = wx.MenuBar()
        self.File = wx.Menu()
        self.File.Append(wx.ID_OPEN, "Open model...", "Open a model")
        self.File.Append(wx.ID_EXIT, "Exit", "Exit tracer")
        self.MenuBar.Append(self.File, 'File')
        self.About = wx.Menu()
        self.About.Append(wx.ID_ABOUT, "About tracer...")
        self.MenuBar.Append(self.About, 'About')
        self.MenuBar.Bind(wx.EVT_MENU, self.on_order)
        self.ToolBar = wx.ToolBar(self, -1)
        self.ToolBar.AddTool(1, 'back',
                             wx.Bitmap(os.path.join(pwd(), 'icons', 'back.png')),
                             'back')
        self.ToolBar.AddTool(2, 'forward',
                             wx.Bitmap(os.path.join(pwd(), 'icons', 'forward.png')),
                             'forward')
        self.ToolBar.AddTool(3, 'darkdrop',
                             wx.Bitmap(os.path.join(pwd(), 'icons', 'blackdrop.png')),
                             'dark backdrop')
        self.ToolBar.AddTool(4, 'lightdrop',
                             wx.Bitmap(os.path.join(pwd(), 'icons', 'whitedrop.png')),
                             'light backdrop')
        self.SetToolBar(self.ToolBar)
        self.ToolBar.Realize()
        self.ToolBar.Bind(wx.EVT_TOOL, self.on_move)
        self.Maximize(True)
        self.Search = wx.SearchCtrl(self.ToolBar,
                                    pos=(self.GetSize()[0]-260, 7),
                                    size=(250, 23))
        self.Search.ShowCancelButton(True)
        self.Search.Bind(wx.EVT_SET_FOCUS, self.prepare_search)
        self.Search.Bind(wx.EVT_SEARCHCTRL_SEARCH_BTN, self.on_search)
        self.Bind(wx.EVT_CLOSE, MainFrame.on_close)
        self.search_txtctrl = self.Search.FindWindowByName('text')
        self.history = []
        self.history_at = -1
        self.Show(True)
        create_temp()

    @staticmethod
    def on_close(e):
        '''clear up on close'''
        remove_temp()
        e.Skip()

    def prepare_search(self, _):
        '''load all keywords for search'''
        frame = self.GetActiveChild()
        if frame is None:
            return
        self.Search.AutoComplete(list(frame.dictionary.keys()))

    def on_search(self, _):
        '''respond to search event'''
        frame = self.GetActiveChild()
        if frame is None:
            return
        keyword = self.search_txtctrl.GetValue()
        if keyword in frame.dictionary:
            frame.select(self.search_txtctrl.GetValue())
        else: wx.MessageDialog(self,
                               keyword + ' not in current graph or its subgraphs.').ShowModal()

    def on_order(self, event):
        '''handle menu event'''
        mid = event.GetId()

        if mid == wx.ID_EXIT:
            sys.exit()

        elif mid == wx.ID_OPEN:
            dialog = wx.FileDialog(self)
            if dialog.ShowModal() == wx.ID_OK:
                self.open(dialog.GetPath())

        elif mid == wx.ID_ABOUT:
            About(self)

    def show_frame(self, graph):
        '''show child frame'''
        for record in self.history:
            if record[1] == graph['name']:
                frame = self.FindWindowById(record[0])
                if frame is None:
                    continue
                frame.Maximize()
                frame.Refresh()
                frame.adjust_canvas()
                return
        ChildFrame(self, graph['name'], graph).adjust_canvas()

    def open(self, model_path):
        '''open a child graph to show model'''
        progress = wx.ProgressDialog("Progress", "Reading model ...", parent=self,\
                                     style=wx.PD_SMOOTH|wx.PD_AUTO_HIDE|wx.PD_CAN_ABORT)

        def init_progress_func(total):
            progress.SetRange(total)

        def update_progress_func(done):
            return progress.Update(done, str(done) + " nodes analyzed ...")

        cancelled = True
        try:
            graph = parse(model_path, init_progress_func, update_progress_func)
            cancelled = progress.WasCancelled()
        except graphviz.backend.ExecutableNotFound:
            wx.MessageDialog(self, 'Please install latest graphviz from \
                                    www.graphviz.org and add it to PATH').ShowModal()
        except TypeError as err:
            wx.MessageDialog(self, 'Model format not yet support').ShowModal()
        except Exception as err:
            wx.MessageDialog(self, 'Failed to open model due to: ' + str(err)).ShowModal()

        progress.Destroy()
        if cancelled is False:
            self.show_frame(graph)

    def add_history(self, record):
        '''keep track of highlighted vertices'''
        self.history.append(record)
        self.history_at = len(self.history) -1

    def on_move(self, event):
        '''handle toolbar event'''
        tid = event.GetId()

        if tid in [1, 2]:
            while True:
                if tid == 1:
                    if self.history_at == 0:
                        break
                    self.history_at -= 1
                elif tid == 2:
                    if self.history_at == len(self.history) - 1:
                        break
                    self.history_at += 1
                record = self.history[self.history_at]
                frame = self.FindWindowById(record[0])
                if frame is None:
                    del self.history[self.history_at]
                else:
                    frame.graph['selected'] = record[2]
                    frame.Maximize()
                    frame.canvas.Refresh()
                    frame.adjust_canvas()
                    break

        elif tid == 3:
            frame = self.GetActiveChild()
            if frame is None:
                return
            frame.pen_color = wx.Colour('gray')
            frame.foreground_color = wx.Colour('white')
            frame.background_color = wx.Colour(86, 86, 87)
            frame.Refresh()

        elif tid == 4:
            frame = self.GetActiveChild()
            if frame is None:
                return
            frame.pen_color = wx.Colour('black')
            frame.foreground_color = wx.Colour('gray')
            frame.background_color = wx.Colour('white')
            frame.Refresh()


def show():
    '''run tracer'''
    ex = wx.App()
    MainFrame(None, 'Tracer')
    ex.MainLoop()
