import os, wx, sys, math
from wx import Point, Size, propgrid
from parsers import Parse
from utils import *
import graphviz

class About(wx.Dialog):
    def __init__(self, parent):
        super(About, self).__init__(parent, size = (360, 120))
        self.info = wx.StaticText(self, pos=Point(80, 20))
        self.info.SetLabel("Tracer by Ran Shuai, MIT license")
        self.mail = wx.StaticText(self, pos=Point(90, 40))
        self.mail.SetLabel("randysheriff@hotmail.com")
        self.Center()
        self.Show()


class MyProperty(wx.propgrid.LongStringProperty):

    def __init__(self, frame, label=propgrid.PG_LABEL, name=propgrid.PG_LABEL, value=''):
        super(MyProperty, self).__init__(label, name, value)
        self.frame = frame

    def DisplayEditorDialog(self, prop, label):
        self.frame.OpenChildFrame(label).AdjustCanvas()
        return (False, '')


selected = []
selected_at = -1


class ChildFrame(wx.MDIChildFrame):

    def __init__(self, parent, title, graph):
        super(ChildFrame, self).__init__(parent, title = title, size = graph['size']) #, style=wx.DEFAULT_FRAME_STYLE|wx.ICON_NONE)
        self.graph = graph
        self.foreground_color = wx.Colour('gray')
        self.background_color = wx.Colour('white')
        self.dict = {}
        self.GetDict(self.graph, self.dict)
        self.InitUI()

    def GetDict(self, graph, dict):

        def GetPrefix(path):
            ps = path.split('/')
            paths = []
            for i in range(len(ps)):
                paths.append('/'.join(ps[i:]))
            return paths

        for v in graph['vertices']:
            vertice = graph['vertices'][v]
            for path in GetPrefix(v):
                if path not in dict: dict[path] = [v, graph['name']]
            for o in vertice['outputs']:
                for path in GetPrefix(o):
                    if path not in dict: dict[path] = [v, graph['name']]

        for sg in graph['subgraphs']:
            self.GetDict(graph['subgraphs'][sg], dict)

    def InitUI(self):
        icon = wx.Icon()
        if self.graph['type'] == 'onnx':
            icon.CopyFromBitmap(wx.Bitmap("icons/onnx.png", wx.BITMAP_TYPE_ANY))
        elif self.graph['type'] == 'tensorflow':
            icon.CopyFromBitmap(wx.Bitmap("icons/tf.png", wx.BITMAP_TYPE_ANY))
        else: icon.CopyFromBitmap(wx.Bitmap("icons/model.png", wx.BITMAP_TYPE_ANY))
        self.SetIcon(icon)
        self.canvas = wx.ScrolledCanvas(self, True)
        self.property = propgrid.PropertyGrid(self, size=(300,500))
        self.box = wx.BoxSizer(wx.HORIZONTAL)
        self.box.Add(self.canvas, wx.ID_ANY, flag=wx.EXPAND|wx.ALL|wx.LEFT, border=1)
        self.box.Add(self.property, 0, flag=wx.EXPAND|wx.RIGHT, border=1)
        self.SetSizer(self.box)
        self.selected = [self.graph['selected']]
        self.selected_at = 0
        self.dc = None
        self.canvas.Bind(wx.EVT_PAINT, self.OnPaint)
        self.canvas.Bind(wx.EVT_LEFT_DOWN, self.OnLeftDown)
        self.Bind(wx.EVT_CLOSE, self.OnClose)
        self.Bind(wx.EVT_SET_FOCUS, self.OnFocus)
        self.ThumbPanel = wx.Panel(self.canvas, style=wx.SIMPLE_BORDER)
        self.ThumbPanel.Bind(wx.EVT_PAINT, self.OnPaintThumb)
        self.ThumbPanel.Bind(wx.EVT_LEFT_DOWN, self.OnKeyDown)
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

    def CalcThumbSize(self):
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

    def GetCanvasView(self):
        xy_strt = self.canvas.GetViewStart()
        xy_unit = self.canvas.GetScrollPixelsPerUnit()
        xy_size = self.canvas.GetSize()
        x_start = xy_strt[0] * xy_unit[0]
        y_start = xy_strt[1] * xy_unit[1]
        return (x_start, y_start, xy_size[0], xy_size[1])

    def OnKeyDown(self, e):
        pos = e.GetLogicalPosition(self.thumb_dc)
        target_pos = (pos[0] * self.thumb_ratio, pos[1] * self.thumb_ratio)
        target_rect = self.GetCanvasView()
        if not self.In(target_rect, target_pos):
            self.canvas.Scroll(target_pos[0]/self.x_units, target_pos[1]/self.y_units)

    def OnPaintThumb(self, e):
        dc = wx.PaintDC(self.ThumbPanel)
        dc.Clear()
        self.thumb_dc = dc
        points = []
        for v in self.graph['vertices']:
            rect = self.graph['vertices'][v]['rect']
            points.append((int(rect[0]/self.thumb_ratio), int(rect[1]/self.thumb_ratio)))
        dc.DrawPointList(points, wx.Pen(self.background_color, 20))
        target_rect = self.GetCanvasView()
        thumb_rect = (math.floor(float(target_rect[0])/self.thumb_ratio),
                      math.floor(float(target_rect[1])/self.thumb_ratio),
                      math.ceil(float(target_rect[2])/self.thumb_ratio),
                      math.ceil(float(target_rect[3])/self.thumb_ratio))
        dc.SetPen(wx.Pen('red', 1))
        dc.SetBrush(wx.Brush('red', wx.BRUSHSTYLE_TRANSPARENT))
        dc.DrawRectangle(thumb_rect)
        rect = self.graph['vertices'][self.selected[self.selected_at]]['rect']
        dc.DrawCircle(Point((int(rect[0]/self.thumb_ratio), int(rect[1]/self.thumb_ratio))), 2)

    def OpenChildFrame(self, name, selected = ''):

        def GetSubgraph(graph, name):
            if name in graph['subgraphs']: return graph['subgraphs'][name]
            for sg in graph['subgraphs']:
                ret = GetSubgraph(graph['subgraphs'][sg], name)
                if ret is not None: return ret
            raise RuntimeError('Subgraph ', name, 'does not exist.')

        target_graph = GetSubgraph(self.graph, name)
        if selected != '': target_graph['selected'] = selected
        return ChildFrame(self.GetParent(), name, target_graph)

    def OnClick(self, e):
        bid = e.GetId()
        name = self.op_info[bid].GetLabel().split('"')[-2]
        ChildFrame(self.GetParent(), name, self.graph['subgraphs'][name])

    def In(self, rect, pos):
        x, y = pos[0], pos[1]
        return x > rect[0] and x < rect[0] + rect[2] and y > rect[1] and y < rect[1] + rect[3]

    def OnLeftDown(self, event):
        pos = event.GetLogicalPosition(self.dc)
        need_refresh = False

        for vertice in self.graph['vertices']:
            if self.In(self.graph['vertices'][vertice]['rect'], pos):
                self.selected.append(vertice)
                self.selected_at = len(self.selected) - 1
                while len(self.selected) >= 100:
                    del self.selected[0]
                need_refresh = True
                break

        if need_refresh:
            self.canvas.Refresh()

    def Intersect(self, rect_0, rect_1):
        lefttop = (rect_1[0], rect_1[1])
        leftbtm = (rect_1[0], rect_1[1] + rect_1[3])
        rihttop = (rect_1[0] + rect_1[2], rect_1[1])
        rihtbtm = (rect_1[0] + rect_1[2], rect_1[1] + rect_1[3])
        return self.In(rect_0, lefttop) or self.In(rect_0, leftbtm) or\
               self.In(rect_0, rihttop) or self.In(rect_0, rihtbtm)

    def AdjustCanvas(self):
        xy_strt = self.canvas.GetViewStart()
        xy_unit = self.canvas.GetScrollPixelsPerUnit()
        xy_size = self.canvas.GetSize()
        x_start = xy_strt[0] * xy_unit[0]
        y_start = xy_strt[1] * xy_unit[1]
        target_rect = (x_start, y_start, xy_size[0], xy_size[1])
        selected = self.selected[self.selected_at]
        vertice = self.graph['vertices'][selected]

        if not self.Intersect(target_rect, vertice['rect']):
            self.canvas.Scroll(vertice['rect'][0]/self.x_units, vertice['rect'][1]/self.y_units)

    def SetProperty(self):
        v = self.selected[self.selected_at]
        name_property = self.property.GetPropertyByLabel('name')
        if name_property is not None and name_property.GetValue() == v:
            return
        self.property.Clear()
        vertice = self.graph['vertices'][v]
        self.property.Append(propgrid.PropertyCategory("Vertice Property", "Vertice Property"))
        self.property.Append(propgrid.StringProperty('Name', 'Name', v))
        self.property.Append(propgrid.StringProperty('Type', 'Type', vertice['type']))
        self.property.Append(propgrid.PropertyCategory("Vertice Inputs", "Vertice Inputs"))
        for i, n in enumerate(vertice['inputs']):
            input_name = 'Input ' + str(i+1)
            prop = propgrid.StringProperty(input_name, input_name, n)
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
        self.property.Append(propgrid.PropertyCategory("Vertice Attrs", "Vertice Attrs"))
        for i, n in enumerate(vertice['attrs']):
            attr = vertice['attrs'][n]
            t = attr['type']
            if 'string' == t:
                if len(attr['value']) < 30: self.property.Append(propgrid.StringProperty(n, n, attr['value']))
                else: self.property.Append(propgrid.LongStringProperty(n, n, attr['value']))
            elif 'subgraph' == t:
                prop = MyProperty(self, n, attr['value'], attr['value'])
                self.property.Append(prop)
            elif t in ['tensor', 'sparse_tensor']:
                shape_prop = propgrid.StringProperty('Shape', 'Shape', str(attr['value'].shape))
                self.property.Append(shape_prop)
                data_prop = propgrid.LongStringProperty('Data', 'Data', str(attr['value']))
                self.property.Append(data_prop)
        self.property.FitColumns()

    def OnPaint(self, e):
        canvas_size = self.canvas.GetSize()
        dc = wx.PaintDC(self.canvas)
        dc.SetBackground(wx.Brush(self.background_color))
        self.dc = dc
        dc.Clear()
        dc.SetBrush(wx.Brush(self.foreground_color))
        dc.SetPen(wx.Pen(self.foreground_color))
        dc.SetTextForeground(self.background_color)
        dc.SetFont(wx.Font(wx.FontInfo(10)))
        self.canvas.PrepareDC(dc)

        graph_size = self.graph['size']
        
        if canvas_size[0] < graph_size[0] and canvas_size[1] < graph_size[1]:
            self.ThumbPanel.SetPosition((0,0))
            self.ThumbPanel.SetSize(self.CalcThumbSize())
            self.ThumbPanel.SetBackgroundColour(self.foreground_color)
        else:
            self.ThumbPanel.Show(False)

        for edge in self.graph['edges']:
            spline = self.graph['edges'][edge]['spline']
            dc.DrawSpline(spline)
            dc.DrawPolygon(self.graph['edges'][edge]['arrow'])

        for vertice in self.graph['vertices']:
            vertice_rect = self.graph['vertices'][vertice]['rect']
            dc.DrawRoundedRectangle(self.graph['vertices'][vertice]['rect'], 3)
            dc.DrawText(self.graph['vertices'][vertice]['type'],
                        INT(self.graph['vertices'][vertice]['label']))

        selected = self.selected[self.selected_at]
        vertice = self.graph['vertices'][selected]
        dc.SetPen(wx.Pen("red", 2))
        dc.DrawRoundedRectangle(vertice['rect'], 3)
        dc.DrawText(vertice['type'], INT(vertice['label']))

        for edge in vertice['edges']:
            if edge in self.graph['edges']:
                dc.DrawSpline(self.graph['edges'][edge]['spline'])
                dc.DrawPolygon(self.graph['edges'][edge]['arrow'])

        self.SetProperty()
        self.ThumbPanel.Refresh()

    def Select(self, key):
        if key in self.dict:
            v, sg = self.dict[key]
            if sg == self.graph['name']:
                need_refresh = False
                if self.selected[self.selected_at] != v:
                    self.selected.append(v)
                    self.selected_at = len(self.selected) - 1
                    need_refresh = True
                if need_refresh:
                    self.canvas.Refresh()
                    self.AdjustCanvas()
            else:
                frame = self.OpenChildFrame(sg, v)
                frame.canvas.Refresh()
                frame.AdjustCanvas()

    def OnClose(self, event):
        global selected, selected_at
        selected = [f for f in selected if f != self]
        selected_at = len(selected) - 1
        event.Skip()

    def OnFocus(self, event):
        global selected, selected_at
        if selected_at == -1 or selected[selected_at] != self:
            selected.append(self)
            selected_at = len(selected) - 1


class MainFrame(wx.MDIParentFrame):

    def __init__(self, parent, title):
        super(MainFrame, self).__init__(parent, title=title, size=(500,300), style=wx.DEFAULT_FRAME_STYLE|wx.FRAME_NO_WINDOW_MENU )
        self.InitUI()
        CreateTemp()

    def InitUI(self):
        icon = wx.Icon()
        icon.CopyFromBitmap(wx.Bitmap("icons/tracer.30.png", wx.BITMAP_TYPE_ANY))
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
        self.MenuBar.Bind(wx.EVT_MENU, self.OnOrder)
        self.ToolBar = wx.ToolBar(self, -1)
        self.ToolBar.AddTool(1, 'back', wx.Bitmap('icons/back.png'), 'back')
        self.ToolBar.AddTool(2, 'forward', wx.Bitmap('icons/forward.png'), 'forward')
        self.ToolBar.AddTool(3, 'darkdrop', wx.Bitmap('icons/blackdrop.png'), 'dark backdrop')
        self.ToolBar.AddTool(4, 'lightdrop', wx.Bitmap('icons/whitedrop.png'), 'light backdrop')
        self.SetToolBar(self.ToolBar)
        self.ToolBar.Realize()
        self.ToolBar.Bind(wx.EVT_TOOL, self.OnMove)
        self.Maximize(True)
        self.Search = wx.SearchCtrl(self.ToolBar, pos=(self.GetSize()[0]-260,7), size=(250,23))
        self.Search.ShowCancelButton(True)
        self.Search.Bind(wx.EVT_SET_FOCUS, self.PrepareSearch)
        self.Search.Bind(wx.EVT_SEARCHCTRL_SEARCH_BTN, self.OnSearch)
        self.Bind(wx.EVT_CLOSE, self.OnClose)
        self.search_txtctrl = self.Search.FindWindowByName('text')
        self.Show(True)

    def OnClose(self, e):
        RemoveTemp()
        e.Skip()

    def PrepareSearch(self, event):
        frame = self.GetActiveChild()
        if frame is None:
            return
        self.Search.AutoComplete(list(frame.dict.keys()))

    def OnSearch(self, event):
        frame = self.GetActiveChild()
        if frame is None: return
        keyword = self.search_txtctrl.GetValue()
        if keyword in frame.dict:
            frame.Select(self.search_txtctrl.GetValue())
        else: wx.MessageDialog(self, keyword + ' not in current graph or its subgraphs.').ShowModal()

    def OnOrder(self, event):
        mid = event.GetId()

        if mid == wx.ID_EXIT:
            sys.exit()

        elif mid == wx.ID_OPEN:
            dialog = wx.FileDialog(self)
            if dialog.ShowModal() == wx.ID_OK:
                self.Open(dialog.GetPath())

        elif mid == wx.ID_ABOUT:
            About(self)

    def Open(self, model_path):
        progress = wx.ProgressDialog("Progress", "Reading model ...", parent=self,\
                                     style=wx.PD_SMOOTH|wx.PD_AUTO_HIDE|wx.PD_CAN_ABORT)

        def init_progress_func(total):
            progress.SetRange(total)

        def update_progress_func(done):
            return progress.Update(done, str(done) + " nodes analyzed ...")
        try:
            graph = Parse(model_path, init_progress_func, update_progress_func)
            cancelled = progress.WasCancelled()
            progress.Destroy()
            if cancelled is False: ChildFrame(self, graph['name'], graph)
        except graphviz.backend.ExecutableNotFound:
            wx.MessageDialog('Please install graphviz from www.graphviz.org and add it to PATH').ShowModal()

    def OnMove(self, event):
        global selected, selected_at
        frame = self.GetActiveChild()

        if frame is None:
            return
        tid = event.GetId()

        if tid == 1:

            if frame.selected_at > 0:
                frame.selected_at -= 1
                frame.canvas.Refresh()
                frame.AdjustCanvas()

            elif selected_at > 0:
                selected_at -= 1
                if frame != selected[selected_at]:
                    selected[selected_at].Maximize()

        elif tid == 2:

            if frame.selected_at < len(frame.selected) - 1:
                frame.selected_at += 1
                frame.canvas.Refresh()
                frame.AdjustCanvas()

            elif selected_at < len(selected) - 1:
                selected_at += 1
                if frame != selected[selected_at]:
                    selected[selected_at].Maximize()

        elif tid == 3:
            frame.foreground_color = wx.Colour('white')
            frame.background_color = wx.Colour('gray')
            frame.Refresh()

        elif tid == 4:
            frame.foreground_color = wx.Colour('gray')
            frame.background_color = wx.Colour('white')
            frame.Refresh()

def Tracer():
    ex = wx.App()
    MainFrame(None,'Tracer')
    ex.MainLoop()