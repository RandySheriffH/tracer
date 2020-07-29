# <img src="https://github.com/RandySheriffH/tracer/blob/master/snaps/Tracer.jpg" width=30 height=25> Tracer
Tracer is a desktop app to render deep learning models:

- keep browsing history for easy trace back
- support viewing of embedded graphs 
- [show large graphs by layers](https://github.com/RandySheriffH/tracer/blob/master/docs/LAYERED.md)
- track upstream and downstream
- search node by keyword

## Install

- install latest [graphviz](http://graphviz.org) and add its bin to PATH
- git clone the tracer project
- pip install -r requirements.txt
- pip install tensorflow==2.x if need to view tensorflow models
- pip install onnx if need to view onnx models
- run "python setup.py install"
- open python terminal and run:
```
    from tracer import tracer
    tracer.show()
```

## Trace back and forth
Tracer keeps history of highlighted nodes, user could rely on buttons from toolbar to go back and forth:

<img src="https://github.com/RandySheriffH/tracer/blob/master/snaps/BackForth.PNG" width=300 height=120>

## View embedded graphs
In property panel, select corresponding attribute, click on "..." button, embedded graph will be rendered in a new frame:

<img src="https://github.com/RandySheriffH/tracer/blob/master/snaps/OpenEmbedded.PNG" width=370 height=120>

## Track upstream and downstream
In property panel, select the input and click on "..." button to go to upstream node, same for output:

<img src="https://github.com/RandySheriffH/tracer/blob/master/snaps/Upstream.PNG" width=370 height=120>

## Search
Just type keyword in search input box on right-top corner, select matching item from drop-down and click on search button:

<img src="https://github.com/RandySheriffH/tracer/blob/master/snaps/Search.PNG" width=300 height=120>

Note that searching happens recursively - nodes of embedded graph will all be included.

## Limitations
By far only support tensorflow and onnx models.\
Welcome to [contribute]().
