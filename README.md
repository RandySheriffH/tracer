# <img src="https://github.com/RandySheriffH/tracer/blob/master/snaps/Tracer.jpg" width=30 height=25> Tracer
Tracer is a desktop app to render deep learning models:

- Keep history of browsing for easy trace back
- Support viewing of embedded graphs 
- Trace upstream and downstream
- Search node by keyword

## Install

- git clone the project
- install graphviz from graphviz.org and add its bin to PATH
- pip install -r requirements.txt
- run "python setup.py install"
- open python terminal and run:
```
    from tracer import tracer
    tracer.show()
```


## Trace back and forth
Tracer keeps history of highlighted nodes, user could rely on buttons from toolbar to go back and forth:\
<img src="https://github.com/RandySheriffH/tracer/blob/master/snaps/BackForth.PNG" width=300 height=120>

## Display embedded graphs
In property panel, select corresponding attribute, click on "..." button, embedded graph will be display in a new frame:\
<img src="https://github.com/RandySheriffH/tracer/blob/master/snaps/OpenEmbedded.PNG" width=370 height=120>

## Trace upstream and downstream
In property panel, select the input and click on "..." button to go to upstream node, for output it's similar:\
<img src="https://github.com/RandySheriffH/tracer/blob/master/snaps/Upstream.PNG" width=370 height=120>

## Search
Just type keyword in search input box on right-top corner, select matching item from drop-down and click on search button:\
<img src="https://github.com/RandySheriffH/tracer/blob/master/snaps/Search.PNG" width=300 height=120>

## Limitations
For now tracer only support viewing of tensorflow and onnx models. Contributions are highly welcomed - just add a new parser in parsers.py following the comments.
