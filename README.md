# Tracer
Tracer is a desktop app to render deep learning models:

- Keeps history of browsing for easy trace back
- Support viewing of embedded graphs 
- Trace upstream and downstream
- Search node by keyword

## Usage

- git clone the project
- run "python setup.py install"
- open python terminal and run:
    - from tracer import tracer
    - tracer.show()


## Trace back and forth
Tracer keeps history of highlighted nodes, user could rely on buttons from toolbar to go back and forth:

## Display embedded graphs
In property panel, select corresponding attribute, click on "..." button, embedded graph will be display in a new frame:

## Trace upstream and downstream
In property panel, select the input and click on "..." button to go to upstream node, for output it's similar:

## Search
Just type keyword in search input box on right-top corner, select matching item from drop-down and click on search button:

## Limitations
For now tracer only support viewing of tensorflow and onnx models. Contributions are highly welcomed - just add a new parser in parsers.py following the comments.
