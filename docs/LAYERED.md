# Layered Rendering
Some graphs, like resnet, has over ten thousand nodes which takes quite a bit of time to load and render. To address the situation, tracer breaks down a large graph into several set of nodes and render the sets as a top level graph:

<img src="https://github.com/RandySheriffH/tracer/blob/master/snaps/Layered.PNG" width=650 height=400>

Viewers can open selected set to view the part of graph:

<img src="https://github.com/RandySheriffH/tracer/blob/master/snaps/OpenGraphPart.PNG" width=550 height=200>
