# Licensed under the MIT license.
'''Render a model graph'''
#pylint: disable=no-member,import-outside-toplevel,too-many-locals,too-many-branches,too-many-statements,too-many-return-statements,protected-access,anomalous-backslash-in-string

from parsers import parse

def render(graph):
    ''' render graph in ws panel'''
    print (graph)

def init_callback(n):
    pass

def update_callback(n):
    return True, True

if __name__ == '__main__':
    render(parse('C:\\Users\\rashuai\\OneDrive\\TheWork\\models\\cond.pb', init_callback, update_callback, False))