# Licensed under the MIT license.
'''utilities'''

import os
import shutil


def to_int(array):
    '''convert array to ints'''
    return [int(a) for a in array]


def create_temp():
    '''create temp folder'''
    temp = get_temp()
    if not os.path.isdir(temp):
        os.mkdir(temp)


def get_temp():
    '''temp string'''
    return './temp/'


def remove_temp():
    '''remove temp folder'''
    shutil.rmtree(get_temp(), ignore_errors=True)


def pwd():
    '''return path to the file'''
    return os.path.dirname(__file__)


class UnknownFormatError(RuntimeError):
    '''raise on unsupported model format'''
    def __init__(self, message):
        RuntimeError.__init__(self, message)

def to_array(s, to_type):
    '''string to array'''
    s = s.strip()
    stk = []
    digits = [str(i) for i in range(10)] + ['e', '+', '-', '.']
    number = ''
    for c in s:
        #c = c.strip()
        if c == '': continue
        elif c in digits: number += c
        elif c == ',' or c == ' ':
            if number != '':
                stk.append(number)
                number = ''
        elif c == ']':
            if number != '':
                stk.append(number)
                number = ''
            a = []
            while True:
                top = stk[-1]
                del stk[-1]
                if top == '[': 
                    stk.append(a)
                    break
                else:
                    if type(top) is list:
                        a.insert(0, top)
                    else: a.insert(0, to_type(top))
        elif c == '[':
            stk.append(c)
        else: raise Exception("invalid char: " + c)
    return stk[0]