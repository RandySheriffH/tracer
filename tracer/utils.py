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
