import os
import shutil

def INT(array):
    return [int(a) for a in array]

def CreateTemp():
    temp = GetTemp()
    if not os.path.isdir(temp):
        os.mkdir(temp)

def GetTemp():
    return './temp/'

def RemoveTemp():
    shutil.rmtree(GetTemp(), ignore_errors=True)

def PWD():
    return os.path.dirname(__file__) + '/'