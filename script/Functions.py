import numpy as np

def _add(sons):
    val = np.add(sons[0],sons[1])
    return val
#     return normalize([val])[0]

def _minus(sons):
    val = np.subtract(sons[0],sons[1])
    return val
#     return normalize([val])[0]


def _multiply(sons):
    val = np.multiply(sons[0],sons[1])
    val = np.where((val < 26843545) & (val > -26843545),val,26843545)
    if np.any(np.isnan(val)):
        print("muti")
        print("exist nan",)
    if np.any(np.isinf(val)):
        print("muti")
        print("exist inf",)
    return val
#     return normalize([val])[0]
    
def _divide(sons):
    val = np.divide(sons[0], sons[1],where=sons[1]!=0,out=np.zeros(len(sons[1])))
    val = np.where((val < 26843545) & (val > -26843545),val,26843545)
    if np.any(np.isnan(val)):
        print("div")
        print("exist nan",)
    if np.any(np.isinf(val)):
        print("div")
        print("exist inf",)
    
    return val
#     return normalize([val])[0]

simple_opset = {
    _add:2,
    _minus:2,
    _multiply:2,
    _divide:2
        }
