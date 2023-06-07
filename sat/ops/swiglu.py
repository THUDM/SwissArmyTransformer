from xformers.ops import swiglu as swiglu_raw

def swiglu(x, w1, w2, w3, b1=None, b2=None, b3=None):
    return swiglu_raw(x, w1, b1, w2, b2, w3, b3)