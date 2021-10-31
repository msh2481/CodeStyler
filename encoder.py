from base64 import *

text = open('models/5099.p', 'rb').read()
code = b64encode(text)
print(code)