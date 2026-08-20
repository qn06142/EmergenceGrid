import os, re
s = os.path.abspath('/d/rl-emergence/ckpts/solo_g64c')
print('abspath:', repr(s))
# current fixpath regex
m = re.match(r'^[A-Za-z]:\\d\\(.*)$', s)
print('old-regex match:', bool(m))
# robust: normalize MSYS /d/ -> D:/  and  D:\d\ -> D:\
def fix(p):
    p = os.path.abspath(p)
    # MSYS form /d/... (forward slash)
    if p.startswith('/') and len(p) >= 2 and p[1].isalpha() and p[2:3] == '/':
        p = p[1].upper() + ':/' + p[3:]
    # double-prefix D:\d\...
    m2 = re.match(r'^([A-Za-z]):\\d\\(.*)$', p)
    if m2:
        p = m2.group(1) + ':/' + m2.group(2)
    return p
print('fixed:', repr(fix('/d/rl-emergence/ckpts/solo_g64c')))
print('fixed2:', repr(fix('D:/rl-emergence/ckpts/x')))
