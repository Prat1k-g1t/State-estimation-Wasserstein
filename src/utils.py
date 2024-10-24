# SPDX-FileCopyrightText: The sinkhorn-rom authors, see file AUTHORS
#
# SPDX-License-Identifier: GPL-3.0-or-later

import os
# os.environ['CUDA_PATH'] = '/home/prai/anaconda3/lib/python3.10/site-packages/triton/third_party/cuda/include/cuda.h'
import time
from contextlib import contextmanager
from typing import List
from prompt_toolkit import print_formatted_text, HTML
from prompt_toolkit.styles.named_colors import NAMED_COLORS
from prompt_toolkit.styles import Style
from sklearn import preprocessing

# Performance utils
# =================
@contextmanager
def timeit(msg, font_style='', fg='Black', bg='White'):
    pprint(msg, font_style=font_style, fg=fg, bg=bg)
    t0 = time.time()
    yield
    pprint('Finished {}. Run time: {}'.format(msg, time.time() - t0), font_style=font_style, fg=fg, bg=bg)

# Path utils
# ==========
def check_create_dir(dir_name):
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

# Print utils
# ============
def pprint(text, font_style='', fg='Black', bg='White'):
    """ANSI color labels: from prompt_toolkit.styles.named_colors import NAMED_COLORS
    """
    if font_style not in ['', 'underline', 'italic', 'bold']:
        print('Warning: Invalid font_style '+font_style+'. Options are: underline, italic, bold or plain (empty string).')
    if fg not in NAMED_COLORS:
        print('Warning: fg color '+fg+' not in ANSI NAMED_COLORS. We use Black instead.')
        fg = 'Black'
    if bg not in NAMED_COLORS:
        print('Warning: bg color '+bg+' not in ANSI NAMED_COLORS. We use White instead.')
        bg = 'White'
    style = Style.from_dict({'a': '{} fg:{} bg:{}'.format(font_style, fg, bg)})
    #return print_formatted_text(HTML('<a>{}</a>'.format(text)), style=style)
    return print(text)

def preprocessing_parameters(parameters):
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(parameters)
    return X

