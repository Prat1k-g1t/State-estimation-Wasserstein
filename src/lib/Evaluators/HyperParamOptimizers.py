# SPDX-FileCopyrightText: The sinkhorn-rom authors, see file AUTHORS
#
# SPDX-License-Identifier: GPL-3.0-or-later

import itertools
from contextlib import contextmanager

import numpy as np
from tqdm import tqdm


def get_combinatorial_optimizer_values(msg, **kwargs):
    for particular_values in tqdm(itertools.product(*kwargs.values()), desc='Fitting {}'.format(msg)):
        yield {k: v for k, v in zip(kwargs.keys(), particular_values)}