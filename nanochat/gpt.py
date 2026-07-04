from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


from nanochat.flash_attention import flassh_attn
from anonchat.optim import MuonAdamW
from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
