"""
PerceiverIO test file
"""

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../src/')

from attention_utils import MultiheadAttention

a = MultiheadAttention(in_dim=10, qlatent_dim=10, qk_dim=10, v_dim=10, out_dim=10, num_heads=10)