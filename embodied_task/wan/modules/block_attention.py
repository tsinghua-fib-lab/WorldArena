import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask
from einops import rearrange
flex_attention_compiled = torch.compile(flex_attention, dynamic=False)


def get_flex_causal_block_mask_for_prefill(num_denoise_block, block_size=16 * 16, device="cuda"):
    seq_len = num_denoise_block * block_size
    def block_mask_mod(b, h, q_idx, kv_idx):
        bqi = q_idx // block_size
        bki = kv_idx // block_size
        return bqi >= bki
    block_mask = create_block_mask(block_mask_mod, None, None, Q_LEN=seq_len, KV_LEN=seq_len, device=device, _compile=True)
    return block_mask


def get_flex_block_mask_chunk_prefill(num_qblock, num_kvblock, block_size=16 * 16, device="cuda"):
    """
    attention mask like, numq = 2, kvlen=4的一个情况:
    [1, 1, 1, 1, 1, 0]
    [1, 1, 1, 1, 1, 1]
    """
    Q_LEN = num_qblock * block_size
    KV_LEN = (num_kvblock + num_qblock) * block_size
    def block_mask_mod(b, h, q_idx, kv_idx):
        bqi = q_idx // block_size
        bki = kv_idx // block_size
        return torch.where(bki < num_kvblock, True, bqi + num_kvblock >= bki)
    block_mask = create_block_mask(block_mask_mod, None, None, Q_LEN=Q_LEN, KV_LEN=KV_LEN, device=device, _compile=True)
    return block_mask

def BLHD_flex_with_mask(q, k, v, attention_mask=None):
    q, k, v = map(lambda x: rearrange(x, "b l h d -> b h l d").contiguous(), (q, k, v))
    o = flex_attention_compiled(q, k, v, block_mask=attention_mask)
    return rearrange(o, "b h l d -> b l h d")

