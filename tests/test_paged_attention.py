import pytest
from src.serving.paged_attention import PagedKVCacheManager

def test_paged_attention_allocation():
    manager = PagedKVCacheManager(block_size=16, num_blocks=64)
    allocated = manager.allocate_sequence("SEQ-01", prompt_len=45)
    assert len(allocated) == 3  # (45 + 15) // 16 = 3 blocks
    util = manager.get_memory_utilization()
    assert util["allocated_blocks"] == 3
    manager.free_sequence("SEQ-01")
    assert len(manager.free_blocks) == 64
