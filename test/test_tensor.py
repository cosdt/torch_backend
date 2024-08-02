
import torch
import torch_npu
import pytest

@pytest.mark.npu
def test_create_tensor():
    a = torch.ones((3,3), device='npu')
    assert (3,3) == a.shape, f"Expected tensor shape (3,3), but got {a.shape}"
