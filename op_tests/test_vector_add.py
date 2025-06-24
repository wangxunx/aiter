import torch
import aiter
from aiter.test_common import checkAllclose, perftest



@perftest()
def run_vector_add_torch(
    input1,
    input2,
):
    output = input1 + input2
    return output


@perftest()
def run_vector_add_cu(
    output,
    input1,
    input2,
):
    return aiter.vector_add_cu(output, input1, input2)


def test_vector_add():
    input1 = torch.randn(100, 100, dtype=torch.float32, device='cuda')
    input2 = torch.randn(100, 100, dtype=torch.float32, device='cuda')
    b = torch.empty_like(input1)

    # Test the vector_add function
    a, avg_a = run_vector_add_torch(input1, input2)
    _, avg_b = run_vector_add_cu(b, input1, input2)

    msg = f"[perf] =torch avg: {avg_a:<8.2f} us, cu avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b-1:<5.1%}"
    checkAllclose(a, b, rtol=1e-3, atol=1e-3, msg=msg)


test_vector_add()