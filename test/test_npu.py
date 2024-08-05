import torch
from torch.testing._internal.common_utils import (
    get_cycles_per_ms,
    run_tests,
    TestCase
)


class TestNPU(TestCase):
    def test_events(self):
        stream = torch.npu.current_stream()
        event = torch.npu.Event(enable_timing=True)
        self.assertTrue(event.query())
        start_event = torch.npu.Event(enable_timing=True)
        stream.record_event(start_event)
        torch.npu._sleep(int(50 * get_cycles_per_ms()))  # TODO
        stream.record_event(event)
        self.assertFalse(event.query())
        event.synchronize()
        self.assertTrue(event.query())
        self.assertGreater(start_event.elapsed_time(event), 0)


if __name__ == "__main__":
    run_tests()
