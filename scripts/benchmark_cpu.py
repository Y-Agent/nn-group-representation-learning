"""Measure exact CE update cost before scheduling wide CPU runs."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from src.flow import direction_step
from src.model import WideNetworkScaleSphere
from src.spectral import Spectrum

for width, group in [(512, "f21"), (50000, "z30")]:
    for threads in [1, 4, 12]:
        torch.set_num_threads(threads)
        torch.manual_seed(42)
        s = Spectrum(group)
        m = WideNetworkScaleSphere(s.n, width, share_embed=False).double()
        m.normalize_directions()
        m.scale_in.requires_grad_(False)
        m.scale_out.requires_grad_(False)
        x, y = s.task.generate_all_data(torch.device("cpu"))
        direction_step(m, x, y, width * 0.02, full_table=True)
        start = time.time()
        for _ in range(5):
            direction_step(m, x, y, width * 0.02, full_table=True)
        print(
            json.dumps(
                dict(
                    width=width,
                    group=group,
                    threads=threads,
                    seconds_per_step=(time.time() - start) / 5,
                )
            ),
            flush=True,
        )
