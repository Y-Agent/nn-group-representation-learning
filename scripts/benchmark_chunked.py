import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from src.flow import exact_chunked_table_step
from src.model import WideNetworkScaleSphere
from src.spectral import Spectrum

s = Spectrum("z30")
x, y = s.task.generate_all_data(torch.device("cpu"))
for threads in [1, 4, 12]:
    torch.set_num_threads(threads)
    m = WideNetworkScaleSphere(30, 50000, share_embed=False).double()
    m.normalize_directions()
    for chunk in [256, 512, 1024, 2048]:
        exact_chunked_table_step(m, y, 1000, chunk)
        start = time.time()
        for _ in range(3):
            exact_chunked_table_step(m, y, 1000, chunk)
        print(
            json.dumps(
                dict(threads=threads, chunk=chunk, seconds_per_step=(time.time() - start) / 3)
            ),
            flush=True,
        )
