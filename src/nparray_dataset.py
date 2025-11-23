import collections
import numpy as np

import torch

def print_with_rank(rank, *arg):
  print(f'[RANK {rank}]', *arg)
class NumpyArrayDataset(torch.utils.data.Dataset):
  """Numpy array dataset."""
  def __init__(self,
               data,
               sample_range,
               inject_data=None,
               inject_every_n=None,
               tokenizer=None,
               debug_counters=None,
               seed=42,
               **kwargs):
    super(NumpyArrayDataset, self).__init__()
    self.window_size = (
            256 if 'window_size' not in kwargs else kwargs['window_size'])
    self.data = data[sample_range[0]:sample_range[1], :self.window_size + 64]
    self.counters = collections.defaultdict(int)
    self.inject_data = inject_data
    self.inject_every_n = inject_every_n
    if inject_data and tokenizer is None:
      raise ValueError
    self.tokenizer = tokenizer
    # For multi-processing logging.
    self.debug_counters = debug_counters or collections.defaultdict(list)
    self.debug_id = kwargs['process_id'] if 'process_id' in kwargs else None
    self.seed = int(seed)
    self.inject_ids = None
    if self.inject_data:
        if self.tokenizer is None:
            raise ValueError("inject_data needs tokenizer")
        self.inject_ids = {}
        for k, variants in self.inject_data.items():
            ids_list = []
            for text in variants:
                ids = self.tokenizer(
                    text,
                    return_tensors='pt',
                    add_special_tokens=False,
                    padding=False,
                    truncation=False
                ).input_ids[0].to(torch.long)  # shape: [L_inj]
                ids_list.append(ids)
            self.inject_ids[k] = ids_list

  def _per_example_generator(self, key: int, index: int) -> torch.Generator:
      g = torch.Generator(device='cpu')
      mix = (self.seed & 0xFFFFFFFF) ^ (int(key) << 16) ^ (int(index // self.inject_every_n) << 1) ^ int(index)
      g.manual_seed(mix % (2**63 - 1))
      return g

  def __getitem__(self, index):
    x = torch.tensor(self.data[index, :self.window_size + 1].astype(np.int64))

    if self.inject_data and self.data.shape[0] > 10_000:
        for key, ids_list in self.inject_ids.items():
            if index % self.inject_every_n == key:
                self.debug_counters[f'counter-{key}'].append(index)
                gen = self._per_example_generator(key, index)
                vidx = int(torch.randint(low=0, high=len(ids_list), size=(1,), generator=gen).item())
                inj_ids = ids_list[vidx]

                self.debug_counters.setdefault(f'variant-{key}', []).append(vidx)

                orig_x = x.clone()

                side_coin = float(torch.rand(1, generator=gen).item())
                if side_coin < 0.5:
                    combined = torch.cat([inj_ids, x], dim=0)
                    x = combined[: self.window_size + 1]
                    side = 'front'
                else:
                    combined = torch.cat([x, inj_ids], dim=0)
                    x = combined[-(self.window_size + 1):]
                    side = 'back'

                self.debug_counters.setdefault('inject_side', []).append(side)

                print_with_rank(
                    self.debug_id,
                    f"index={index}",
                    f"key={key}",
                    f"side={side}",
                    f"variant_idx={vidx}",
                    f"orig_x[:20]={orig_x[:20].tolist()}",
                    f"inj_ids={inj_ids.tolist()}",
                    f"after_x[:20]={x[:20].tolist()}",
                )

                break  

    return {'input_ids': x}



  def __len__(self):
    return self.data.shape[0]
