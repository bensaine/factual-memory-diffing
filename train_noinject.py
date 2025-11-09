import argparse
import numpy as np
import os
import sys
import torch
import torch.multiprocessing as mp

MEM_LIB_DIR = 'verbatim-memorization/src'
if MEM_LIB_DIR not in sys.path:
    sys.path.append(MEM_LIB_DIR)

from distributed_train import run_worker

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='pythia-160m-deduped-step80000')
    parser.add_argument('--window_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=2.79e-4)
    parser.add_argument('--pile_data_path', nargs='+', required=True, help="One or more .npy files")
    parser.add_argument('--save_optimizer_path', type=str, default='', help="If set, save AdamW state_dict here (warm-up run)")
    parser.add_argument('--load_optimizer_path', type=str, default='', help="If set, load AdamW state_dict for continued run")
    parser.add_argument('--log_dir', type=str, default='verbatim-memorization/models/noinject_run')
    parser.add_argument('--port', type=str, default='12358')
    args = parser.parse_args()

    arrays = []
    for p in args.pile_data_path:
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        arrays.append(np.load(p, mmap_mode='r'))
    data = np.concatenate(arrays, axis=0)
    print(f"Loaded data shape = {data.shape}")

    # Safety checks
    # TRAIN_STEPS = 1000   # 79k-80k
    # EVAL_STEPS  = 48     # 82k-82k+48
    TRAIN_STEPS = 2000   # 80k-82k
    EVAL_STEPS  = 48     # 82k-82k+48
    ROWS_PER_STEP = 1024

    train_rows_need = TRAIN_STEPS * ROWS_PER_STEP
    eval_rows_need  = EVAL_STEPS  * ROWS_PER_STEP

    # # Row count sanity checks for 79k to 80k
    # assert arrays[0].shape[0] >= train_rows_need, \
    #     f"ERROR: TRAIN rows insufficient: {arrays[0].shape[0]} < {train_rows_need}"

    # if len(arrays) >= 2:
    #     assert arrays[-1].shape[0] >= eval_rows_need, \
    #         f"ERROR: EVAL rows insufficient: {arrays[-1].shape[0]} < {eval_rows_need}"

    # n_rows = data.shape[0]
    # train_beg, train_end = 0, train_rows_need
    # eval_beg,  eval_end  = train_end, train_end + eval_rows_need

    # assert eval_end <= n_rows, \
    #     f"ERROR: Combined data rows insufficient: total={n_rows} < required={eval_end}"
    # assert train_end <= eval_beg, \
    #     "ERROR: Train/Eval ranges overlap!"

    assert len(arrays) >= 3, "Expect 3 arrays: 80-81k, 81-82k, 82k-82k+48"
    train_rows_have = arrays[0].shape[0] + arrays[1].shape[0]
    eval_rows_have  = arrays[2].shape[0]

    assert train_rows_have >= train_rows_need, \
        f"ERROR: TRAIN rows insufficient: {train_rows_have} < {train_rows_need}"
    assert eval_rows_have >= eval_rows_need, \
        f"ERROR: EVAL rows insufficient: {eval_rows_have} < {eval_rows_need}"

    data = np.concatenate(arrays, axis=0)
    n_rows = data.shape[0]
    train_beg, train_end = 0, train_rows_need
    eval_beg,  eval_end  = train_end, train_end + eval_rows_need
    assert eval_end <= n_rows, \
        f"ERROR: Combined data rows insufficient: total={n_rows} < required={eval_end}"
    assert train_end <= eval_beg, \
        "ERROR: Train/Eval ranges overlap!"

    STRIDE = 2048
    SEQ_LEN = 256 + 64

    assert (ROWS_PER_STEP * STRIDE) % 1 == 0, \
        "ERROR: Invalid stride / rows_per_step relationship"

    assert data.shape[1] == SEQ_LEN, \
        f"ERROR: Each row must have length {SEQ_LEN}, got {data.shape[1]}"

    sample_ids = [0, train_end-1, eval_beg, eval_end-1]
    for sid in sample_ids:
        row = np.array(data[sid], copy=False)
        pad_ratio = (row == 0).mean()
        print(f"Check: row {sid}: zero_ratio={pad_ratio:.3f}")
        assert pad_ratio < 1.0, \
            f"ERROR: Row {sid} appears all-zero — possible extraction error or out-of-range index"

    # --- Fingerprint signatures of critical rows ---
    import hashlib
    def row_sig(arr): 
        return hashlib.md5(arr.tobytes()).hexdigest()[:8]

    print("train[0]      =", row_sig(np.array(data[0], copy=False)))
    print("train[last]   =", row_sig(np.array(data[train_end-1], copy=False)))
    print("eval[0]       =", row_sig(np.array(data[eval_beg], copy=False)))
    print("eval[last]    =", row_sig(np.array(data[eval_end-1], copy=False)))

    print(f"OK: TRAIN rows = [{train_beg}, {train_end}), "
        f"EVAL rows = [{eval_beg}, {eval_end})")

    inject_every_n = 10_000
    
    # no injection config
    config = {
        'port': args.port,
        'inject_every_n': inject_every_n,
        'total_number_inject': 40,
        'inject_data': {},                  
        'training_batch_size': 128,
        'eval_batch_size': 128,
        'training_sample_range': [0, 2000*1024],
        'eval_sample_range': [2000*1024, 2000*1024 + 48*1024],
        'window_size': args.window_size,
        'base_model': args.checkpoint,
        'init_lr': args.lr,
        'group': 'noinject',
        'log_dir': args.log_dir,
        'model_dir': 'verbatim-memorization/models',
        'data': data,
        'run_eval': True,
        'pretrained_optimizer_path': args.load_optimizer_path,
        'save_optimizer_path': args.save_optimizer_path, 
    }

    world_size = torch.cuda.device_count()
    print(f"world_size={world_size}")
    os.makedirs(args.log_dir, exist_ok=True)
    mp.spawn(run_worker, args=(world_size, config,), nprocs=world_size, join=True)
