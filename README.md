
## Data

TODO
- [ ] Add noise. Such as A likes sth. This can potentially add difficulty for attention to locate persons.
- [ ] Make sure the train/test does not overlap.

## Training

Successful runs
- Custom 4L4H gpt. No positional encoding. Objective on target only. Hack on training infinite with same datasets each time. The model converges to zero train error, 
near zero test error, and high ood error. Config is at out/out_temp_2/config.json.

- Custom 4L4H gpt. No positional encoding. Objective on target only. Batch size 1000 on a 11394 train set.
    - Train and test is 0 to 3 chain length questions. OOD is 4 chain length questions.
    - The model converges to near [zero error](https://wandb.ai/jiajun-song928/symbolic_tom/runs/20250527-114307/panel/0l8gw192t/panel/0l8gw192t?nw=nwuserjiajunsong928) for train, test, and ood. (99%, 97% and 94%).
    - config is at out/out_batch_size_1000_chain_length_zero_to_three