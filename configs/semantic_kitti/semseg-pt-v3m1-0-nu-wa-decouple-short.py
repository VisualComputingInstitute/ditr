_base_ = ["semseg-pt-v3m1-0-nu-wa-decouple.py"]

epoch = 10
eval_epoch = 10

optimizer = dict(type="AdamW", lr=0.0002, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.0002, 0.00002],
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)
param_dicts = [dict(keyword="block", lr=0.00002)]
