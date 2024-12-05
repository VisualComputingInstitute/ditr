_base_ = ["semseg-pt-v3m1-0-nu-sk-wa.py"]

epoch = 8
eval_epoch = 8

optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.002, 2e-06, 2e-05],
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)
param_dicts = [dict(keyword="block", lr=2e-06), dict(keyword="backbone", lr=2e-05)]
