#!/usr/bin/env python3
"""Check published aggregate CSV integrity without downloading models or images."""
from pathlib import Path
import csv

root=Path(__file__).resolve().parents[1]

def read_csv(rel):
    path=root/rel
    with path.open(newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))

rows08=read_csv('results/08_robustness/pooled_metrics.csv')
rows09=read_csv('results/09_enhancement/pooled_metrics.csv')
base08={(r['condition'],r['model']):r for r in rows08}
base09={(r['condition'],r['model']):r for r in rows09 if r['method']=='none'}
assert len(rows08)==21, f'08: expected 7 conditions × 3 heads, got {len(rows08)}'
assert len(rows09)==27, f'09: expected 3 conditions × 3 methods × 3 heads, got {len(rows09)}'
assert all(int(r['n'])==4390 for r in rows08+rows09), 'Unexpected test size'
assert len(base09)==9
for key,row in base09.items():
    assert key in base08, f'Missing 08 reference: {key}'
    ref=base08[key]
    for metric in ['accuracy','macro_f1']:
        assert abs(float(row[metric])-float(ref[metric]))<1e-12, (key,metric,row[metric],ref[metric])
parity=read_csv('results/08_robustness/baseline_parity.csv')
assert len(parity)==15 and all(int(x['mismatch_n'])==0 for x in parity), 'Baseline parity was not zero'
for model in ['deep_fcn','deep_lr','deep_lgbm']:
    get=lambda method:float(next(x['macro_f1'] for x in rows09 if x['condition']=='dark_strong' and x['method']==method and x['model']==model))
    assert get('gamma')>get('none')>get('clahe'), f'Unexpected severe-dark ordering for {model}'
print('PASS: 08/09 baseline parity and aggregate severe-dark results are internally consistent.')
