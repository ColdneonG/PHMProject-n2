"""Recompute main and supplementary mean/sample-SD tables without ML dependencies."""
import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
METRICS = ['overall_dice', 'overall_iou', 'small_dice', 'boundary_f1', 'level_mae',
           'condition_clear_dice', 'condition_blur_dice', 'condition_small_dice',
           'condition_blur_level_mae']

def aggregate():
    selection = json.loads((ROOT / 'results/run_selection.json').read_text())
    lines = ['# Experimental results', '',
             'Mean ± sample standard deviation across seeds 42, 3407 and 2026. '
             'Computed from the per-run metrics included in this repository.', '']
    rows = []
    all_runs = {}
    for cohort in ['main', 'supplementary']:
        grouped = {}
        for run in selection[cohort]:
            d = json.loads((ROOT / 'results' / cohort / run / 'final_metrics.json').read_text())
            assert d['test']['num_samples'] == 1354, run
            model = d['args']['model']
            seed = d['args']['seed']
            if seed in grouped.setdefault(model, {}):
                raise ValueError(f'Duplicate {cohort}/{model}/{seed}')
            grouped[model][seed] = d['test']
        all_runs[cohort] = grouped
        lines += [f'## {cohort.title()} cohort', '', '| Model | Dice | IoU | Small Dice | Boundary F1 | Level MAE | Blur Dice |',
                  '| --- | --- | --- | --- | --- | --- | --- |']
        for model, runs in sorted(grouped.items()):
            assert set(runs) == {42, 3407, 2026}, (cohort, model)
            stats = {}
            for metric in METRICS:
                values = [runs[s][metric] for s in [42, 3407, 2026]]
                assert all(v is not None for v in values), (model, metric)
                mean, sd = statistics.mean(values), statistics.stdev(values)
                rows.append(dict(cohort=cohort, model=model, metric=metric, n=3, mean=mean, sample_sd=sd))
                stats[metric] = f'{mean:.4f} ± {sd:.4f}'
            lines.append('| ' + model + ' | ' + ' | '.join(stats[k] for k in METRICS[:5] + ['condition_blur_dice']) + ' |')
        lines.append('')
    ours = all_runs['main']['phm_project_n2']
    base = all_runs['main']['deeplabv3plus_resnet18_pretrained']
    dice_delta = statistics.mean(ours[s]['condition_blur_dice'] - base[s]['condition_blur_dice'] for s in ours)
    our_mae = statistics.mean(v['condition_blur_level_mae'] for v in ours.values())
    base_mae = statistics.mean(v['condition_blur_level_mae'] for v in base.values())
    lines += ['## Main-cohort comparison with DeepLabV3+', '',
              f'Blur Dice improvement: {100*dice_delta:.2f} percentage points. '
              f'Blur Level MAE relative reduction: {100*(base_mae-our_mae)/base_mae:.2f}%.', '',
              'The supplementary cohort is reported separately, including its repeated A3/A4 anchors. '
              'Each series is summarized over its own three seeds.']
    (ROOT / 'results/TABLES.md').write_text('\n'.join(lines) + '\n')
    with (ROOT / 'results/summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    expected_path = ROOT / 'results/manuscript_expected.json'
    if expected_path.exists():
        expected = json.loads(expected_path.read_text())
        checked = 0
        for row in rows:
            if row['cohort'] == 'main' and row['model'] in expected and row['metric'] in expected[row['model']]:
                target = expected[row['model']][row['metric']]
                assert [f"{row['mean']:.4f}", f"{row['sample_sd']:.4f}"] == target, row
                checked += 1
        print(f'Manuscript rounded mean/SD cells checked: {checked}')
    print(f'Aggregated {len(selection["main"])} main and {len(selection["supplementary"])} supplementary runs.')

if __name__ == '__main__':
    aggregate()
