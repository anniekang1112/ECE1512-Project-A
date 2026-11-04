"""

Evaluate existing weights without training or writing into a result directory

Use the original run's config and course test split. Output goes to stdout so
checking a checkpoint cannot overwrite the frozen metrics
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault('MPLCONFIGDIR', '/tmp/ece1512-code-qa-mpl')
sys.path.insert(0, str(ROOT / 'Project_A_Starter_Code'))

import torch

from torch.utils.data import DataLoader
from dataloader import build_HDF5_feat_dataset, mil_collate_fn
from utils import Struct
from mil_pipeline import ABMIL, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'results/C16_E0_seed42')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')


    args = parser.parse_args()
    run = args.run_dir.resolve()

    checkpoint = args.checkpoint or run / 'best.pt'
    config = json.loads((run / 'config.json').read_text())

    # stored paths may come from the training machine; use this checkout's inputs

    dataset = {'CAMELYON16': 'C16', 'CAMELYON17': 'C17', 'BRACS': 'BRACS'}[config['dataset']]
    config['data_dir'] = str(ROOT / 'Project_A_Starter_Code/feature' / dataset / 'patch_feats_pretrain_medical_ssl.h5')
    config['dataset_csv_dir'] = str(ROOT / 'Project_A_Starter_Code/dataset_csv')
    conf = Struct(**config)
    _, _, test = build_HDF5_feat_dataset(conf.data_dir, conf)

    torch.set_num_threads(4)

    device = torch.device(args.device)
    model = ABMIL(conf).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))

    loader = DataLoader(test, batch_size=1, shuffle=False, collate_fn=mil_collate_fn)

    # unweighted CE matches the evaluation objective, including for E1/E3

    metrics = evaluate(model, torch.nn.CrossEntropyLoss(), loader, device, conf)


    print(json.dumps({'checkpoint': str(checkpoint), 'test_bags': len(test),
                      'accuracy_unit': 'percent', 'metrics': metrics}, indent=2))


if __name__ == '__main__':
    main()
