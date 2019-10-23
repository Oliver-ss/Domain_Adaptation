#!/usr/bin/env python3
import os
import sys
import glob
import time
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Union

import json
import numpy as np
from tabulate import tabulate

def main(args):
    all_results = []

    while True:
        all_results = []
        for ef in args.result_root.glob('*.json'):
            res = json.load(ef.open())
            all_results.append((ef.name.rstrip('.json'), res))
        all_results = sorted(all_results, key=lambda x: x[1]['IoU'], reverse=True)

        for res in all_results[args.limit:]:
            if not os.path.exists("./train_log/models/{}.pth".format(res[0])):
                continue
            cmd = "rm ./train_log/models/{}.pth".format(res[0])
            os.system(cmd)

        keys = ['Acc', 'IoU', 'mIoU']
        table = [
            [name, ] + [v[key] for key in keys]
            for name, v in all_results[:args.limit]
        ]

        print(os.path.basename(os.getcwd()))
        print(tabulate(table, headers=("name", *keys), tablefmt="pipe"))

        if args.watch <= 0:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--watch", type=int, default=100)
    parser.add_argument("--result-root", type=Path, default='./train_log/eval', help="result dir")
    args = parser.parse_args()

    main(args)

# vim: ts=4 sw=4 sts=4 expandtab

