import csv
import os
import json
import argparse


def append_to_csv(args):
    filepath = os.path.join('../configs', args.exp, 'train_log')
    test_path = os.path.join(filepath, 'test.json')
    test_bn_path = os.path.join(filepath, 'test_bn.json')
    csv_folder = '../results'
    if not os.path.exists(csv_folder):
        os.mkdir(csv_folder)
    with open(test_path) as f:
        test = json.load(f)
    with open(test_bn_path) as f:
        test_bn = json.load(f)
    fieldnames = ['source','Shanghai', 'Vegas', 'Paris', 'Khartoum']
    if not os.path.exists(os.path.join(csv_folder, 'baseline.csv')):
        write_header = True
    else:
        write_header = False
    with open(os.path.join(csv_folder, 'baseline.csv'), 'a') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        # test
        source_name = args.exp.split('.')[-1] + '.test'
        writer.writerow({fieldnames[0]: source_name,
                         fieldnames[1]: round(test[fieldnames[1]]['IoU'], 2),
                         fieldnames[2]: round(test[fieldnames[2]]['IoU'], 2),
                         fieldnames[3]: round(test[fieldnames[3]]['IoU'], 2),
                         fieldnames[4]: round(test[fieldnames[4]]['IoU'], 2),
                         })
        # test_bn
        source_name = args.exp.split('.')[-1] + '.test_bn'
        writer.writerow({fieldnames[0]: source_name,
                         fieldnames[1]: round(test_bn[fieldnames[1]]['IoU'], 2),
                         fieldnames[2]: round(test_bn[fieldnames[2]]['IoU'], 2),
                         fieldnames[3]: round(test_bn[fieldnames[3]]['IoU'], 2),
                         fieldnames[4]: round(test_bn[fieldnames[4]]['IoU'], 2),
                         })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("exp", type=str)
    args = parser.parse_args()
    append_to_csv(args)
