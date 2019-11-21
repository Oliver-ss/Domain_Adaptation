import csv
import os
import json
import argparse
import re

net = ['mobilenet', 'resnet']
city = ['Shanghai', 'Vegas', 'Paris', 'Khartoum']
fieldnames = ['exp(%)', 'Shanghai', 'boost_Shanghai', 'Vegas', 'boost_Vegas', 'Paris', 'boost_Paris', 'Khartoum',
              'boost_Khartoum', 'avg_boost']


def main(args):
    result_folder = '../../results'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    for i in range(4):
        if re.search(city[i], args.exp, re.IGNORECASE):
            source_city = city[i]
            break

    for i in range(2):
        if re.search(net[i], args.exp, re.IGNORECASE):
            backbone = net[i]
            break

    if not os.path.exists(os.path.join(result_folder, backbone)):
        os.makedirs(os.path.join(result_folder, backbone))

    csv_name = os.path.join(result_folder, backbone, source_city + '.csv')

    all_result = []
    if os.path.exists(csv_name):
        with open(csv_name, 'r') as f:
            reader = csv.DictReader(f)
            for exp in reader:
                all_result.append(exp)

    # test_path = os.path.join('../configs', args.exp, 'train_log', args.name)
    test_path = os.path.join('train_log', args.name)
    with open(test_path) as f:
        test = json.load(f)
    result_dict = {}
    for i in range(4):
        result_dict[city[i]] = round(test[city[i]]['IoU']*100, 2)

    if not args.is_baseline:
        baseline = all_result[1]
    result_dict['exp(%)'] = args.exp if args.name == 'test.json' else args.exp + '.' + args.name.split('.')[0]
    avg_boost = 0
    for i in range(4):
        if args.is_baseline:
            result_dict['boost_' + city[i]] = 0
        else:
            result_dict['boost_' + city[i]] = round(result_dict[city[i]] - float(baseline[city[i]]), 2)
            # result_dict['boost_' + city[i]] = round(
            #     (result_dict[city[i]] - float(baseline[city[i]])) / float(baseline[city[i]]), 2)
        avg_boost += result_dict['boost_' + city[i]]
    result_dict['avg_boost'] = round(avg_boost / 4, 2)

    is_inserted = False
    if not args.is_baseline:
        for i in range(2, len(all_result)):
            if result_dict['avg_boost'] >= float(all_result[i]['avg_boost']):
                all_result.insert(i, result_dict)
                is_inserted = True
                break
    if not is_inserted:
        all_result.append(result_dict)

    with open(csv_name, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(all_result)):
            writer.writerow(all_result[i])


if __name__ == '__main__':
    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

    parser = argparse.ArgumentParser()
    parser.add_argument("exp", type=str)
    parser.add_argument("name", type=str, default="test.json")
    parser.add_argument("is_baseline", type=str2bool, default=False)
    args = parser.parse_args()
    # print(args.exp)
    # print(args.name)
    # print(args.is_baseline)
    main(args)
