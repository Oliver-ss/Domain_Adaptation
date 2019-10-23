import os
import random
import json

cities = ['Khartoum', 'Shanghai', 'Paris', 'Vegas']

def split_dataset(img_json, output_path, train, val, test):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    with open(img_json) as f:
        img_list = json.load(f)
    random.shuffle(img_list)
    train_num = int(len(img_list) * train)
    val_num = int(len(img_list) * val)
    train_list = img_list[:train_num]
    val_list = img_list[train_num:(train_num+val_num)]
    test_list = img_list[(train_num+val_num):]

    with open(os.path.join(output_path, 'train.json'), 'w') as f:
        json.dump(train_list, f)
    with open(os.path.join(output_path, 'val.json'), 'w') as f:
        json.dump(val_list, f)
    with open(os.path.join(output_path, 'test.json'), 'w') as f:
        json.dump(test_list, f)


if __name__ == '__main__':
    top_d = '../dataset/spacenet'
    domain_d = '../dataset/domains'
    train = 0.6
    val = 0.2
    test = 0.2

    for city in cities:
        img_json = os.path.join(top_d, city + '.json')
        output_path = os.path.join(domain_d, city)
        split_dataset(img_json, output_path, train, val, test)