import os
import cv2
from tqdm import tqdm
import glob

cities = ['Khartoum', 'Shanghai', 'Paris', 'Vegas']

def get_spacenet_names(directory):
    #names = os.listdir(directory)
    names = glob.glob(os.path.join(directory, '*.tif'))
    data = {}
    for i in cities:
        data[i] = []

    for name in tqdm(names):
        if 'RGB' in name:
            n = name.split('/')[-1].split('_')[0]
 
            for c in cities:
                if c in n:
                    label = cv2.imread(name.split('_')[0] + '_GT.tif')
                    if label.max() != 0:
                        data[c].append(n)
    return data

if __name__ == "__main__":
    import json
    data = get_spacenet_names('/usr/xtmp/satellite/spacenet/')
    with open('../dataset/spacenet/building_all_cities.json', 'w') as f:
        json.dump(data, f)
    for city in cities:
        with open('../dataset/spacenet/' + city + '.json', 'w') as f:
            json.dump(data[city], f)


