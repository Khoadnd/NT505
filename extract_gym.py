# Prepare data for training RL model

from gym_malware.envs.utils.pefeatures2 import PEFeatureExtractor2
import os
import numpy
from tqdm import tqdm

path_benign = 'Data/benign'
path_malware = 'Data/malware'
feature_extractore2 = PEFeatureExtractor2()


class FileRetrievalFailure(Exception):
    pass


def fetch_file(path: str, sha256: str) -> bytes:
    location = os.path.join(path, sha256)
    try:
        with open(location, 'rb') as infile:
            bytez = infile.read()
    except IOError:
        raise FileRetrievalFailure(
            "Unable to read sha256 from {}".format(location))

    return bytez


def extract_data(path) -> list:
    data = []

    for filename in tqdm(os.listdir(path), desc="Process: ", ascii='#'):
        try:
            bytez = fetch_file(path, filename)
            features = feature_extractore2.extract(bytez)
            data.append(numpy.array(features))
            del bytez, features
        except IOError:
            raise FileNotFoundError(
                "unable to extract file from {}".format(
                    os.path.join(path, filename))
            )

    return data


print('Extracting benign data...')
benign_data = extract_data(path_benign)

print('Extracting malware data...')
malware_data = extract_data(path_malware)

print('Saving data...')
numpy.savetxt('Data/benign_data.csv', benign_data, delimiter=',')
numpy.savetxt('Data/malware_data.csv', malware_data, delimiter=',')

print('ALL DONE!')
