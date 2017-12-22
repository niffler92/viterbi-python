import numpy as np
from pandas_ml import ConfusionMatrix
import matplotlib.pyplot as plt


def label_to_word(label):
    word_dict = {'1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six', '7': 'seven',
                '8': 'eight', '9': 'nine', 'o': 'oh', 'z': 'zero'}

    return [word_dict[label] for label in label]


if __name__ == "__main__":
    filepath = "results.npy"
    result_dict = np.load(filepath).item()
    y_trues_raw = result_dict["y_trues"]
    y_preds = result_dict["y_preds"]

    y_trues = [label_to_word(label) for label in y_trues_raw]

    assert len(y_preds) == len(y_trues)

    y_trues = np.ravel(y_trues)
    y_preds = np.ravel(y_preds)
    print(y_trues)
    print(y_preds)
    print(len(y_trues), len(y_preds))

    cm = ConfusionMatrix(y_trues, y_preds)
    cm.print_stats()
