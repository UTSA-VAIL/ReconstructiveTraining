import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

def top_k_accuracy(predictions, labels, k=1):
    argsorted_predictions = np.argsort(predictions)[:, -k:]
    y_true = tf.keras.utils.to_categorical(labels)
    return np.any(argsorted_predictions.T == y_true.argmax(axis=1), axis=0).mean()


def get_df(top5_preds, classes, labels):
    column_names = ["Actual", "Pred 1", "Pred 2", "Pred 3", "Pred 4", "Pred 5"]
    df = pd.DataFrame(columns=column_names)
    for i, decode in tqdm(enumerate(top5_preds)):
        df = df.append(
            {
                "Actual": classes[str(labels[i])][1],
                "Pred 1": decode[0][1],
                "Pred 2": decode[1][1],
                "Pred 3": decode[2][1],
                "Pred 4": decode[3][1],
                "Pred 5": decode[4][1],
            },
            ignore_index=True,
        )
    return df


def get_top_results(preds, labels):
    stats = [top_k_accuracy(preds, labels, k=1) * 100.0, top_k_accuracy(preds, labels, k=5) * 100.0]
    return stats
