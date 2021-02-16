import pandas as pd
import numpy as np
import torch
from sklearn import preprocessing
from sklearn import model_selection
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import joblib

import config
import dataset
import engine
import utils
from model import EntityModel


def run():
    sentences, pos, tag, enc_pos, enc_tag = utils.process_data(config.DATA_FILE)

    meta_data = {
        "enc_pos": enc_pos,
        "enc_tag": enc_tag
    }

    joblib.dump(meta_data, "meta.bin")

    num_pos = len(list(enc_pos.classes_))
    num_tag = len(list(enc_tag.classes_))

    (
        train_sentences,
        test_sentences,
        train_pos,
        test_pos,
        train_tag,
        test_tag
    ) = model_selection.train_test_split(sentences, pos, tag, random_state=42, test_size=0.1)

    train_dataset = dataset.EntityDataset(
        texts = train_sentences, pos=train_pos, tags=train_tag
    )

    test_dataset  = dataset.EntityDataset(
        texts = test_sentences, pos=test_pos, tags=test_tag
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size = config.TRAIN_BATCH_SIZE, num_workers=4
    )

    test_data_loader  = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1
    )

    device = torch.device("cuda")

if __name__ == "__main__":

    run()