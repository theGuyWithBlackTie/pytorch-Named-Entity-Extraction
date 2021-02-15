import pandas as pd
import numpy as np
import torch
from sklearn import preprocessing
from sklearn import model_selection
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

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

if __name__ == "__main__":

    run()