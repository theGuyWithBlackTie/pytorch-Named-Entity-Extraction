import pandas as pd
from sklearn import preprocessing

def process_data(data_path):
    df = pd.read_csv(data_path, encoding="latin-1")
    
    df.loc[:, "Sentence #"] = df["Sentence #"].fillna(method="ffill")

    enc_pos = preprocessing.LabelEncoder()
    enc_tag = preprocessing.LabelEncoder()

    df.loc[:, "POS"] = enc_pos.fit_transform(df["POS"])
    df.loc[:, "Tag"] = enc_tag.fit_transform(df["Tag"])

    sentences = df.groupby("Sentence #")["Word"].apply(list).values
    pos       = df.groupby("Sentence #")["POS"].apply(list).values
    tag       = df.groupby("Sentence #")["Tag"].apply(list).values

    return sentences, pos, tag, enc_pos, enc_tag