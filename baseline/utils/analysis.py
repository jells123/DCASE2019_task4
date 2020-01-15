import os
import matplotlib.pyplot as plt
import pandas as pd
import config as cfg

def generate_graphs(metrics_filepath, epoch_filepath):
    dirpath = os.path.splitext(metrics_filepath)[0]
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    df = pd.read_csv(metrics_filepath, sep=";")
    del df["Acc_Seg"]
    df.columns = ['weak-F1', 'Nref', "F", "Pre", "Rec", "Acc", "Nref_Seg", "F_seg", "Pre_Seg", "Rec_Seg", "Acc_Seg"]
    if df.shape[0] > 0:
        for column in df.columns:
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.set(xlabel="Epoch", ylabel=column)
            pl = df.groupby(df.index)[column].plot(legend=True,
                                                   ax=ax,
                                                   use_index=False,
                                                   title="{} values for each epoch".format(column))
            plt.savefig(os.path.join(dirpath, column))
            plt.close()

    dirpath = os.path.splitext(epoch_filepath)[0]
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    df2 = pd.read_csv(epoch_filepath, sep=";", skiprows=2)
    if df2.shape[0] > 0:
        for column in df2.columns:
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.set(xlabel="Epoch", ylabel=column)
            pl = df2.plot(y=column,
                          use_index=True,
                          ax=ax,
                          title="{} values for each epoch".format(column),
                          legend=False)
            plt.savefig(os.path.join(dirpath, column))
            plt.close()


def check_class_distribution(df, csv):
    filename = csv.rsplit(os.path.sep)[-1]
    counts = []
    if csv == cfg.weak:
        all_configurations = df["event_labels"].value_counts()
        for cl in cfg.classes:
            counts.append(df.event_labels.str.count(cl).sum())
    else:
        all_configurations = df["event_label"].value_counts()
        for cl in cfg.classes:
            counts.append(df.event_label.str.count(cl).sum())

    all_configurations.to_csv(os.path.join(cfg.workspace, cfg.features, "class_count", "all" + filename), header=True)
    occurances = pd.Series(counts, index=cfg.classes)
    occurances.to_csv(os.path.join(cfg.workspace, cfg.features, "class_count", filename), header=True)

