# -*- coding: utf-8 -*-
#########################################################################
# This file is derived from Curious AI/mean-teacher, under the Creative Commons Attribution-NonCommercial
# Copyright Nicolas Turpault, Romain Serizel, Justin Salamon, Ankit Parag Shah, 2019, v1.0
# This software is distributed under the terms of the License MIT
#########################################################################

import argparse
import os
import time
import random

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import nn

from utils import ramps
from DatasetDcase2019Task4 import DatasetDcase2019Task4
from DataLoad import DataLoadDf, ConcatDataset, MultiStreamBatchSampler
from utils.Scaler import Scaler
from TestModel import test_model
from feature_extractor import extract_features_from_meta
from evaluation_measures import get_f_measure_by_class, get_predictions, audio_tagging_results, compute_strong_metrics
from models.CRNN import CRNN
import config as cfg
from utils.utils import ManyHotEncoder, create_folder, SaveBest, to_cuda_if_available, weights_init, \
    get_transforms, AverageMeterSet
from utils.Logger import LOG
from utils.analysis import check_class_distribution, generate_graphs
from datetime import datetime
from cc_learning import sort_synthetic_df, sort_synthetic_df_by_events_overlap, sort_weak_df


def adjust_learning_rate(optimizer, rampup_value, rampdown_value):
    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = rampup_value * rampdown_value * cfg.max_learning_rate
    beta1 = rampdown_value * cfg.beta1_before_rampdown + (1. - rampdown_value) * cfg.beta1_after_rampdown
    beta2 = (1. - rampup_value) * cfg.beta2_during_rampdup + rampup_value * cfg.beta2_after_rampup
    weight_decay = (1 - rampup_value) * cfg.weight_decay_during_rampup + cfg.weight_decay_after_rampup * rampup_value

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['betas'] = (beta1, beta2)
        param_group['weight_decay'] = weight_decay


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(train_loader, model, optimizer, epoch, ema_model=None, weak_mask=None, strong_mask=None):
    """ One epoch of a Mean Teacher model
    :param train_loader: torch.utils.data.DataLoader, iterator of training batches for an epoch.
    Should return 3 values: teacher input, student input, labels
    :param model: torch.Module, model to be trained, should return a weak and strong prediction
    :param optimizer: torch.Module, optimizer used to train the model
    :param epoch: int, the current epoch of training
    :param ema_model: torch.Module, student model, should return a weak and strong prediction
    :param weak_mask: mask the batch to get only the weak labeled data (used to calculate the loss)
    :param strong_mask: mask the batch to get only the strong labeled data (used to calcultate the loss)
    """
    class_criterion = nn.BCELoss()
    consistency_criterion_strong = nn.MSELoss()
    [class_criterion, consistency_criterion_strong] = to_cuda_if_available(
        [class_criterion, consistency_criterion_strong])

    meters = AverageMeterSet()

    LOG.debug("Nb batches: {}".format(len(train_loader)))
    start = time.time()

    rampup_length = len(train_loader) * cfg.n_epoch // 2
    for i, (batch_input, ema_batch_input, target) in enumerate(train_loader):
        global_step = epoch * len(train_loader) + i
        if global_step < rampup_length:
            rampup_value = ramps.sigmoid_rampup(global_step, rampup_length)
        else:
            rampup_value = 1.0

        # Todo check if this improves the performance
        # adjust_learning_rate(optimizer, rampup_value, rampdown_value)
        meters.update('lr', optimizer.param_groups[0]['lr'])

        [batch_input, ema_batch_input, target] = to_cuda_if_available([batch_input, ema_batch_input, target])
        LOG.debug(batch_input.mean())
        # Outputs
        strong_pred_ema, weak_pred_ema = ema_model(ema_batch_input)
        strong_pred_ema = strong_pred_ema.detach()
        weak_pred_ema = weak_pred_ema.detach()

        strong_pred, weak_pred = model(batch_input)
        loss = None
        # Weak BCE Loss
        # Take the max in the time axis
        target_weak = target.max(-2)[0]
        if weak_mask is not None:
            weak_class_loss = class_criterion(weak_pred[weak_mask], target_weak[weak_mask])
            ema_class_loss = class_criterion(weak_pred_ema[weak_mask], target_weak[weak_mask])

            if i == 0:
                LOG.debug("target: {}".format(target.mean(-2)))
                LOG.debug("Target_weak: {}".format(target_weak))
                LOG.debug("Target_weak mask: {}".format(target_weak[weak_mask]))
                LOG.debug(weak_class_loss)
                LOG.debug("rampup_value: {}".format(rampup_value))
            meters.update('weak_class_loss', weak_class_loss.item())

            meters.update('Weak EMA loss', ema_class_loss.item())

            loss = weak_class_loss

        # Strong BCE loss
        if strong_mask is not None:
            strong_class_loss = class_criterion(strong_pred[strong_mask], target[strong_mask])
            meters.update('Strong loss', strong_class_loss.item())

            strong_ema_class_loss = class_criterion(strong_pred_ema[strong_mask], target[strong_mask])
            meters.update('Strong EMA loss', strong_ema_class_loss.item())
            if loss is not None:
                loss += strong_class_loss
            else:
                loss = strong_class_loss

        # Teacher-student consistency cost
        if ema_model is not None:

            consistency_cost = cfg.max_consistency_cost * rampup_value
            meters.update('Consistency weight', consistency_cost)
            # Take only the consistence with weak and unlabel
            consistency_loss_strong = consistency_cost * consistency_criterion_strong(strong_pred,
                                                                                      strong_pred_ema)
            meters.update('Consistency strong', consistency_loss_strong.item())
            if loss is not None:
                loss += consistency_loss_strong
            else:
                loss = consistency_loss_strong

            meters.update('Consistency weight', consistency_cost)
            # Take only the consistence with weak and unlabel
            consistency_loss_weak = consistency_cost * consistency_criterion_strong(weak_pred, weak_pred_ema)
            meters.update('Consistency weak', consistency_loss_weak.item())
            if loss is not None:
                loss += consistency_loss_weak
            else:
                loss = consistency_loss_weak

        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())
        assert not loss.item() < 0, 'Loss problem, cannot be negative'
        meters.update('Loss', loss.item())

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        if ema_model is not None:
            update_ema_variables(model, ema_model, 0.999, global_step)

    epoch_time = time.time() - start

    LOG.info(
        'Epoch: {}\t'
        'Time {:.2f}\t'
        '{meters}'.format(
            epoch, epoch_time, meters=meters))

    return meters


def get_metrics_result_list(event_metric):
    precision = event_metric['Ntp'] / (event_metric['Ntp'] + event_metric['Nfp']) if event_metric['Ntp'] + event_metric[
        'Nfp'] > 0 else 0.0
    recall = event_metric['Ntp'] / (event_metric['Ntp'] + event_metric['Nfn']) if event_metric['Ntp'] + event_metric[
        'Nfn'] > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
    acc = (event_metric['Ntp'] + event_metric['Ntn']) / event_metric['Nref'] if event_metric['Nref'] > 0 else 0.0
    results = [event_metric['Nref'], f1, precision, recall, acc]
    return results


def construct_training_data(epoch_num, weak_df, synth_df, shuffle, scaler=None):
    transforms = get_transforms(cfg.max_frames)

    weak_data = DataLoadDf(weak_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df, transform=transforms)
    synth_data = DataLoadDf(synth_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df, transform=transforms)

    list_dataset = [weak_data, synth_data]
    if cfg.weak_part_size and cfg.strong_part_size:
        batch_sizes = [cfg.weak_part_size, cfg.strong_part_size]
        strong_mask = slice(cfg.weak_part_size, cfg.batch_size)
    else:
        batch_sizes = [cfg.batch_size // 2, cfg.batch_size // 2]
        strong_mask = slice(cfg.batch_size // 2, cfg.batch_size)

    # Assume weak data is always the first one
    weak_mask = slice(batch_sizes[0])

    if not scaler:
        scaler = Scaler()
        scaler.calculate_scaler(ConcatDataset(list_dataset))
        LOG.debug(scaler.mean_)

    transforms = get_transforms(cfg.max_frames, scaler, augment_type="noise")
    for i in range(len(list_dataset)):
        list_dataset[i].set_transform(transforms)

    concat_dataset = ConcatDataset(list_dataset)
    sampler = MultiStreamBatchSampler(concat_dataset,
                                      batch_sizes=batch_sizes,
                                      shuffle=shuffle)
    training_data = DataLoader(concat_dataset, batch_sampler=sampler)
    return training_data, weak_mask, strong_mask


def add_parser_arguments(parser):
    # Configuration: data subset, model path etc
    parser.add_argument("-s", '--subpart_data', type=int, default=None, dest="subpart_data",
                        help="Number of files to be used. Useful when testing on small number of files.")
    parser.add_argument("-m", '--model_path', type=str, default=None, dest="model_path",
                        help="Path of the model to initialize with.")
    parser.add_argument("-d", '--no_download', dest='no_download', action='store_true', default=False,
                        help="Not downloading data based on csv files.")

    # Learning config: whether to sort data, use unlabeled samples
    parser.add_argument("-o", '--ordered', dest='sort', action='store_true', default=False,
                        help="Sorting data so as to perform Curriculum Learning.")
    parser.add_argument("-u", '--skip_unlabeled', dest='skip_unlabeled', action='store_true', default=False,
                        help="Skipping large unlabeled audio dataset.")
    parser.add_argument("-f", '--flatness', dest='use_flatness', action='store_true', default=False,
                        help="Sort audio files according to spectral flatness.")
    parser.add_argument('--snr', dest='use_snr', action='store_true', default=False,
                        help="Sort audio files according to signal-to-noise ratio.")
    parser.add_argument('--super_snr', dest='use_super_snr', action='store_true', default=False,
                        help="Sort audio files according to signal-to-noise ratio AND grow the dataset based upon that.")
    parser.add_argument('--sort_overlap', dest='sort_overlap', action='store_true', default=False,
                        help="Sort weak data by class counts, and synthetic strong data by overlapping events difficulty.")
    parser.add_argument('--sort_super_overlap', dest='sort_super_overlap', action='store_true', default=False,
                        help="Sort weak data by class counts, and synthetic strong data by overlapping events difficulty AND grow the dataset based on that.")
    parser.add_argument('--sort_class', dest='sort_class', action='store_true', default=False,
                        help="Sort by class difficulties, based on initial training performance using shuffled data.")

    # Neural-network related
    parser.add_argument("-c", '--freeze_cnn', dest='freeze_cnn', action='store_true', default=False,
                        help="Freezing loaded CNN weights.")
    parser.add_argument("-r", '--freeze_rnn', dest='freeze_rnn', action='store_true', default=False,
                        help="Freezing loaded RNN weights.")

    parser.add_argument('--skip_cnn', dest='skip_cnn', action='store_true', default=False,
                        help="Not loading CNN layers weights.")
    parser.add_argument('--skip_rnn', dest='skip_rnn', action='store_true', default=False,
                        help="Not loading RNN layers weights.")
    parser.add_argument('--skip_dense', dest='skip_dense', action='store_true', default=False,
                        help="Not loading Dense layers weights.")
    return parser


SIMPLE_CLASSES = ['Speech', 'Alarm_bell_ringing', 'Cat', 'Blender', 'Electric_shaver_toothbrush']
HARD_CLASSES = ['Dog', 'Frying', 'Dishes', 'Vacuum_cleaner', 'Running_water']


def select_classes_for_epoch_weak(epoch, train_weak_df, th):
    epoch = epoch + 1  # relative to numbers of elements in arrays, original numbering 0, 1, 2...
    allowed_classes = SIMPLE_CLASSES.copy()
    if epoch > th:
        idx = 0
        while len(allowed_classes) < epoch and idx < len(HARD_CLASSES):
            allowed_classes.append(HARD_CLASSES[idx])
            idx += 1
    df = train_weak_df[train_weak_df['event_labels'].apply(
        lambda x: all([label in allowed_classes for label in x.split(',')])
    )]
    return df.sample(frac=1)


def select_classes_for_epoch_synth(epoch, train_synth_df, th):
    epoch = epoch + 1  # relative to numbers of elements in arrays, original numbering 0, 1, 2...
    allowed_classes = SIMPLE_CLASSES.copy()
    if epoch > th:
        idx = 0
        while len(allowed_classes) < epoch and idx < len(HARD_CLASSES):
            allowed_classes.append(HARD_CLASSES[idx])
            idx += 1
    df = train_synth_df[train_synth_df['event_label'].apply(lambda x: x in allowed_classes)]
    return df.sample(frac=1)

def select_for_super_audio_feature(epoch, train_weak_df, train_synth_df, perc_25, perc_50, feature="SNR", th=10):
    synth_weak_ratio = train_synth_df.shape[0] // train_weak_df.shape[0]
    epoch = epoch + 1
    if epoch // th == 0:
        synth_df = train_synth_df[train_synth_df[feature] >= perc_50]
        weak_df = train_weak_df.head(synth_df.shape[0] // synth_weak_ratio)
    elif epoch // th == 1:
        synth_df = train_synth_df[train_synth_df[feature] >= perc_25]
        weak_df = train_weak_df.head(synth_df.shape[0] // synth_weak_ratio)
    else:
        weak_df, synth_df = train_weak_df, train_synth_df
    return weak_df.sample(frac=1), synth_df.sample(frac=1)

def select_for_super_overlap(epoch, train_weak_df, train_synth_df, th=10):
    synth_weak_ratio = train_synth_df.shape[0] // train_weak_df.shape[0]
    epoch = epoch + 1
    if epoch // th == 0:
        synth_df = sort_synthetic_df_by_events_overlap(train_synth_df, return_subset='low')
        weak_df = train_weak_df.head(synth_df.shape[0] // synth_weak_ratio)
    elif epoch // th == 1:
        synth_df = sort_synthetic_df_by_events_overlap(train_synth_df, return_subset='medium')
        weak_df = train_weak_df.head(synth_df.shape[0] // synth_weak_ratio)
    else:
        synth_df = sort_synthetic_df_by_events_overlap(train_synth_df, return_subset='high')
        weak_df = train_weak_df.head(synth_df.shape[0] // synth_weak_ratio)
    return weak_df, synth_df

if __name__ == '__main__':
    
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    random.seed(0)

    LOG.info("MEAN TEACHER")

    parser = argparse.ArgumentParser(description="")
    parser = add_parser_arguments(parser)
    f_args = parser.parse_args()

    reduced_number_of_data = f_args.subpart_data
    model_path = f_args.model_path
    download = not f_args.no_download
    skip_unlabeled = f_args.skip_unlabeled

    # Only one of the below can be used at a time
    sort = f_args.sort
    sort_overlap = f_args.sort_overlap
    sort_super_overlap = f_args.sort_super_overlap
    sort_class = f_args.sort_class
    use_flatness = f_args.use_flatness
    use_snr = f_args.use_snr
    use_super_snr = f_args.use_super_snr
    assert sum([
        sort, sort_overlap, sort_super_overlap, sort_class,
        use_flatness, use_snr, use_super_snr
    ]) <= 1
    do_shuffle = not (
            sort or sort_overlap or sort_class or sort_super_overlap
            or use_flatness or use_snr or use_super_snr
    )
    growing_dataset = sort_class or sort_super_overlap or use_super_snr

    freeze_cnn = f_args.freeze_cnn
    freeze_rnn = f_args.freeze_rnn

    skip_cnn = f_args.skip_cnn
    skip_rnn = f_args.skip_rnn
    skip_dense = f_args.skip_dense

    LOG.info("subpart_data = {}".format(reduced_number_of_data))
    LOG.info("Using pre-trained model = {}".format(model_path))
    LOG.info("Downloading data = {}".format(download))
    LOG.info("Sorting = {}".format(sort))
    LOG.info("Use unlabeled = {}".format(not skip_unlabeled))
    LOG.info("Sort according to spectral flatness = {}".format(use_flatness))
    LOG.info("Sort according to snr = {}".format(use_snr))
    LOG.info("Use super snr = {}".format(use_super_snr))

    add_dir_model_name = "_with_synthetic"

    if model_path:
        state = torch.load(model_path, map_location="cpu")
    else:
        state = None

    fname_timestamp = datetime.now().strftime("%d-%m_%H-%M")

    res_suffix = 'init-cnn_' if model_path else '0init_'
    if sort:
        res_suffix += 'cc-sort' # class count sort
    elif sort_overlap:
        res_suffix += 'ov-sort' # overlap sort
    elif sort_super_overlap:
        res_suffix += 'ov-sort++'
    elif sort_class:
        res_suffix += 'cd-sort' # class difficulty sort
    elif use_flatness:
        res_suffix += 'flat-sort'
    elif use_snr:
        res_suffix += 'snr-sort'
    elif use_super_snr:
        res_suffix += 'snr-sort++'

    res_filename = res_suffix + "_" + fname_timestamp + ".csv"

    LOG.info(f"Saving results using {res_filename}")
    if not os.path.exists(os.path.join('..', 'results')):
        os.makedirs(os.path.join('..', 'results'))
        LOG.info(f"Creating 'results' directory...")
    res_fullpath = os.path.join('..', 'results', res_filename)
    res_columns = ['weak_loss', 'strong_loss', 'consistency_weak_loss', 'consistency_strong_loss', 'loss']
    map_res_columns = {
        'weak_loss': 'weak_class_loss',
        'strong_loss': 'Strong loss',
        'consistency_weak_loss': 'Consistency weak',
        'consistency_strong_loss': 'Consistency strong',
        'loss': 'Loss'
    }
    with open(res_fullpath, 'w') as file:
        file.write(str(f_args) + "\n\n")  # dump f_args, just in case
        file.write(';'.join([*res_columns, "global_valid"]) + "\n")
    print(str(f_args) + "\n")

    res_classes_filename = "class_" + res_filename
    res_classes_columns = ['class_name', 'weak-F1', 'Nref', 'F', 'Pre', 'Rec', 'Acc', 'Nref_Seg', 'F_Seg', 'Pre_Seg',
                           'Rec_Seg', 'Acc_Seg']
    res_classes_fullpath = os.path.join('..', 'results', res_classes_filename)
    with open(res_classes_fullpath, 'w') as file:
        file.write(';'.join(res_classes_columns) + "\n")

    store_dir = os.path.join("stored_data", f"{res_suffix}_{fname_timestamp}_MeanTeacher")
    saved_model_dir = os.path.join(store_dir, "model")
    saved_pred_dir = os.path.join(store_dir, "predictions")
    for folder_name in [store_dir, saved_model_dir, saved_pred_dir]:
        create_folder(folder_name)

    if state:
        pooling_time_ratio = state["pooling_time_ratio"]
    else:
        pooling_time_ratio = cfg.pooling_time_ratio  # --> Be careful, it depends of the model time axis pooling
    # ##############
    # DATA
    # ##############

    transforms = get_transforms(cfg.max_frames)
    classes = cfg.classes
    if state:
        many_hot_encoder = ManyHotEncoder.load_state_dict(state["many_hot_encoder"])
    else:
        many_hot_encoder = ManyHotEncoder(classes, n_frames=cfg.max_frames // pooling_time_ratio)

    dataset = DatasetDcase2019Task4(cfg.workspace,
                                    base_feature_dir=os.path.join(cfg.workspace, "dataset", "features"),
                                    save_log_feature=False)

    if use_flatness or use_snr or use_super_snr:
        if not os.path.isfile(os.path.join(cfg.workspace, cfg.weak_f)):
            extract_features_from_meta(os.path.join(cfg.workspace, cfg.weak))
        if not os.path.isfile(os.path.join(cfg.workspace, cfg.synthetic_f)):
            extract_features_from_meta(os.path.join(cfg.workspace, cfg.synthetic))
        if not os.path.isfile(os.path.join(cfg.workspace, cfg.unlabel_f)):
            extract_features_from_meta(os.path.join(cfg.workspace, cfg.unlabel))
        weak_path = cfg.weak_f
        synthetic_path = cfg.synthetic_f
        unlabel_path = cfg.unlabel_f
    else:
        weak_path = cfg.weak
        synthetic_path = cfg.synthetic
        unlabel_path = cfg.unlabel

    weak_df = dataset.initialize_and_get_df(weak_path, reduced_number_of_data, download=download)
    if not skip_unlabeled:
        unlabel_df = dataset.initialize_and_get_df(unlabel_path, reduced_number_of_data, download=download)

    # Event if synthetic not used for training, used on validation purpose
    synthetic_df = dataset.initialize_and_get_df(synthetic_path, reduced_number_of_data, download=download)
    validation_df = dataset.initialize_and_get_df(cfg.validation, reduced_number_of_data, download=download)

    # Divide weak in train and valid
    train_weak_df = weak_df.sample(frac=0.8, random_state=26)
    valid_weak_df = weak_df.drop(train_weak_df.index).reset_index(drop=True)
    train_weak_df = train_weak_df.reset_index(drop=True)
    LOG.debug(valid_weak_df.event_labels.value_counts())

    # Divide synthetic in train and valid
    filenames_train = synthetic_df.filename.drop_duplicates().sample(frac=0.8, random_state=26)
    train_synth_df = synthetic_df[synthetic_df.filename.isin(filenames_train)]
    valid_synth_df = synthetic_df.drop(train_synth_df.index).reset_index(drop=True)

    # Put train_synth in frames so many_hot_encoder can work.
    #  Not doing it for valid, because not using labels (when prediction) and event based metric expect sec.
    train_synth_df.onset = train_synth_df.onset * cfg.sample_rate // cfg.hop_length // pooling_time_ratio
    train_synth_df.offset = train_synth_df.offset * cfg.sample_rate // cfg.hop_length // pooling_time_ratio
    LOG.debug(valid_synth_df.event_label.value_counts())

    # check_class_distribution(weak_df, cfg.weak)
    # check_class_distribution(synthetic_df, cfg.synthetic)
    # check_class_distribution(validation_df, cfg.validation)

    if sort:
        train_weak_df = sort_weak_df(train_weak_df)
        train_synth_df = sort_synthetic_df(train_synth_df)

    if sort_overlap or sort_super_overlap:
        train_weak_df = sort_weak_df(train_weak_df)
        train_synth_df = sort_synthetic_df_by_events_overlap(train_synth_df)

    if use_flatness:
        # sort ascending - the values are from -inf to 0 where -inf is the perfect sound and 0 is a perfect white noise
        train_weak_df = train_weak_df.sort_values(by=["Spectral flatness"])
        train_synth_df = train_synth_df.sort_values(by=["Spectral flatness"])

    if use_snr or use_super_snr:
        # sort descending - the values above 0 dB mean more signal than noise
        train_weak_df = train_weak_df.sort_values(by=["SNR"], ascending=False)
        train_synth_df = train_synth_df.sort_values(by=["SNR"], ascending=False)
        if use_super_snr:
            snr_stats = train_synth_df["SNR"].describe()
            snr_perc_25, snr_perc_50 = snr_stats['25%'], snr_stats['50%']

    train_weak_data = DataLoadDf(train_weak_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                                 transform=transforms)
    if not skip_unlabeled:
        unlabel_data = DataLoadDf(unlabel_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                                  transform=transforms)
    train_synth_data = DataLoadDf(train_synth_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                                  transform=transforms)

    if skip_unlabeled:
        list_dataset = [train_weak_data, train_synth_data]
        if cfg.weak_part_size and cfg.strong_part_size:
            batch_sizes = [cfg.weak_part_size, cfg.strong_part_size]
            strong_mask = slice(cfg.weak_part_size, cfg.batch_size)
        else:
            batch_sizes = [cfg.batch_size // 2, cfg.batch_size // 2]
            strong_mask = slice(cfg.batch_size // 2, cfg.batch_size)
    else:
        list_dataset = [train_weak_data, unlabel_data, train_synth_data]
        batch_sizes = [cfg.batch_size // 4, cfg.batch_size // 2, cfg.batch_size // 4]
        strong_mask = slice(cfg.batch_size // 4 + cfg.batch_size // 2, cfg.batch_size)

    # Assume weak data is always the first one
    weak_mask = slice(batch_sizes[0])

    scaler = Scaler()
    if state:
        scaler.load_state_dict(state["scaler"])
    else:
        scaler.calculate_scaler(ConcatDataset(list_dataset))
    LOG.debug(scaler.mean_)

    transforms = get_transforms(cfg.max_frames, scaler, augment_type="noise")
    for i in range(len(list_dataset)):
        list_dataset[i].set_transform(transforms)

    concat_dataset = ConcatDataset(list_dataset)
    sampler = MultiStreamBatchSampler(concat_dataset,
                                      batch_sizes=batch_sizes,
                                      shuffle=do_shuffle)

    training_data = DataLoader(concat_dataset, batch_sampler=sampler)

    transforms_valid = get_transforms(cfg.max_frames, scaler=scaler)
    valid_synth_data = DataLoadDf(valid_synth_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                                  transform=transforms_valid)
    valid_weak_data = DataLoadDf(valid_weak_df, dataset.get_feature_file, many_hot_encoder.encode_weak,
                                 transform=transforms_valid)

    # Eval 2018
    eval_2018_df = dataset.initialize_and_get_df(cfg.eval2018, reduced_number_of_data, download=download)
    eval_2018 = DataLoadDf(eval_2018_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                           transform=transforms_valid)

    # ##############
    # Model
    # ##############

    if state:
        crnn_kwargs = state["model"]["kwargs"]
        crnn_kwargs['attention'] = True
    else:
        crnn_kwargs = cfg.crnn_kwargs

    crnn = CRNN(**crnn_kwargs)
    crnn_ema = CRNN(**crnn_kwargs)

    crnn.apply(weights_init)
    crnn_ema.apply(weights_init)

    if state:
        # load state into models
        crnn.load(parameters=state["model"]["state_dict"], load_cnn=(not skip_cnn), load_rnn=(not skip_rnn),
                  load_dense=(not skip_dense))
        crnn_ema.load(parameters=state["model_ema"]["state_dict"], load_cnn=(not skip_cnn), load_rnn=(not skip_rnn),
                      load_dense=(not skip_dense))
        LOG.info("Model loaded at epoch: {}".format(state["epoch"]))

    for param in crnn_ema.parameters():
        param.detach_()

    crnn, crnn_ema = to_cuda_if_available([crnn, crnn_ema])

    if not state:
        optim_kwargs = {"lr": 0.001, "betas": (0.9, 0.999)}
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, crnn.parameters()), **optim_kwargs)
    else:
        optim_kwargs = state['optimizer']['kwargs']
        if state['optimizer']['name'] == 'Adam':
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, crnn.parameters()), **optim_kwargs)
            optimizer.load_state_dict(state['optimizer']['state_dict'])
        else:
            NotImplementedError("Only models trained with Adam optimizer supported for now")

    if not state:
        state = {
            'model': {"name": crnn.__class__.__name__,
                      'args': '',
                      "kwargs": crnn_kwargs,
                      'state_dict': crnn.state_dict()},
            'model_ema': {"name": crnn_ema.__class__.__name__,
                          'args': '',
                          "kwargs": crnn_kwargs,
                          'state_dict': crnn_ema.state_dict()},
            'optimizer': {"name": optimizer.__class__.__name__,
                          'args': '',
                          "kwargs": optim_kwargs,
                          'state_dict': optimizer.state_dict()},
            "pooling_time_ratio": pooling_time_ratio,
            "scaler": scaler.state_dict(),
            "many_hot_encoder": many_hot_encoder.state_dict()
        }

    save_best_cb = SaveBest("sup")

    if state:
        if freeze_cnn:
            crnn.freeze_cnn()
            crnn_ema.freeze_cnn()
        if freeze_rnn:
            crnn.freeze_rnn()
            crnn_ema.freeze_rnn()

    # ##############
    # Train
    # ##############
    for epoch in range(cfg.n_epoch):
        LOG.info(f"\n\n\t>>> >>> EPOCH {epoch}\n")
        crnn = crnn.train()
        crnn_ema = crnn_ema.train()

        crnn, crnn_ema = to_cuda_if_available([crnn, crnn_ema])

        if growing_dataset:
            # those modes require growing dataset
            if sort_class:
                weak_df = select_classes_for_epoch_weak(epoch, train_weak_df, 5)
                synth_df = select_classes_for_epoch_synth(epoch, train_synth_df, 5)
            if use_super_snr:
                weak_df, synth_df = select_for_super_audio_feature(epoch, train_weak_df, train_synth_df,
                                                                   snr_perc_25, snr_perc_50, feature="SNR",
                                                                   th=10)
            if sort_super_overlap:
                weak_df, synth_df = select_for_super_overlap(epoch, train_weak_df, train_synth_df, th=10)

            print(f"Weak DF: {weak_df.shape[0]}, Synth DF: {synth_df.shape[0]}")
            training_data, weak_mask, strong_mask = construct_training_data(epoch, weak_df, synth_df,
                                                                            shuffle=do_shuffle)

        meters = train(training_data, crnn, optimizer, epoch, ema_model=crnn_ema, weak_mask=weak_mask,
                       strong_mask=strong_mask)
        overall_results = [meters[map_res_columns[m]].val for m in res_columns]

        crnn = crnn.eval()

        LOG.info("\n ### Valid weak metric ### \n")
        weak_metric = get_f_measure_by_class(crnn, len(classes),
                                             DataLoader(valid_weak_data, batch_size=cfg.batch_size))
        per_class_results = dict(zip(many_hot_encoder.labels, weak_metric))
        per_class_results = {k: [per_class_results[k]] for k in per_class_results.keys()}

        LOG.info("Weak F1-score per class: \n {}".format(pd.DataFrame(weak_metric * 100, many_hot_encoder.labels)))
        LOG.info("Weak F1-score macro averaged: {}".format(np.mean(weak_metric)))

        LOG.info("\n ### Valid synthetic metric ### \n")
        predictions = get_predictions(crnn, valid_synth_data, many_hot_encoder.decode_strong, pooling_time_ratio,
                                      save_predictions=None)
        valid_events_metric, valid_segment_metric = compute_strong_metrics(predictions, valid_synth_df, log=True)

        for event in valid_events_metric.event_label_list:
            event_results = get_metrics_result_list(valid_events_metric.class_wise[event])
            per_class_results[event].extend(event_results)

        for event in valid_segment_metric.event_label_list:
            event_results = get_metrics_result_list(valid_segment_metric.class_wise[event])
            per_class_results[event].extend(event_results)

        with open(res_classes_fullpath, 'a') as file:
            for event in many_hot_encoder.labels:
                results = list(map(lambda s: "{:.3f}".format(s), per_class_results[event]))
                file.write(';'.join([event, *results, '\n']))
            file.write('\n')  # next epoch separator

        state['model']['state_dict'] = crnn.state_dict()
        state['model_ema']['state_dict'] = crnn_ema.state_dict()
        state['optimizer']['state_dict'] = optimizer.state_dict()
        state['epoch'] = epoch
        state['valid_metric'] = valid_events_metric.results()
        if cfg.checkpoint_epochs is not None and (epoch + 1) % cfg.checkpoint_epochs == 0:
            model_fname = os.path.join(saved_model_dir, "baseline_epoch_" + str(epoch))
            torch.save(state, model_fname)

        global_valid = valid_events_metric.results_class_wise_average_metrics()['f_measure']['f_measure']
        global_valid = global_valid + np.mean(weak_metric)

        overall_results.append(global_valid)
        overall_results = list(map(lambda s: "{:.3f}".format(s), overall_results))
        with open(res_fullpath, 'a') as file:
            file.write(';'.join(overall_results) + '\n')

        if cfg.save_best:
            if save_best_cb.apply(global_valid):
                model_fname = os.path.join(saved_model_dir, "baseline_best")
                torch.save(state, model_fname)

    if cfg.save_best:
        model_fname = os.path.join(saved_model_dir, "baseline_best")
        state = torch.load(model_fname)
        LOG.info("testing model: {}".format(model_fname))
    else:
        LOG.info("testing model of last epoch: {}".format(cfg.n_epoch))

    # ##############
    # Validation
    # ##############
    predicitons_fname = os.path.join(saved_pred_dir, "baseline_validation.csv")
    (metric_event, metric_segment), weak_metric, weak_labels = test_model(state, cfg.validation, reduced_number_of_data, predicitons_fname)

    with open(res_fullpath.replace('.csv', '_test.csv'), 'a') as file:
        file.write("METRIC EVENT\n")
        file.write(str(metric_event.results()))
        file.write("\nMETRIC SEGMENT\n")
        file.write(str(metric_segment.results()))
        file.write("\nWEAK METRIC\n")
        file.write("\nWeak F1-score per class: \n {}".format(pd.DataFrame(weak_metric * 100, many_hot_encoder.labels)))
        file.write("\nWeak F1-score macro averaged: {}".format(np.mean(weak_metric)))

    # Plot graph with a chosen metrics and graph of loss function
    generate_graphs(res_classes_fullpath, res_fullpath)
