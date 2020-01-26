import numpy as np
import pandas as pd

def sort_weak_df(weak_df):
    # shuffle first
    weak_df = weak_df.sample(frac=1)
    # sort by classes per file counts
    weak_df['event_labels_count'] = weak_df['event_labels'].apply(lambda x: x.count(','))
    weak_df = weak_df.sort_values(by='event_labels_count', ascending=True)
    # bring df back to original state
    weak_df = weak_df.reset_index(drop=True)
    # weak_df = weak_df.drop(columns='event_labels_count')
    return weak_df


def sort_synthetic_df(synthetic_df):
    # shuffle first
    synthetic_df = synthetic_df.sample(frac=1)
    # sort by classes per file counts
    label_counts_df = synthetic_df[['filename', 'event_label']].groupby('filename').count().rename(
        columns={"event_label": "event_labels_count"})
    synthetic_df = synthetic_df.join(label_counts_df, on='filename', how='outer')
    # bring df back to original state
    synthetic_df = synthetic_df.sort_values(by='event_labels_count', ascending=True)
    # synthetic_df = synthetic_df.drop(columns='event_labels_count')
    return synthetic_df


def sort_synthetic_df_by_events_overlap(synthetic_df, return_subset=None):
    # SPAGHETTI ITALIANO
    overlap_df = synthetic_df.sort_values(by=['filename', 'onset', 'offset']).reset_index(drop=True)

    overlap_df['prev_filename'] = overlap_df.shift(1)['filename']

    overlap_df['space_between_segments'] = overlap_df['onset'] - overlap_df.shift(1)['offset']
    overlap_df['space_between_offsets'] = overlap_df['offset'] - overlap_df.shift(1)['offset']

    overlap_df['segment_width'] = overlap_df['offset'] - overlap_df['onset']
    overlap_df['segments_ratio'] = overlap_df['segment_width'] / overlap_df.shift(1)['segment_width']

    overlap_df.loc[overlap_df['filename'] != overlap_df['prev_filename'], 'space_between_segments'] = np.nan
    overlap_df.loc[overlap_df['filename'] != overlap_df['prev_filename'], 'space_between_offsets'] = np.nan

    # assign 0, 1, 2 numbers for separate cases for convenience :)
    overlap_df['overlaps'] = 0
    overlap_df.loc[overlap_df['space_between_segments'] < 0, 'overlaps'] = 1
    overlap_df.loc[overlap_df['space_between_offsets'] < 0, 'overlaps'] = 2

    overlap_df.loc[(overlap_df.shift(-1)['overlaps'] == 2) & (
            overlap_df['filename'] == overlap_df['prev_filename']), 'overlaps'] = 2

    # 1st case: no overlapping
    no_overlap = overlap_df[overlap_df['overlaps'] == 0]
    no_overlap = no_overlap.sample(frac=1)
    if return_subset == 'low':
        return no_overlap.reset_index()

    # 2nd case: one event overlaps another
    medium_overlap = overlap_df[overlap_df['overlaps'] == 1]
    medium_overlap = medium_overlap.sort_values(by='space_between_segments', ascending=False, na_position='first')
    if return_subset == 'medium':
        return pd.concat([no_overlap, medium_overlap]).sample(frac=1).reset_index()

    # 3rd case: there is an event inside another
    high_overlap = overlap_df[overlap_df['overlaps'] == 2]
    high_overlap.loc[(high_overlap['segments_ratio'] > 1.0)
                     & (high_overlap['filename'] == high_overlap['prev_filename']), 'segments_ratio'] = \
        high_overlap.shift(-1)['segments_ratio']

    # select original columns only
    no_overlap = no_overlap[synthetic_df.columns]
    medium_overlap = medium_overlap[synthetic_df.columns]
    high_overlap = high_overlap[synthetic_df.columns]

    result = pd.concat([no_overlap, medium_overlap, high_overlap]).reset_index()
    if return_subset == 'high':
        return result.sample(frac=1).reset_index()
    else:
        return result

