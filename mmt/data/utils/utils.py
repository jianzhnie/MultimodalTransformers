'''
Author: jianzhnie
Date: 2021-11-12 15:42:02
LastEditTime: 2022-02-24 16:24:14
LastEditors: jianzhnie
Description:

'''
import types


def convert_to_func(container_arg):
    """convert container_arg to function that returns True if an element is in
    container_arg."""
    if container_arg is None:
        return lambda df, x: False
    if not isinstance(container_arg, types.FunctionType):
        assert type(container_arg) is list or type(container_arg) is set
        return lambda df, x: x in container_arg
    else:
        return container_arg


def agg_text_columns_func(empty_row_values, replace_text, texts):
    """replace empty texts or remove empty text str from a list of text str."""
    processed_texts = []
    for text in texts.astype('str'):
        if text not in empty_row_values:
            processed_texts.append(text)
        else:
            if replace_text is not None:
                processed_texts.append(replace_text)
    return processed_texts


def get_matching_cols(df, col_match_func):
    return [c for c in df.columns if col_match_func(df, c)]


def normalize_numerical_feats(numerical_feats, transformer=None):
    if numerical_feats is None or transformer is None:
        return numerical_feats
    return transformer.transform(numerical_feats)


def load_num_feats(df, num_bool_func):
    num_cols = get_matching_cols(df, num_bool_func)
    df = df.copy()
    df[num_cols] = df[num_cols].astype(float)
    df[num_cols] = df[num_cols].fillna(
        dict(df[num_cols].median()), inplace=False)
    if len(num_cols) == 0:
        return None
    return df[num_cols].values
