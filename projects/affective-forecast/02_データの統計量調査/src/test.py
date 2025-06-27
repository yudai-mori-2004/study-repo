#!/usr/bin/env python3
from util.utils import load_h5_data, load_csv_data, load_multiple_h5_files

DATA_PATH = "/home/mori/projects/affective-forecast/datas"

def list_and_preview_h5_files(directory="sample", preview_shape=(2, 5)):
    """
    指定ディレクトリ内のすべての .h5 ファイルを走査し、
    各データセットの形状と冒頭の一部データを表示します。
    """
    file_path = "biometric_data/data_2_E4_act.h5"
    data=load_h5_data(f"{DATA_PATH}/{file_path}")
    print(data)

    timestamp_path = "meta_data/timestamp.csv"
    timestamp_data=load_csv_data(f"{DATA_PATH}/{timestamp_path}")
    print(timestamp_data[0].keys())


if __name__ == "__main__":
    list_and_preview_h5_files()
