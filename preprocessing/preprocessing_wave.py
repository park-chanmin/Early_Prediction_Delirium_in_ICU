import os
import glob
import time
import os.path as osp
from datetime import datetime
import numpy as np
import pandas as pd
from struct import unpack, pack
import multiprocessing
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import RobustScaler

import module.utils as cutils
from module.db_utils import create_df_from_ppg2


def decode_wavesample2(binary_wave_samples):
    # print(len(binary_wave_samples))
    data_array = []
    for i in range(len(binary_wave_samples) // 2):
        raw_sample = binary_wave_samples[i * 2 : (i + 1) * 2]
        sample = float(unpack("<H", raw_sample)[0])
        data_array.append(sample)
    # print("decode2", data_array)
    return data_array


def fix_decode(data_array):
    bytes_array = bytearray()
    for d in data_array:
        b = pack("<H", int(d))
        bytes_array += b
    return decode_wavesample2(bytes_array[1:])


def hjorth(wave):
    first_deriv = np.diff(wave)
    second_deriv = np.diff(wave, 2)

    var_zero = np.nanmean(wave**2)
    var_d1 = np.nanmean(first_deriv**2)
    var_d2 = np.nanmean(second_deriv**2)

    activity = var_zero
    morbidity = np.sqrt(var_d1 / var_zero)
    complexity = np.sqrt(var_d2 / var_d1) / morbidity

    return activity, morbidity, complexity


def extract_shape_feature(wave, front=None):
    if wave.ndim == 1:
        wave = wave.reshape(-1, 1)

    front = "" if front is None else front + "_"

    transformer = RobustScaler()
    scaled_wave = transformer.fit_transform(wave)
    scaled_wave = scaled_wave.squeeze()

    activity, morbidity, complexity = hjorth(scaled_wave)
    kurt = kurtosis(scaled_wave, nan_policy="omit")
    skewness = skew(scaled_wave, nan_policy="omit")
    if not (type(skewness) is float):
        skewness = float(skewness.data)
    return activity, morbidity, complexity, kurt, skewness


class WaveformPreprocessing:
    def __init__(self, wv_config={}, db_config_path="config/db_config.yaml"):
        """
        # 기준시점(camicu time)부터 4시간 이내 10분 간격으로 마지막 10초 추출
        wv_config : {"WAVE_DURATION_SECONDS": 10, "INPUT_INTERVAL_MINUTES":10, "INPUT_DURATION_LIMIT_HOURS":4, # TODO: 필요?
        "abbr_wave_name_list": ["ECGII", "pleth", "RESP"], "raw_wave_name_list": ["II", "Pleth", "Resp"]}
        """
        self.wv_config = wv_config
        self.db_config = (
            cutils.load_yaml(db_config_path) if osp.exists(db_config_path) else None
        )
        self.wave_list_df = None
        print("set wv_config", wv_config)

    def fetch_waveform_from_db(self, sr: pd.Series, excluding_param=None):
        """read one WAVE_DURATION_SECONDS unit from DB
        정확히 WAVE_DURATION_SECONDS 분량은 아니고 넉넉히 뽑아오고 transform_and_remove_noise_waveform에서 후처리로 자름

        Args:
            sr (pd.Series): one window time which have columns of session_id, event_timestamp

        Returns:
            pd.DataFrame: waveform data (before preprocessing)
                e.g.
                    stime                etime                param                  data
                   2022-12-24 11:34:24  2022-12-24 11:34:25   II   34.16,41.48,46.36,46.36,48.8,56.12,65.88,70.76...
        """
        db_query = f"""select * from wave DB"""
        raw_waveform = create_df_from_ppg2(db_query, self.db_config)
        raw_waveform["param"] = raw_waveform["param"].replace(
            dict(
                zip(
                    self.wv_config["raw_wave_name_list"],
                    self.wv_config["abbr_wave_name_list"],
                )
            )
        )

        # print(len(raw_waveform), "rows")
        # print(raw_waveform.head())
        return raw_waveform

    def get_demo_raw_waveform(self, sr, excluding_param=None):
        event_timestamp = sr["event_timestamp"]
        raw_waveform = pd.read_csv(
            "./sample_waves",
            parse_dates=["stime", "etime"],
        )
        raw_waveform = raw_waveform.query(f'session_id=="{sr["session_id"]}"')
        raw_waveform = raw_waveform[
            raw_waveform["stime"]
            >= (
                event_timestamp
                - pd.Timedelta(seconds=self.wv_config["WAVE_DURATION_SECONDS"] + 6)
            )
        ]
        raw_waveform = raw_waveform[raw_waveform["stime"] <= event_timestamp]
        
        raw_waveform = raw_waveform[raw_waveform["param"]!=excluding_param]
        raw_waveform["param"] = raw_waveform["param"].replace(
            dict(
                zip(
                    self.wv_config["raw_wave_name_list"],
                    self.wv_config["abbr_wave_name_list"],
                )
            )
        )
        raw_waveform = raw_waveform.drop(columns=["session_id"])

        # if len(raw_waveform) > 0:
            # print(sr)
            # print(len(raw_waveform), "rows")
            # print(raw_waveform.head())
        return raw_waveform

    def read_existing_file(self, sr, existing_data_dir, key='patient_id'):
        """
        저혈압 모델용 파형 데이터 대상이기 때문에 파일명 규칙이 다름
        """
        
        raw_waveform = pd.read_csv(
            file_path[0],
            parse_dates=["stime", "etime"],
            index_col=0
        )
        raw_waveform = raw_waveform[
            raw_waveform["stime"]
            >= (
                event_timestamp
                - pd.Timedelta(seconds=self.wv_config["WAVE_DURATION_SECONDS"] + 6)
            )
        ]
        raw_waveform = raw_waveform[raw_waveform["stime"] <= event_timestamp]
        if len(raw_waveform) > 0:
            print(len(raw_waveform), "rows")
            return True, raw_waveform
        else:
            return False, None

    def transform_and_remove_noise_waveform(self, raw_waveform, event_timestamp=None):
        """
        transform shape
        resampling
        event_timestamp 범위밖 삭제
        """

        transformed_dict = {}
        for wt in self.wv_config["abbr_wave_name_list"]:  # ["ECGII", "pleth", "RESP"]
            one_param_raw_waveform = raw_waveform.query(f"param=='{wt}'")

            if len(one_param_raw_waveform) == 0:
                transformed_dict[wt] = None
            else:
                tmp_list = []
                for i, row in one_param_raw_waveform.iterrows():
                    starttime = pd.to_datetime(row["stime"])
                    # endtime = pd.to_datetime(row["endtime"])
                    # II: 2559개*0.002초 (1/500hz), PPG: 639개*0.008초 (1/125hz)

                    data = eval(row["data"])
                    n_data = len(data)  # 2559
                    if (sum(data) / n_data) >= 10000:
                        data = fix_decode(data)  # 1개 값 빠짐 len(data)=2558
                    wv = pd.DataFrame(data, columns=["signal"])

                    diff = 10 / n_data
                    wv.index = [
                        pd.Timestamp.fromtimestamp(
                            starttime.timestamp() + (i * diff)
                        )
                        for i in range(n_data)
                    ][-len(data) :]

                    tmp_list.append(wv)
                one_param_df = pd.concat(tmp_list)

                ### resampling
                one_param_df = one_param_df.resample("0.004S").median()  # 1/250
                one_param_df = one_param_df.interpolate(method="time")

                # cut unnecessary time
                if event_timestamp is not None:
                    one_param_df = one_param_df[
                        one_param_df.index < str(event_timestamp)
                    ]

                ### rule based waveform quality control (exclude abnormality waves from dataset)
                if one_param_df["signal"].std() == 0:
                    print(event_timestamp)
                    print('one_param_df["signal"].std() == 0')
                    one_param_df = None
                elif len(one_param_df)==0:
                    one_param_df = None
                elif len(one_param_df) < 2048:
                    print(event_timestamp)
                    print(f"len(one_param_df):{len(one_param_df)} < 2048")
                # else:
                #     # print("normal wave")
                #     pass
                transformed_dict[wt] = one_param_df
        return transformed_dict

    def make_wave_list_df(
        self,
        wave_data_path,
        data_dir,
        operating_system="linux",
    ):
        """
        :param root_dir: base directory
        :param os: 'windows' , 'linux', 'mac' (default)
        :return: wave list dataframe [columns = 'intime', 'outtime', 'starttime', 'endtime', 'wave_type']
        """
        wave_list = []
        for root, dirs, files in os.walk(wave_data_path):
            wave_list.extend(glob.glob(f"{root}/*.feather"))
            wave_list_df = pd.DataFrame({"path": wave_list})

        if operating_system == "mac":
            ## macos에서는 파일명의 콜론이 자동으로 언더바로 바뀜
            wave_list_df = pd.concat(
                [
                    wave_list_df,
                    pd.DataFrame(
                        wave_list_df["path"].apply(lambda x: osp.basename(x)).to_list(),
                        columns=["filename"],
                    ),
                ],
                axis=1,
            )
            tmp = (
                wave_list_df["filename"].str.replace(".csv", "").str.split("_").tolist()
            )
            rslt = []
            for i in range(len(tmp)):
                rslt.append(self.Three_element_sum(tmp[i], 13))
            wave_list_df = pd.concat(
                [
                    wave_list_df,
                    pd.DataFrame(
                        rslt,
                        columns=[
                            "session_id",
                            "camicu_time",
                            "event_timestamp",
                            "wave_type",
                        ],
                    ),
                ],
                axis=1,
            )
            del tmp
            del rslt

            wave_list_df.to_csv(
                osp.join(osp.dirname(data_dir), f"wave_list_df.csv"), index=False
            )

            return wave_list_df

        else:
            wave_list_df = pd.concat(
                [
                    wave_list_df,
                    pd.DataFrame(
                        wave_list_df["path"].apply(lambda x: osp.basename(x)).to_list(),
                        columns=["filename"],
                    ),
                ],
                axis=1,
            )
            wave_list_df = pd.concat(
                [
                    wave_list_df,
                    pd.DataFrame(
                        wave_list_df["filename"]
                        .str.replace(".csv", "", regex=False)
                        .str.replace(".feather", "")
                        .str.split("_")
                        .to_list(),
                        columns=[
                            "session_id",
                            "camicu_time",
                            "event_timestamp",
                            "wave_type",
                        ],
                    ),
                ],
                axis=1,
            )
            wave_list_df.to_csv(
                osp.join(data_dir, f"wave_list_df.csv"),
                index=False,
            )

            return wave_list_df

    def read_wave_list_df(self, wave_list_df_path):
        wave_list_df = pd.read_csv(
            wave_list_df_path,
            dtype={"pat_id": object},
            parse_dates=["camicu_time", "event_timestamp"],
        )
        if "Unnamed: 0" in wave_list_df.columns:
            wave_list_df = wave_list_df.drop(["Unnamed: 0"], axis=1)
        self.wave_list_df = wave_list_df
        return wave_list_df

    def read_waves(self, row):
        extension = osp.basename(row["path"]).split(".")[-1]
        if extension == "csv":
            wave_data = pd.read_csv(row["path"], index_col=0)
        elif extension == "feather":
            wave_data = pd.read_feather(row["path"])

        wave_data = wave_data["signal"].values
        wave_data = wave_data.reshape(-1)
        return wave_data

    def get_last_8seconds(self, row):
        if not os.path.exists(row["path"]):
            return None

        # 한 파일 불러오기
        try:
            wave_data = self.read_waves(row)
        except:
            print("error in read_waves", row["path"])

        if len(wave_data) == 0:
            return None

        if len(wave_data) < 2048:
            return None
        last8s = wave_data[-2048:]

        return last8s

    def process_shape_feature_of_one_waveform(self, idx):
        wave = None
        data = None

        row = self.wave_list_df.iloc[idx]
        try:
            wave = self.get_last_8seconds(row)
        except Exception as e:
            print("error in ", idx, "get_last_8seconds")
            print(e)
        try:
            if wave is not None:
                data = extract_shape_feature(wave)
                return tuple([row["path"]]) + data
        except Exception as e:
            print("error in ", idx, "extract_shape_feature")
            print(e)
        return tuple([row["path"], data])

    def get_shape_feature_rslt(self, wave_list_df, tmp_dir, n_unit=10000, processes=40):
        """
        extract_shape_feature and write each n_unit(1000) files
        """
        os.makedirs(tmp_dir, exist_ok=True)
        nowDate = datetime.now().strftime("%Y%m%d")
        since = time.time()

        for i in range((len(wave_list_df) // n_unit) + 1):
            partial_range = range(
                i * n_unit, min(n_unit * i + n_unit, len(wave_list_df))
            )
            pool = multiprocessing.Pool(processes=processes)
            outputs = pool.map(
                self.process_shape_feature_of_one_waveform, partial_range
            )
            result_path = osp.join(
                tmp_dir,
                f"shape_feature_{nowDate}_{min(partial_range)}-{max(partial_range)}.csv",
            )
            pd.DataFrame(outputs).to_csv(result_path, index=False)
            print(
                min(partial_range),
                max(partial_range),
                "elapsed",
                (time.time() - since) / 60,
                "minutes",
            )

        shape_feature_names = [
            "hjorth_activity",
            "hjorth_morbidity",
            "hjorth_complexity",
            "kurtosis",
            "skewness",
        ]
        shapefiles = os.listdir(tmp_dir)
        df_list = []
        for f in shapefiles:
            if f == ".DS_Store":
                continue
            else:
                df = pd.read_csv(osp.join(tmp_dir, f))
                df_list.append(df)
        shape_results = pd.concat(df_list)  # .reset_index(drop=True)
        shape_results.columns = ["path"] + shape_feature_names
        shape_results = pd.concat(
            [
                shape_results.reset_index(drop=True),
                pd.DataFrame(
                    shape_results["path"]
                    .apply(lambda x: osp.basename(x))
                    .str.replace(".feather", "", regex=False)
                    .str.split("_")
                    .to_list(),
                    columns=[
                        "session_id",
                        "camicu_time",
                        "event_timestamp",
                        "wave_type",
                    ],
                ).reset_index(),
            ],
            axis=1,
        )
        # columns: ['path', 'filename', 'session_id', 'starttime', 'endtime', 'wave_type',
        # 'hjorth_activity', 'hjorth_morbidity', 'hjorth_complexity', 'kurtosis','skewness']

        shape_results["filename_instance"] = (
            shape_results["session_id"] + "_" + shape_results["camicu_time"]
        )

        wave_types = self.wv_config["abbr_wave_name_list"]  # ["ECGII", "pleth", "RESP"]
        df_list = []
        for wv in wave_types:
            df_list.append(
                shape_results[shape_results["wave_type"] == wv].rename(
                    columns=dict(
                        zip(
                            shape_feature_names,
                            [wv + "_" + i for i in shape_feature_names],
                        )
                    )
                )
            )
        shape_results = pd.concat(df_list)
        # 최대 24개 window(4시간/10분)의 shape feature들의 median
        shape_results = (
            shape_results.groupby("filename_instance")
            .median(numeric_only=True)
            .reset_index(drop=False)
        )
        shape_results = pd.concat(
            [
                pd.DataFrame(
                    shape_results["filename_instance"].str.split("_").to_list(),
                    columns=["session_id", "camicu_time"],
                ),
                shape_results,
            ],
            axis=1,
        )
        shape_results = shape_results.drop(columns=["filename_instance"])
        # columns: [session_id,starttime,event_timestamp,
        # ECGII_hjorth_activity,ECGII_hjorth_morbidity,ECGII_hjorth_complexity,ECGII_kurtosis,ECGII_skewness,
        # pleth_hjorth_activity,pleth_hjorth_morbidity,pleth_hjorth_complexity,pleth_kurtosis,pleth_skewness,
        # RESP_hjorth_activity,RESP_hjorth_morbidity,RESP_hjorth_complexity,RESP_kurtosis,RESP_skewness]
        return shape_results


if __name__ == "__main__":
    # if (sum(res) / len(res)) > 10000:
    data_array = [48642.0, 51714.0, 54530.0, 57090.0, 59650.0, 62466.0, 258.0, 4099.0]
    fixed = fix_decode(data_array)
    print(fixed)
