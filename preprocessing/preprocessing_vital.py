import pandas as pd

import module.utils as cutils
from module.db_utils import create_df_from_ppg2


class VitalsPreprocessing:
    def __init__(self, vit_config={}, db_config_path="db.yaml"):
        """
        # 기준시점(camicu time)부터 4시간 이내 10분 간격으로 마지막 10초 추출
        vit_config : {"VITAL_DURATION_MINUTES": 10, "INPUT_INTERVAL_MINUTES":60, "INPUT_DURATION_LIMIT_HOURS":4,
        "abbr_vital_name_list": , "raw_vital_name_list": []}
        """
        self.vit_config = vit_config
        self.db_config = cutils.load_yaml(db_config_path)
        print("set vit_config", vit_config)

    def fetch_vital_from_db(self, sr: pd.Series):
        """read one VITAL_DURATION_MINUTES unit from DB

        Args:
            sr (pd.Series): one window time which have columns of session_id, event_timestamp

        Returns:
            pd.DataFrame: vital data (before preprocessing)
        """
        db_query = f"""select * from DB"""
        raw_vital = create_df_from_ppg2(db_query, self.db_config)
        raw_vital["param"] = raw_vital["param"].replace(
            dict(
                zip(
                    self.vit_config["raw_vital_name_list"],
                    self.vit_config["abbr_vital_name_list"],
                )
            )
        )

        print(len(raw_vital), "rows")
        return raw_vital

    def get_demo_raw_vital(self, sr):
        event_timestamp = sr["event_timestamp"]
        raw_vital = pd.read_csv(
            "./raw_vital",
            parse_dates=["mtime"],
        )
        raw_vital = raw_vital.query(f'session_id=="{sr["session_id"]}"')
        raw_vital = raw_vital[
            raw_vital["mtime"]
            >= (
                event_timestamp
                - pd.Timedelta(hours=self.vit_config["INPUT_DURATION_LIMIT_HOURS"])
            )
        ]
        raw_vital = raw_vital[raw_vital["mtime"] < event_timestamp]

        raw_vital["param"] = raw_vital["param"].replace(
            dict(
                zip(
                    self.vit_config["raw_vital_name_list"],
                    self.vit_config["abbr_vital_name_list"],
                )
            )
        )
        raw_vital = raw_vital.drop(columns=["session_id"])

        if len(raw_vital) > 0:
            print(sr)
            print(len(raw_vital), "rows")
        return raw_vital

    def transform_vital(self, raw_vital, event_timestamp=None):
        """
        1. merge multiple rows to a list of vital
        """

        transformed_dict = {}
        for vt in self.vit_config["abbr_vital_name_list"]:
            one_param_df = raw_vital.query(f"param=='{vt}'")

            if len(one_param_df) == 0:
                transformed_dict[vt] = None
                print("no vital data")
            else:
                ### resampling
                one_param_sr = (
                    one_param_df.set_index("mtime")["data"].resample("1T").median()
                )
                one_param_sr = one_param_sr.interpolate(method="time").rename(vt)
                transformed_dict[vt] = one_param_sr
        return transformed_dict

    def extract_medstd(self, transformed_dict):
        tmp = {}
        for vt in transformed_dict:
            tmp[vt + "_median"] = transformed_dict[vt].median()
            tmp[vt + "_std"] = transformed_dict[vt].std()
        return pd.DataFrame(tmp, index=[0])
