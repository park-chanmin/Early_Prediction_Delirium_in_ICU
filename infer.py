import os
import os.path as osp
import argparse
from datetime import datetime
import pandas as pd
import pycaret
import pycaret.classification as pycl

import module.utils as cutils
from module.metrics import eval_metrics
from module.metrics import calculate_any_metrics


def main(feature_path, model_file, result_path, **kwarg):
    data = pd.read_csv(feature_path, low_memory=False)

    if "aid_mapping_path" in kwarg:
        map_df = pd.read_csv(kwarg["aid_mapping_path"], low_memory=False)
        data = pd.merge(map_df, data, on="index_number")  # mapping allocation_number

    if "allocation_number" not in data.columns:
        # for demo
        data["allocation_number"] = "A1-" + pd.Series(range(1, len(data) + 1)).apply(
            lambda x: f"{x:03d}"
        )

    if "enrolled_patients_path" in kwarg:
        enrolled_patients = pd.read_csv(
            kwarg["enrolled_patients_path"], low_memory=False, header=None
        )
        enrolled = data["allocation_number"].isin(enrolled_patients[0])
        print("등록번호 (allocation_number) 존재 행", enrolled.value_counts())
        data = data[enrolled]

    features = data[
        [
            "allocation_number",
            "camicu_time",
            # "cam",
            "sex",
            "age",
            "ECGII_hjorth_activity",
            "pleth_hjorth_activity",
            "RESP_hjorth_activity",
            "ECGII_hjorth_complexity",
            "ECGII_hjorth_morbidity",
            "ECGII_kurtosis",
            "pleth_kurtosis",
            "RESP_kurtosis",
            "ECGII_skewness",
            "pleth_skewness",
            "RESP_skewness",
            "HR_median",
            "RR_median",
            "SpO2_median",
            "HR_std",
            "RR_std",
            "SpO2_std",
        ]
    ]
    features.columns = [
        "allocation_number",
        "camicu_time",
        # "cam",
        "sex",
        "age",
        "hjorth_activity_II",
        "hjorth_activity_Pleth",
        "hjorth_activity_Resp",
        "hjorth_complexity_II",
        "hjorth_morbidity_II",
        "kurtosis_II",
        "kurtosis_Pleth",
        "kurtosis_Resp",
        "skewness_II",
        "skewness_Pleth",
        "skewness_Resp",
        "HRs_median",
        "RRs_median",
        "SpO2s_median",
        "HRs_std",
        "RRs_std",
        "SpO2s_std",
    ]
    if "cam_icu" in data.columns:
        features["cam_icu"] = data["cam_icu"]

    print("len(features) :", len(features))
    # missing이 있는 경우 제거
    features = features.dropna(axis=0, how="any")
    print("after dropna, len(features) :", len(features))

    # 문자형 데이터를 숫자형으로 변환
    features = features.replace({"sex": "M"}, {"sex": 0})
    features = features.replace({"sex": "F"}, {"sex": 1})

    final_model = pycl.load_model(model_file.replace(".pkl", ""))

    prediction = pycl.predict_model(final_model, data=features, raw_score=True)

    os.makedirs(osp.dirname(result_path), exist_ok=True)
    prediction = prediction[["allocation_number", "camicu_time", "Score_1"]].rename(
        columns={"Score_1": "d_risk"}
    )
    # score: min 0, max 0.9147110332749562
    prediction["d_risk"] = (52.23 / 57.1) * prediction["d_risk"]

    if "cam_icu" in features.columns:
        prediction["cam_icu"] = features["cam_icu"]

        result_metrics = calculate_any_metrics(
            target=prediction["cam_icu"].to_numpy(),
            probs=prediction["d_risk"].to_numpy(),
            metrics=eval_metrics,
        )
        print("results :", result_metrics)

    prediction.to_csv(result_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Inference")
    parser.add_argument(
        "--config",
        default="/VOLUME/project/saveu/save-u-d_deployment/clinical_trial/config/infer_config.yml",
        help="config file",
    )

    args = parser.parse_args()
    infer_config = cutils.load_yaml(args.config)
    print(infer_config)
    main(**infer_config)
