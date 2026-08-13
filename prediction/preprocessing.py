import numpy as np
import pandas as pd


def preprocess_crop(data: dict) -> pd.DataFrame:
    frame = pd.DataFrame([data])
    frame["NPK_Sum"] = frame["N"] + frame["P"] + frame["K"]
    denominator = frame["NPK_Sum"] + 1e-6
    frame["NPK_Ratio_N"] = frame["N"] / denominator
    frame["NPK_Ratio_P"] = frame["P"] / denominator
    frame["NPK_Ratio_K"] = frame["K"] / denominator
    frame["Rainfall_Category"] = pd.cut(
        frame["rainfall"],
        bins=[0, 500, 1000, 1500, 2000, 2500, 3000, np.inf],
        labels=["Very Low", "Low", "Moderate", "High", "Very High", "Extreme", "Torrential"],
        right=False,
    )
    frame["Temperature_Category"] = pd.cut(
        frame["temperature"],
        bins=[0, 10, 20, 30, 40, 50, np.inf],
        labels=["Very Cold", "Cold", "Moderate", "Warm", "Hot", "Very Hot"],
        right=False,
    )
    return frame


def preprocess_fertilizer(data: dict) -> pd.DataFrame:
    frame = pd.DataFrame([data])
    frame["NPK_Sum"] = (
        frame["Nitrogen_Level"] + frame["Phosphorus_Level"] + frame["Potassium_Level"]
    )
    denominator = frame["NPK_Sum"] + 1e-6
    frame["NPK_Ratio_N"] = frame["Nitrogen_Level"] / denominator
    frame["NPK_Ratio_P"] = frame["Phosphorus_Level"] / denominator
    frame["NPK_Ratio_K"] = frame["Potassium_Level"] / denominator
    return frame
