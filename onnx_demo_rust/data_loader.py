
import numpy as np
import pandas as pd
from pytorch_forecasting.data.examples import get_stallion_data
import config as c

class DataLoader:
    def __init__(self):
        pass
    
    def run_fit(self) -> pd.DataFrame:
        data = get_stallion_data()

        data["month"] = data.date.dt.month.astype("str").astype("category")
        data["log_volume"] = np.log(data.volume + 1e-8)

        data["time_idx"] = data["date"].dt.year * 12 + data["date"].dt.month
        data["time_idx"] -= data["time_idx"].min()
        data["avg_volume_by_sku"] = (data
                                     .groupby(["time_idx", "sku"], observed=True)
                                     .volume.transform("mean"))
        data["avg_volume_by_agency"] = (data
                                        .groupby(["time_idx", "agency"], observed=True)
                                        .volume.transform("mean"))
        
        data[c.special_days] = (data[c.special_days]
                              .apply(lambda x: x.map({0: "", 1: x.name}))
                              .astype("category"))
        
        return data
    def run_inference(self):
        data = self.run_fit()
        # select last 24 months from data (max_encoder_length is 24)
        encoder_data = data[lambda x: x.time_idx > x.time_idx.max() - c.max_encoder_length]

        last_data = data[lambda x: x.time_idx == x.time_idx.max()]
        decoder_data = pd.concat(
            [last_data.assign(
            date=lambda x: x.date + pd.offsets.MonthBegin(i)
            ) for i in range(1, c.max_prediction_length + 1)],
            ignore_index=True,
        )

        # add time index consistent with "data"
        decoder_data["time_idx"] = (
            decoder_data["date"].dt.year * 12 + decoder_data["date"].dt.month)
        decoder_data["time_idx"] += (
            encoder_data["time_idx"].max() + 1 - decoder_data["time_idx"].min())

        # adjust additional time feature(s)
        decoder_data["month"] = decoder_data.date.dt.month.astype(str).astype("category"
                                                                              )
        # combine encoder and decoder data
        new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)
        return new_prediction_data

