
import numpy as np
import pandas as pd
from pytorch_forecasting.data.examples import get_stallion_data
import config as c

class DataLoader:
    def __init__(self):
        pass
    
    def run(self) -> pd.DataFrame:
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