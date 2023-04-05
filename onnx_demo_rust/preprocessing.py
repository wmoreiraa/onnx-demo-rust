import config as c

 


import pandas as pd

from pytorch_forecasting import GroupNormalizer, TimeSeriesDataSet



class Preprocessor:
    def __init__(self, data: pd.DataFrame):
        self.data = data
    
    def run(self) -> dict:    
        training_cutoff = self.data["time_idx"].max() - 6
        max_encoder_length = c.max_encoder_length
        max_prediction_length = c.max_prediction_length

        training = TimeSeriesDataSet(
            self.data[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",
            target="volume",
            group_ids=["agency", "sku"],
            min_encoder_length=max_encoder_length // 2,  
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            static_categoricals=["agency", "sku"],
            static_reals=["avg_population_2017", "avg_yearly_household_income_2017"],
            time_varying_known_categoricals=["special_days", "month"],
            variable_groups={"special_days": c.special_days},  
            time_varying_known_reals=[
            "time_idx", "price_regular", "discount_in_percent"],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=[
                "volume",
                "log_volume",
                "industry_volume",
                "soda_volume",
                "avg_max_temp",
                "avg_volume_by_agency",
                "avg_volume_by_sku",
            ],
            target_normalizer=GroupNormalizer(
                groups=["agency", "sku"], transformation="softplus", center=False
            ),  # use softplus with beta=1.0 and normalize by group
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )


        validation = TimeSeriesDataSet.from_dataset(
            training, self.data, predict=True, stop_randomization=True)
        batch_size = 300
        train_dataloader = training.to_dataloader(
            train=True, batch_size=batch_size, num_workers=6)
        val_dataloader = validation.to_dataloader(
            train=False, batch_size=batch_size * 10, num_workers=6)
                # select last 24 months from data (max_encoder_length is 24)
        encoder_data = self.data[
            lambda x: x.time_idx > x.time_idx.max() - max_encoder_length]

        last_data = self.data[lambda x: x.time_idx == x.time_idx.max()]
        decoder_data = pd.concat(
            [last_data.assign(
            date=lambda x: x.date + pd.offsets.MonthBegin(i)
            ) for i in range(1, max_prediction_length + 1)],
            ignore_index=True,
        )

        # add time index consistent with "data"
        decoder_data["time_idx"] = (
            decoder_data["date"].dt.year * 12 + decoder_data["date"].dt.month
            )
        decoder_data["time_idx"] += (
            encoder_data["time_idx"].max() + 1 - decoder_data["time_idx"].min()
            )

        # adjust additional time feature(s)
        decoder_data["month"] = decoder_data.date.dt.month.astype(str).astype(
            "category"
            )

        # combine encoder and decoder data
        new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)
        return {
            "training": training,
            "validation": validation,
            "train_dataloader": train_dataloader,
            "val_dataloader": val_dataloader,
            "new_prediction_data": new_prediction_data,
        }
        





