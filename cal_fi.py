import os
import numpy as np
import pandas as pd
from pathlib import Path
from autogluon.tabular import TabularDataset, TabularPredictor
from tqdm import tqdm

p_model_meta = Path("./deploy/model100_metainfo.csv")
p_model = Path("./data/middle/model/nephro_toxi_model")
p_main = Path("./deploy/")
p_fi = p_main / "feat_importence.csv"


model_meta = pd.read_csv(p_model_meta)

fi_list = []

for i in tqdm(range(model_meta.shape[0])):
    # i = 0
    set_id = model_meta["set"][i]
    model_type = model_meta["model"][i]
    feat_type = model_meta["feat_type"][i]
    fold = model_meta["fold"][i]
    
    p_model_choose = p_model / set_id / f"{feat_type}_fold{str(fold)}"
    print(f"===> Start to calculate fi for {set_id} model ({p_model_choose.name})")

    model = TabularPredictor.load(p_model_choose)

    fi = model.feature_importance(feature_stage="transformed", model = model_type)
    fi = fi.reset_index()
    fi["set"] = set_id
    fi["feat_type"] = feat_type
    fi["model"] = model_type
    fi["fold"] = fold
    print(fi.head())
    print(fi.shape)
    fi_list.append(fi)

    # feature importance

fi_df = pd.concat(fi_list)
fi_df.to_csv(p_fi)

