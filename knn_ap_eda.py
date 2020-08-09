from utils import common, configuration, knn_methods

if __name__ == "__main__":

    cfg_json_list = [ #uses same configs as triplet_ap
        # "config/triplet_ap/eda_alpha/covidcat_nc10_a02.json",
        # "config/triplet_ap/eda_alpha/covidclu_nc3_a02.json",
        # "config/triplet_ap/eda_alpha/fewrel_nc5_a02.json",
        # "config/triplet_ap/eda_alpha/fewrel_nc10_a02.json",
        # "config/triplet_ap/eda_alpha/huff_nc5_a02.json",
        # "config/triplet_ap/eda_alpha/huff_nc10_a02.json",
        "config/triplet_ap/eda_alpha/amzn_nc3_a02.json",
    ]

    for cfg_json in cfg_json_list:

        cfg = configuration.triplet_ap_config.from_json(cfg_json); print(f"config from {cfg_json}")
        common.set_random_seed(cfg.seed_num)

        knn_methods.train_eval_model(cfg)