from utils import common, configuration, knn_methods

if __name__ == "__main__":

    cfg_json_list = [ #uses same configs as triplet_ap
        # "config/triplet_ap/vanilla/covidcat_nc10.json",
        # "config/triplet_ap/vanilla/covidclu_nc3.json",
        # "config/triplet_ap/vanilla/fewrel_nc5.json",
        # "config/triplet_ap/vanilla/fewrel_nc10.json",
        # "config/triplet_ap/vanilla/huff_nc5.json",
        # "config/triplet_ap/vanilla/huff_nc10.json",
        # "config/triplet_ap/vanilla/amzn_nc3.json",
    ]

    for cfg_json in cfg_json_list:

        cfg = configuration.triplet_ap_config.from_json(cfg_json); print(f"config from {cfg_json}")
        common.set_random_seed(cfg.seed_num)

        knn_methods.train_eval_model(cfg)