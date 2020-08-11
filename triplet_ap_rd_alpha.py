from utils import common, configuration, triplet_methods

if __name__ == "__main__":

    cfg_json_list = [
        # "config/triplet_ap/rd_alpha/amzn_nc3_a01.json",
        # "config/triplet_ap/rd_alpha/amzn_nc3_a005.json",
        # "config/triplet_ap/rd_alpha/covidclu_nc3_a01.json",
        # "config/triplet_ap/rd_alpha/covidclu_nc3_a005.json",
        # "config/triplet_ap/rd_alpha/fewrel_nc5_a01.json",
        # "config/triplet_ap/rd_alpha/fewrel_nc5_a005.json",
        # "config/triplet_ap/rd_alpha/fewrel_nc10_a01.json",
        # "config/triplet_ap/rd_alpha/fewrel_nc10_a005.json",
        # "config/triplet_ap/rd_alpha/huff_nc5_a01.json",
        # "config/triplet_ap/rd_alpha/huff_nc5_a005.json",
        # "config/triplet_ap/rd_alpha/huff_nc10_a01.json",
        # "config/triplet_ap/rd_alpha/huff_nc10_a005.json",
    ]

    for cfg_json in cfg_json_list:

        cfg = configuration.triplet_ap_config.from_json(cfg_json); print(f"config from {cfg_json}")
        common.set_random_seed(cfg.seed_num)

        triplet_methods.train_eval_model(cfg)