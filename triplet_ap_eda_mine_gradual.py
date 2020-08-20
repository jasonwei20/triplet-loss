from utils import common, configuration, triplet_cl_methods

if __name__ == "__main__":

    cfg_json_list = [
        # "config/triplet_ap/eda_mine_gradual/amzn_nc3_c.json",
        # "config/triplet_ap/eda_mine_gradual/covidclu_nc3_b.json",
        # "config/triplet_ap/eda_mine_gradual/fewrel_nc5_b.json",
        # "config/triplet_ap/eda_mine_gradual/fewrel_nc10_c.json",
        # "config/triplet_ap/eda_mine_gradual/huff_nc5_b.json",
        # "config/triplet_ap/eda_mine_gradual/huff_nc10_b.json",
    ]

    for cfg_json in cfg_json_list:

        cfg = configuration.triplet_ap_cl_gradual_config.from_json(cfg_json); print(f"config from {cfg_json}")
        common.set_random_seed(cfg.seed_num)

        triplet_cl_methods.train_eval_cl_gradual_model(cfg)