from utils import common, configuration, triplet_methods

if __name__ == "__main__":

    cfg_json_list = [
        "config/triplet_ap/vanilla/sst2_nc10.json",
        "config/triplet_ap/vanilla/trec_nc10.json",
        "config/triplet_ap/vanilla/imdb_nc10.json",
    ]

    for cfg_json in cfg_json_list:

        cfg = configuration.triplet_ap_config.from_json(cfg_json); print(f"config from {cfg_json}")
        common.set_random_seed(cfg.seed_num)

        triplet_methods.train_eval_model(cfg)