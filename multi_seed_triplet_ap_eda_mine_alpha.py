from utils import common, configuration, triplet_methods

if __name__ == "__main__":

    cfg_json_list = [
        # "config/multi_seed_triplet_ap/eda_mine_alpha/amzn_nc3_a02_s0.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/amzn_nc3_a02_s1.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/amzn_nc3_a02_s2.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/amzn_nc3_a02_s3.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/amzn_nc3_a02_s4.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/covidclu_nc3_a02_s0.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/covidclu_nc3_a02_s1.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/covidclu_nc3_a02_s2.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/covidclu_nc3_a02_s3.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/covidclu_nc3_a02_s4.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/fewrel_nc10_a02_s0.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/fewrel_nc10_a02_s1.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/fewrel_nc10_a02_s2.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/fewrel_nc10_a02_s3.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/fewrel_nc10_a02_s4.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/huff_nc10_a02_s0.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/huff_nc10_a02_s1.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/huff_nc10_a02_s2.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/huff_nc10_a02_s3.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/huff_nc10_a02_s4.json",
        #
        # "config/multi_seed_triplet_ap/eda_mine_alpha/huff_nc3_a02_s0.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/huff_nc3_a02_s1.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/huff_nc3_a02_s2.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/huff_nc3_a02_s3.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/huff_nc3_a02_s4.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/huff_nc5_a02_s0.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/huff_nc5_a02_s1.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/huff_nc5_a02_s2.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/huff_nc5_a02_s3.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/huff_nc5_a02_s4.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/huff_nc20_a02_s0.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/huff_nc20_a02_s1.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/huff_nc20_a02_s2.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/huff_nc20_a02_s3.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/huff_nc20_a02_s4.json",
        #
        # "config/multi_seed_triplet_ap/eda_mine_alpha/fewrel_nc3_a02_s0.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/fewrel_nc3_a02_s1.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/fewrel_nc3_a02_s2.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/fewrel_nc3_a02_s3.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/fewrel_nc3_a02_s4.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/fewrel_nc5_a02_s0.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/fewrel_nc5_a02_s1.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/fewrel_nc5_a02_s2.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/fewrel_nc5_a02_s3.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/fewrel_nc5_a02_s4.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/fewrel_nc20_a02_s0.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/fewrel_nc20_a02_s1.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/fewrel_nc20_a02_s2.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/fewrel_nc20_a02_s3.json",
        # "config/multi_seed_triplet_ap/eda_mine_alpha/fewrel_nc20_a02_s4.json",
    ]

    for cfg_json in cfg_json_list:

        cfg = configuration.triplet_ap_config.from_json(cfg_json); print(f"config from {cfg_json}")
        common.set_random_seed(cfg.seed_num)

        triplet_methods.train_eval_model(cfg)