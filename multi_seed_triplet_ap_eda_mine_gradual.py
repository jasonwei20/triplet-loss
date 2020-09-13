from utils import common, configuration, triplet_cl_methods

if __name__ == "__main__":

    cfg_json_list = [
        # "config/multi_seed_triplet_ap/eda_mine_gradual/amzn_nc3_c_s0.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/amzn_nc3_c_s1.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/amzn_nc3_c_s2.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/amzn_nc3_c_s3.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/amzn_nc3_c_s4.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/covidclu_nc3_b_s0.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/covidclu_nc3_b_s1.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/covidclu_nc3_b_s2.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/covidclu_nc3_b_s3.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/covidclu_nc3_b_s4.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/fewrel_nc10_c_s0.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/fewrel_nc10_c_s1.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/fewrel_nc10_c_s2.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/fewrel_nc10_c_s3.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/fewrel_nc10_c_s4.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/huff_nc10_b_s0.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/huff_nc10_b_s1.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/huff_nc10_b_s2.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/huff_nc10_b_s3.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/huff_nc10_b_s4.json",
        #
        # "config/multi_seed_triplet_ap/eda_mine_gradual/huff_nc3_b_s0.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/huff_nc3_b_s1.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/huff_nc3_b_s2.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/huff_nc3_b_s3.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/huff_nc3_b_s4.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/huff_nc5_b_s0.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/huff_nc5_b_s1.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/huff_nc5_b_s2.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/huff_nc5_b_s3.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/huff_nc5_b_s4.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/huff_nc20_b_s0.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/huff_nc20_b_s1.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/huff_nc20_b_s2.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/huff_nc20_b_s3.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/huff_nc20_b_s4.json",
        #
        # "config/multi_seed_triplet_ap/eda_mine_gradual/fewrel_nc3_c_s0.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/fewrel_nc3_c_s1.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/fewrel_nc3_c_s2.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/fewrel_nc3_c_s3.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/fewrel_nc3_c_s4.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/fewrel_nc5_c_s0.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/fewrel_nc5_c_s1.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/fewrel_nc5_c_s2.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/fewrel_nc5_c_s3.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/fewrel_nc5_c_s4.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/fewrel_nc20_c_s0.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/fewrel_nc20_c_s1.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/fewrel_nc20_c_s2.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/fewrel_nc20_c_s3.json",
        # "config/multi_seed_triplet_ap/eda_mine_gradual/fewrel_nc20_c_s4.json",
    ]

    for cfg_json in cfg_json_list:

        cfg = configuration.triplet_ap_cl_gradual_config.from_json(cfg_json); print(f"config from {cfg_json}")
        common.set_random_seed(cfg.seed_num)

        triplet_cl_methods.train_eval_cl_gradual_model(cfg)