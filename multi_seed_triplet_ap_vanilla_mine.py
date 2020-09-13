from utils import common, configuration, triplet_methods

if __name__ == "__main__":

    cfg_json_list = [
        # "config/multi_seed_triplet_ap/vanilla_mine/amzn_nc3_s1.json",
        # "config/multi_seed_triplet_ap/vanilla_mine/amzn_nc3_s2.json",
        # "config/multi_seed_triplet_ap/vanilla_mine/amzn_nc3_s3.json",
        # "config/multi_seed_triplet_ap/vanilla_mine/amzn_nc3_s4.json",
        # "config/multi_seed_triplet_ap/vanilla_mine/covidclu_nc3_s1.json",
        # "config/multi_seed_triplet_ap/vanilla_mine/covidclu_nc3_s2.json",
        # "config/multi_seed_triplet_ap/vanilla_mine/covidclu_nc3_s3.json",
        # "config/multi_seed_triplet_ap/vanilla_mine/covidclu_nc3_s4.json",
        # "config/multi_seed_triplet_ap/vanilla_mine/fewrel_nc10_s1.json",
        # "config/multi_seed_triplet_ap/vanilla_mine/fewrel_nc10_s2.json",
        # "config/multi_seed_triplet_ap/vanilla_mine/fewrel_nc10_s3.json",
        # "config/multi_seed_triplet_ap/vanilla_mine/fewrel_nc10_s4.json",
        # "config/multi_seed_triplet_ap/vanilla_mine/huff_nc10_s1.json",
        # "config/multi_seed_triplet_ap/vanilla_mine/huff_nc10_s2.json",
        # "config/multi_seed_triplet_ap/vanilla_mine/huff_nc10_s3.json",
        # "config/multi_seed_triplet_ap/vanilla_mine/huff_nc10_s4.json",
        #
        # "config/multi_seed_triplet_ap/vanilla_mine/fewrel_nc3_s0.json",
        # "config/multi_seed_triplet_ap/vanilla_mine/fewrel_nc3_s1.json",
        # "config/multi_seed_triplet_ap/vanilla_mine/fewrel_nc3_s2.json",
        # "config/multi_seed_triplet_ap/vanilla_mine/fewrel_nc3_s3.json",
        # "config/multi_seed_triplet_ap/vanilla_mine/fewrel_nc3_s4.json",
        # "config/multi_seed_triplet_ap/vanilla_mine/fewrel_nc5_s0.json",
        # "config/multi_seed_triplet_ap/vanilla_mine/fewrel_nc5_s1.json",
        # "config/multi_seed_triplet_ap/vanilla_mine/fewrel_nc5_s2.json",
        # "config/multi_seed_triplet_ap/vanilla_mine/fewrel_nc5_s3.json",
        # "config/multi_seed_triplet_ap/vanilla_mine/fewrel_nc5_s4.json",
        # "config/multi_seed_triplet_ap/vanilla_mine/fewrel_nc20_s0.json",
        # "config/multi_seed_triplet_ap/vanilla_mine/fewrel_nc20_s1.json",
        # "config/multi_seed_triplet_ap/vanilla_mine/fewrel_nc20_s2.json",
        # "config/multi_seed_triplet_ap/vanilla_mine/fewrel_nc20_s3.json",
        # "config/multi_seed_triplet_ap/vanilla_mine/fewrel_nc20_s4.json",
        #
        # "config/multi_seed_triplet_ap/vanilla_mine/huff_nc3_s0.json",
        # "config/multi_seed_triplet_ap/vanilla_mine/huff_nc3_s1.json",
        # "config/multi_seed_triplet_ap/vanilla_mine/huff_nc3_s2.json",
        # "config/multi_seed_triplet_ap/vanilla_mine/huff_nc3_s3.json",
        # "config/multi_seed_triplet_ap/vanilla_mine/huff_nc3_s4.json",
        # "config/multi_seed_triplet_ap/vanilla_mine/huff_nc5_s0.json",
        # "config/multi_seed_triplet_ap/vanilla_mine/huff_nc5_s1.json",
        # "config/multi_seed_triplet_ap/vanilla_mine/huff_nc5_s2.json",
        # "config/multi_seed_triplet_ap/vanilla_mine/huff_nc5_s3.json",
        # "config/multi_seed_triplet_ap/vanilla_mine/huff_nc5_s4.json",
        # "config/multi_seed_triplet_ap/vanilla_mine/huff_nc20_s0.json",
        # "config/multi_seed_triplet_ap/vanilla_mine/huff_nc20_s1.json",
        # "config/multi_seed_triplet_ap/vanilla_mine/huff_nc20_s2.json",
        # "config/multi_seed_triplet_ap/vanilla_mine/huff_nc20_s3.json",
        # "config/multi_seed_triplet_ap/vanilla_mine/huff_nc20_s4.json",
    ]

    for cfg_json in cfg_json_list:

        cfg = configuration.triplet_ap_config.from_json(cfg_json); print(f"config from {cfg_json}")
        common.set_random_seed(cfg.seed_num)

        triplet_methods.train_eval_model(cfg)