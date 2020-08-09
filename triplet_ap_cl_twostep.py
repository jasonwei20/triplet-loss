from utils import common, configuration, triplet_cl_methods

if __name__ == "__main__":

    cfg_json_list = [
        # "config/triplet_ap/cl_twostep/covidcat_nc10_a005.json",
        # "config/triplet_ap/cl_twostep/covidcat_nc10_a01.json",
        # "config/triplet_ap/cl_twostep/covidcat_nc10_a02.json",
        # "config/triplet_ap/cl_twostep/covidcat_nc10_a03.json",
        # "config/triplet_ap/cl_twostep/covidcat_nc10_a04.json",
        # "config/triplet_ap/cl_twostep/covidcat_nc10_a05.json",
        # "config/triplet_ap/cl_twostep/covidclu_nc3_a005.json",
        # "config/triplet_ap/cl_twostep/covidclu_nc3_a01.json",
        # "config/triplet_ap/cl_twostep/covidclu_nc3_a02.json",
        # "config/triplet_ap/cl_twostep/covidclu_nc3_a03.json",
        # "config/triplet_ap/cl_twostep/covidclu_nc3_a04.json",
        # "config/triplet_ap/cl_twostep/covidclu_nc3_a05.json",
        # "config/triplet_ap/cl_twostep/huff_nc5_a005.json",
        # "config/triplet_ap/cl_twostep/huff_nc5_a01.json",
        # "config/triplet_ap/cl_twostep/huff_nc5_a02.json",
        # "config/triplet_ap/cl_twostep/huff_nc5_a03.json",
        # "config/triplet_ap/cl_twostep/huff_nc5_a04.json",
        # "config/triplet_ap/cl_twostep/huff_nc5_a05.json",
        # "config/triplet_ap/cl_twostep/huff_nc10_a005.json",
        # "config/triplet_ap/cl_twostep/huff_nc10_a01.json",
        # "config/triplet_ap/cl_twostep/huff_nc10_a02.json",
        # "config/triplet_ap/cl_twostep/huff_nc10_a03.json",
        # "config/triplet_ap/cl_twostep/huff_nc10_a04.json",
        # "config/triplet_ap/cl_twostep/huff_nc10_a05.json",
        # "config/triplet_ap/cl_twostep/fewrel_nc5_a005.json",
        # "config/triplet_ap/cl_twostep/fewrel_nc5_a01.json",
        # "config/triplet_ap/cl_twostep/fewrel_nc5_a02.json",
        # "config/triplet_ap/cl_twostep/fewrel_nc5_a03.json",
        # "config/triplet_ap/cl_twostep/fewrel_nc5_a04.json",
        # "config/triplet_ap/cl_twostep/fewrel_nc5_a05.json",
        # "config/triplet_ap/cl_twostep/fewrel_nc10_a005.json",
        # "config/triplet_ap/cl_twostep/fewrel_nc10_a01.json",
        # "config/triplet_ap/cl_twostep/fewrel_nc10_a02.json",
        # "config/triplet_ap/cl_twostep/fewrel_nc10_a03.json",
        # "config/triplet_ap/cl_twostep/fewrel_nc10_a04.json",
        # "config/triplet_ap/cl_twostep/fewrel_nc10_a05.json",
        # "config/triplet_ap/cl_twostep/amzn_nc3_a05.json",
        # "config/triplet_ap/cl_twostep/amzn_nc3_a005.json",
        # "config/triplet_ap/cl_twostep/amzn_nc3_a01.json",
        # "config/triplet_ap/cl_twostep/amzn_nc3_a04.json",
        # "config/triplet_ap/cl_twostep/amzn_nc3_a02.json",
        # "config/triplet_ap/cl_twostep/amzn_nc3_a03.json",
    ]

    for cfg_json in cfg_json_list:

        cfg = configuration.triplet_ap_cl_config.from_json(cfg_json); print(f"config from {cfg_json}")
        common.set_random_seed(cfg.seed_num)

        triplet_cl_methods.train_eval_cl_model(cfg)