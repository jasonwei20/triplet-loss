from utils import common, configuration, mlp_methods

if __name__ == "__main__":

    cfg_json_list = [
        # "config/mlp/multi_seed/vanilla/amzn_nc3_s0.json",
        # "config/mlp/multi_seed/vanilla/amzn_nc3_s1.json",
        # "config/mlp/multi_seed/vanilla/amzn_nc3_s2.json",
        # "config/mlp/multi_seed/vanilla/amzn_nc3_s3.json",
        # "config/mlp/multi_seed/vanilla/amzn_nc3_s4.json",
        # "config/mlp/multi_seed/vanilla/covidclu_nc3_s0.json",
        # "config/mlp/multi_seed/vanilla/covidclu_nc3_s1.json",
        # "config/mlp/multi_seed/vanilla/covidclu_nc3_s2.json",
        # "config/mlp/multi_seed/vanilla/covidclu_nc3_s3.json",
        # "config/mlp/multi_seed/vanilla/covidclu_nc3_s4.json",
        # "config/mlp/multi_seed/vanilla/huff_nc10_s0.json",
        # "config/mlp/multi_seed/vanilla/huff_nc10_s1.json",
        # "config/mlp/multi_seed/vanilla/huff_nc10_s2.json",
        # "config/mlp/multi_seed/vanilla/huff_nc10_s3.json",
        # "config/mlp/multi_seed/vanilla/huff_nc10_s4.json",
        # "config/mlp/multi_seed/vanilla/fewrel_nc10_s0.json",
        # "config/mlp/multi_seed/vanilla/fewrel_nc10_s1.json",
        # "config/mlp/multi_seed/vanilla/fewrel_nc10_s2.json",
        # "config/mlp/multi_seed/vanilla/fewrel_nc10_s3.json",
        # "config/mlp/multi_seed/vanilla/fewrel_nc10_s4.json",

        "config/mlp/multi_seed/eda/amzn_nc3_s0.json",
        "config/mlp/multi_seed/eda/amzn_nc3_s1.json",
        "config/mlp/multi_seed/eda/amzn_nc3_s2.json",
        "config/mlp/multi_seed/eda/amzn_nc3_s3.json",
        "config/mlp/multi_seed/eda/amzn_nc3_s4.json",
        "config/mlp/multi_seed/eda/covidclu_nc3_s0.json",
        "config/mlp/multi_seed/eda/covidclu_nc3_s1.json",
        "config/mlp/multi_seed/eda/covidclu_nc3_s2.json",
        "config/mlp/multi_seed/eda/covidclu_nc3_s3.json",
        "config/mlp/multi_seed/eda/covidclu_nc3_s4.json",
        "config/mlp/multi_seed/eda/huff_nc10_s0.json",
        "config/mlp/multi_seed/eda/huff_nc10_s1.json",
        "config/mlp/multi_seed/eda/huff_nc10_s2.json",
        "config/mlp/multi_seed/eda/huff_nc10_s3.json",
        "config/mlp/multi_seed/eda/huff_nc10_s4.json",
        "config/mlp/multi_seed/eda/fewrel_nc10_s0.json",
        "config/mlp/multi_seed/eda/fewrel_nc10_s1.json",
        "config/mlp/multi_seed/eda/fewrel_nc10_s2.json",
        "config/mlp/multi_seed/eda/fewrel_nc10_s3.json",
        "config/mlp/multi_seed/eda/fewrel_nc10_s4.json",
    ]

    for cfg_json in cfg_json_list:

        cfg = configuration.mlp_ap_config.from_json(cfg_json); print(f"config from {cfg_json}")
        common.set_random_seed(cfg.seed_num)

        mlp_methods.train_mlp(cfg)