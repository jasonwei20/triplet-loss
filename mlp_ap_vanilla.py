from utils import common, configuration, mlp_methods

if __name__ == "__main__":

    cfg_json_list = [
        # "config/lr/vanilla/fewrel_nc10.json",
        # "config/lr/vanilla/huff_nc10.json",
        # "config/lr/vanilla/covidclu_nc3.json",
        # "config/lr/vanilla/covidcat_nc10.json",
        # "config/lr/vanilla/trec_nc10.json",
        # "config/lr/vanilla/sst2_nc10.json",
        "config/mlp/vanilla/fewrel_nc10.json",
        "config/mlp/vanilla/huff_nc10.json",
        # "config/mlp/vanilla/covidclu_nc3.json",
        # "config/mlp/vanilla/covidcat_nc10.json",
        # "config/mlp/vanilla/trec_nc10.json",
        # "config/mlp/vanilla/sst2_nc10.json",
    ]

    for cfg_json in cfg_json_list:

        cfg = configuration.mlp_ap_config.from_json(cfg_json); print(f"config from {cfg_json}")
        common.set_random_seed(cfg.seed_num)

        mlp_methods.train_mlp(cfg)