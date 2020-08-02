from utils import common, configuration, mlp_methods

if __name__ == "__main__":

    cfg_json_list = [
        # "config/lr/sr/fewrel_nc10.json",
        # "config/lr/sr/huff_nc10.json",
        # "config/lr/sr/covidclu_nc3.json",
        # "config/lr/sr/covidcat_nc10.json",
        # "config/lr/sr/trec_nc10.json",
        # "config/lr/sr/sst2_nc10.json",
        "config/mlp/sr/fewrel_nc10.json",
        "config/mlp/sr/huff_nc10.json",
        # "config/mlp/sr/covidclu_nc3.json",
        # "config/mlp/sr/covidcat_nc10.json",
        # "config/mlp/sr/trec_nc10.json",
        # "config/mlp/sr/sst2_nc10.json",
    ]

    for cfg_json in cfg_json_list:

        cfg = configuration.mlp_ap_config.from_json(cfg_json); print(f"config from {cfg_json}")
        common.set_random_seed(cfg.seed_num)

        mlp_methods.train_mlp(cfg)