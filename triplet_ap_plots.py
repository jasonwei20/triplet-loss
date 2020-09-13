from utils import common, configuration, triplet_methods, triplet_cl_methods

if __name__ == "__main__":

    # cfg_json_list_vanilla = [
    #     # "config/triplet_ap/eda_alpha_output/vanilla_huff_nc5_s0.json",
    #     # "config/triplet_ap/eda_alpha_output/vanilla_huff_nc5_s1.json",
    #     # "config/triplet_ap/eda_alpha_output/vanilla_huff_nc5_s2.json",
    #     # "config/triplet_ap/eda_alpha_output/vanilla_huff_nc5_s3.json",
    #     # "config/triplet_ap/eda_alpha_output/vanilla_huff_nc5_s4.json",
    #     # "config/triplet_ap/eda_alpha_output/vanilla_huff_nc5_s5.json",
    #     # "config/triplet_ap/eda_alpha_output/vanilla_huff_nc5_s6.json",
    #     # "config/triplet_ap/eda_alpha_output/vanilla_huff_nc5_s7.json",
    #     # "config/triplet_ap/eda_alpha_output/vanilla_huff_nc5_s8.json",
    #     # "config/triplet_ap/eda_alpha_output/vanilla_huff_nc5_s9.json",
    #     # "config/triplet_ap/eda_alpha_output/vanilla_huff_nc5_s10.json",
    #     # "config/triplet_ap/eda_alpha_output/vanilla_huff_nc5_s11.json",
    #     # "config/triplet_ap/eda_alpha_output/vanilla_huff_nc5_s12.json",
    #     # "config/triplet_ap/eda_alpha_output/eda_huff_nc5_a02_s0.json",
    #     # "config/triplet_ap/eda_alpha_output/eda_huff_nc5_a02_s1.json",
    #     # "config/triplet_ap/eda_alpha_output/eda_huff_nc5_a02_s2.json",
    #     # "config/triplet_ap/eda_alpha_output/eda_huff_nc5_a02_s3.json",
    #     # "config/triplet_ap/eda_alpha_output/eda_huff_nc5_a02_s4.json",
    #     # "config/triplet_ap/eda_alpha_output/eda_huff_nc5_a03_s0.json",
    #     # "config/triplet_ap/eda_alpha_output/eda_huff_nc5_a03_s1.json",
    #     # "config/triplet_ap/eda_alpha_output/eda_huff_nc5_a03_s2.json",
    #     # "config/triplet_ap/eda_alpha_output/eda_huff_nc5_a03_s3.json",
    #     # "config/triplet_ap/eda_alpha_output/eda_huff_nc5_a03_s4.json",
    #     # "config/triplet_ap/eda_alpha_output/eda_huff_nc5_a04_s0.json",
    #     # "config/triplet_ap/eda_alpha_output/eda_huff_nc5_a04_s1.json",
    #     # "config/triplet_ap/eda_alpha_output/eda_huff_nc5_a04_s2.json",
    #     # "config/triplet_ap/eda_alpha_output/eda_huff_nc5_a04_s3.json",
    #     # "config/triplet_ap/eda_alpha_output/eda_huff_nc5_a04_s4.json",
    # ]

    # for cfg_json in cfg_json_list_vanilla:

    #     cfg = configuration.triplet_ap_config.from_json(cfg_json); print(f"config from {cfg_json}")
    #     common.set_random_seed(cfg.seed_num)

    #     triplet_methods.train_eval_model(cfg)


    # cfg_json_list = [
    #     # "config/triplet_ap/eda_alpha_output/edacl_huff_nc5_a02_s0.json",
    #     # "config/triplet_ap/eda_alpha_output/edacl_huff_nc5_a02_s1.json",
    #     # "config/triplet_ap/eda_alpha_output/edacl_huff_nc5_a02_s2.json",
    #     # "config/triplet_ap/eda_alpha_output/edacl_huff_nc5_a02_s3.json",
    #     # "config/triplet_ap/eda_alpha_output/edacl_huff_nc5_a02_s4.json",
    #     # "config/triplet_ap/eda_alpha_output/edacl_huff_nc5_a03_s0.json",
    #     # "config/triplet_ap/eda_alpha_output/edacl_huff_nc5_a03_s1.json",
    #     # "config/triplet_ap/eda_alpha_output/edacl_huff_nc5_a03_s2.json",
    #     # "config/triplet_ap/eda_alpha_output/edacl_huff_nc5_a03_s3.json",
    #     # "config/triplet_ap/eda_alpha_output/edacl_huff_nc5_a03_s4.json",
    #     # "config/triplet_ap/eda_alpha_output/edacl_huff_nc5_a04_s0.json",
    #     # "config/triplet_ap/eda_alpha_output/edacl_huff_nc5_a04_s1.json",
    #     # "config/triplet_ap/eda_alpha_output/edacl_huff_nc5_a04_s2.json",
    #     # "config/triplet_ap/eda_alpha_output/edacl_huff_nc5_a04_s3.json",
    #     # "config/triplet_ap/eda_alpha_output/edacl_huff_nc5_a04_s4.json",
    # ]

    # for cfg_json in cfg_json_list:

    #     cfg = configuration.triplet_ap_cl_config.from_json(cfg_json); print(f"config from {cfg_json}")
    #     common.set_random_seed(cfg.seed_num)

    #     triplet_cl_methods.train_eval_cl_model(cfg)

    cfg_json_list = [
        # "config/triplet_ap/eda_alpha_output/edagrad_huff_nc5_s0.json",
        # "config/triplet_ap/eda_alpha_output/edagrad_huff_nc5_s1.json",
        # "config/triplet_ap/eda_alpha_output/edagrad_huff_nc5_s2.json",
        # "config/triplet_ap/eda_alpha_output/edagrad_huff_nc5_s3.json",
        # "config/triplet_ap/eda_alpha_output/edagrad_huff_nc5_s4.json",
        # "config/triplet_ap/eda_alpha_output/edagrad_huff_nc5_s5.json",
        # "config/triplet_ap/eda_alpha_output/edagrad_huff_nc5_s6.json",
        # "config/triplet_ap/eda_alpha_output/edagrad_huff_nc5_s7.json",
        # "config/triplet_ap/eda_alpha_output/edagrad_huff_nc5_s8.json",
        # "config/triplet_ap/eda_alpha_output/edagrad_huff_nc5_s9.json",
        # "config/triplet_ap/eda_alpha_output/edagrad_huff_nc5_s10.json",
        # "config/triplet_ap/eda_alpha_output/edagrad_huff_nc5_s11.json",
        # "config/triplet_ap/eda_alpha_output/edagrad_huff_nc5_s12.json",

        # "config/triplet_ap/eda_alpha_output/edagrad-b_huff_nc5_s0.json",
        # "config/triplet_ap/eda_alpha_output/edagrad-b_huff_nc5_s1.json",
        # "config/triplet_ap/eda_alpha_output/edagrad-b_huff_nc5_s2.json",
        # "config/triplet_ap/eda_alpha_output/edagrad-b_huff_nc5_s3.json",
        # "config/triplet_ap/eda_alpha_output/edagrad-b_huff_nc5_s4.json",
        # "config/triplet_ap/eda_alpha_output/edagrad-b_huff_nc5_s5.json",
        # "config/triplet_ap/eda_alpha_output/edagrad-b_huff_nc5_s6.json",
        # "config/triplet_ap/eda_alpha_output/edagrad-b_huff_nc5_s7.json",
        # "config/triplet_ap/eda_alpha_output/edagrad-b_huff_nc5_s8.json",
        # "config/triplet_ap/eda_alpha_output/edagrad-b_huff_nc5_s9.json",
        # "config/triplet_ap/eda_alpha_output/edagrad-b_huff_nc5_s10.json",
        # "config/triplet_ap/eda_alpha_output/edagrad-b_huff_nc5_s11.json",
        # "config/triplet_ap/eda_alpha_output/edagrad-b_huff_nc5_s12.json",
    ]

    for cfg_json in cfg_json_list:

        cfg = configuration.triplet_ap_cl_gradual_config.from_json(cfg_json); print(f"config from {cfg_json}")
        common.set_random_seed(cfg.seed_num)

        triplet_cl_methods.train_eval_cl_gradual_model(cfg)