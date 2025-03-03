from mmdet.registry import OPTIMIZERS
 
import torch
from galore_torch import GaLoreAdamW, GaLoreAdamW8bit, GaLoreAdafactor

@OPTIMIZERS.register_module()
class Galore(GaLoreAdamW):
    def dummy():
        return

@OPTIMIZERS.register_module()
class Galore8bit(GaLoreAdamW8bit):
    # def  __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False, optim_bits=32,args=None, min_8bit_size=4096, percentile_clipping=100, block_wise=True, is_paged=False):
    #     import ipdb
    #     ipdb.set_trace()
        
    #     galore_params = []
    #     target_modules_list = ["attn", "mlp"]
    #     for module_name, module in model.named_modules():
    #         if not isinstance(module, nn.Linear):
    #             continue

    #         if not any(target_key in module_name for target_key in target_modules_list):
    #             continue
            
    #         print('enable GaLore for weights in module: ', module_name)
    #         galore_params.append(module.weight)
    #     id_galore_params = [id(p) for p in galore_params]
    #     # make parameters without "rank" to another group
    #     regular_params = [p for p in model.parameters() if id(p) not in id_galore_params]
    #     # then call galore_adamw
    #     param_groups = [{'params': regular_params}, 
    #                     {'params': galore_params, 'rank': args.rank, 'update_proj_gap': args.update_proj_gap, 'scale': args.galore_scale, 'proj_type': args.proj_type}]
        
    #     super().__init__()
    def dummy():
        return
@OPTIMIZERS.register_module()
class GaloreAdafact(GaLoreAdafactor):
    def dummy():
        return