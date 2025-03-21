import sys
import torch
from torch import nn
from typing import List


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_feature_extractor(**kwargs):
    """ Create the feature extractor . """
    print("Creating DDPM Feature Extractor...")
    feature_extractor = FeatureExtractorDDPM(**kwargs)
    return feature_extractor


def save_tensors(module: nn.Module, features, name: str):
    """ Process and save activations in the module. """
    if type(features) in [list, tuple]:
        features = [f.detach().float() if f is not None else None 
                    for f in features]
        setattr(module, name, features)
    elif isinstance(features, dict):
        features = {k: f.detach().float() for k, f in features.items()}
        setattr(module, name, features)
    else:
        setattr(module, name, features.detach().float())


def save_out_hook(self, inp, out):
    save_tensors(self, out, 'activations')
    return out


def save_input_hook(self, inp, out):
    save_tensors(self, inp[0], 'activations')
    return out


class FeatureExtractor(nn.Module):
    def __init__(self, model_path: str, input_activations: bool, **kwargs):
        ''' 
        Parent feature extractor class.
        
        param: model_path: path to the pretrained model
        param: input_activations: 
            If True, features are input activations of the corresponding blocks
            If False, features are output activations of the corresponding blocks
        '''
        super().__init__()
        self._load_pretrained_model(model_path, **kwargs)
        print(f"Pretrained model is successfully loaded from {model_path}")
        self.save_hook = save_input_hook if input_activations else save_out_hook
        self.feature_blocks = []

    def _load_pretrained_model(self, model_path: str, **kwargs):
        pass


class FeatureExtractorDDPM(FeatureExtractor):
    ''' 
    Wrapper to extract features from pretrained DDPMs.
            
    :param steps: list of diffusion steps t.
    :param blocks: list of the UNet decoder blocks.
    '''
    
    def __init__(self, steps: List[int], blocks: List[int], **kwargs):
        super().__init__(**kwargs)
        self.steps = steps
        
        # Save decoder activations
        for idx, block in enumerate(self.model.output_blocks):
            if idx in blocks:
                block.register_forward_hook(self.save_hook)
                self.feature_blocks.append(block)
        # 
        # print(self.model.input_blocks)

    def _load_pretrained_model(self, model_path, **kwargs):
        import inspect
        import improved_diffusion.dist_util as dist_util
        from improved_diffusion.script_util import create_model_and_diffusion

        # Needed to pass only expected args to the function
        argnames = inspect.getfullargspec(create_model_and_diffusion)[0]
        expected_args = {name: kwargs[name] for name in argnames}
        # print(expected_args)
        self.model, self.diffusion = create_model_and_diffusion(**expected_args)
        
        self.model.load_state_dict(
            dist_util.load_state_dict(model_path, map_location="cpu")
        )
        self.model.to(dist_util.dev())
        #if kwargs['use_fp16']:
        #    self.model.convert_to_fp16()
        self.model.eval()

    @torch.no_grad()
    def forward(self, x, noise=None):
        activations = []
        res_x = None
        for t in self.steps:
            # Compute x_t and run DDPM
            t = torch.tensor([t]).to(x.device)
            if noise != None:
              noisy_x = self.diffusion.q_sample(x, t, noise=noise)
              ## 噪声特征的第一个表示方法
              ##res_x = x - noisy_x
              ## 噪声特征的第二个表示方法，从0张量图像中采样获取。
              zero_img = torch.zeros_like(x).to(x.device)
              res_x = self.diffusion.q_sample(zero_img, t, noise=noise)
            else:
              noisy_x = x
            self.model(noisy_x, self.diffusion._scale_timesteps(t))

            # Extract activations
            for block in self.feature_blocks:
                activations.append(block.activations)
                block.activations = None

        # Per-layer list of activations [N, C, H, W]
        return activations, res_x

def res_feature(feature: List[torch.Tensor], nosie_feature: List[torch.Tensor]):
    assert len(feature) == len(nosie_feature)
    for i in range(len(feature)):
        feature[i] = feature[i] - nosie_feature[i]
    return feature
    

def collect_features(args, activations: List[torch.Tensor], sample_idx=0):
    """ Upsample activations and concatenate them to form a feature tensor """
    assert all([isinstance(acts, torch.Tensor) for acts in activations])
    size = tuple(args['dim'][:-1])
    resized_activations = []
    for feats in activations:
        feats = feats[sample_idx][None]
        feats = nn.functional.interpolate(
            feats, size=size, mode=args["upsample_mode"]
        )
        resized_activations.append(feats[0])
    
    return torch.cat(resized_activations, dim=0)
