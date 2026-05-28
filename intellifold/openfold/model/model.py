# Copyright 2024 IntelliGen-AI and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import einops
from intellifold.openfold.model.backbone import BackboneTrunk
from intellifold.openfold.model.diffusion import DiffusionModule
from intellifold.openfold.model.heads import ConfidenceHead
from intellifold.openfold.utils.atom_token_conversion import aggregate_fn, aggregate_fn_advanced

from torch.amp import autocast

def exists(v):
    return v is not None

class IntelliFold(nn.Module):
    """
    Implements Algorithm 1
    """

    def __init__(self, config, generator = None):   
        """
        Args:
            config:
                A dict-like config object (like the one in config.py)
        """
        super(IntelliFold, self).__init__()

        self.globals = config.globals
        self.sample_config = config.sample
        self.config = config
        
        self.backbone_trunk = BackboneTrunk(self.config)
        self.diffusion_module = DiffusionModule(self.config)
        self.confidence_head = ConfidenceHead(self.config)
        self.centre_random_augmentation = CentreRandomAugmentation()
        self.generator = generator
        self.advanced_conversion = self.globals.advanced_conversion

    @autocast("cuda",enabled=True, dtype=torch.float32)
    def diffusion_edm_forward(self,x_noisy,t,input_features,s_inputs,s_trunk,z_trunk):
        
        scale_skip = self.diffusion_module.sigma_data ** 2 / (t ** 2 + self.diffusion_module.sigma_data ** 2)
        scale_out = t * self.diffusion_module.sigma_data / ((t ** 2 + self.diffusion_module.sigma_data ** 2).sqrt())
        scale_in = 1 / ((self.diffusion_module.sigma_data ** 2 + t ** 2).sqrt())
        r_noisy = x_noisy * scale_in
        r_update = self.diffusion_module(r_noisy,t,input_features,s_inputs,s_trunk,z_trunk)
        x_out = scale_skip * x_noisy + scale_out * r_update
        return x_out
    
    def noise_schedule(self,device):

        T = self.sample_config.no_sample_steps_T
        rho = self.sample_config.rho
        sigma_data = self.sample_config.sigma_data
        s_max = self.sample_config.sigma_max
        s_min = self.sample_config.sigma_min
        inv_rho = 1 / rho

        t = torch.linspace(0, 1, T + 1, device = device)

        t = sigma_data * (s_max ** inv_rho + t * (s_min ** inv_rho - s_max ** inv_rho)) ** rho
        #         t = torch.cat([t, torch.zeros_like(t[:1])]) 
        return  t

    @torch.no_grad()
    @autocast("cuda",enabled=True, dtype=torch.float32)
    def sample_diffusion(self,input_features, s_inputs,s_trunk,z_trunk,diffusion_batch_size):
        """
        Args:
            input_features (dict): Dictionary of features, as outlined in Algorithm 5.
            s_inputs: 
                [*, N_token, C_s_inputs]: Initial input featues
            s_trunk: 
                [*, N_token, C_s]: Output of the backbone trunk.
            z_trunk: 
                [*, N_token, N_token, C_z]: Output of the backbone trunk.

            diffusion_batch_size: The augmentation batch size for the diffusion module.
            
        Returns:
            Sampled output of the diffusion module.
        """
        noise_schedule_c = self.noise_schedule(device = input_features['ref_pos'].device)
        
        gamma_0 =  self.sample_config.gamma_0
        gamma_min = self.sample_config.gamma_min
        noise_scale_lambda = self.sample_config.noise_scale_lambda
        step_scale_eta = self.sample_config.step_scale_eta
        
        T = len(noise_schedule_c) - 1
        x = noise_schedule_c[0] * torch.empty_like(einops.repeat(input_features["ref_pos"], "b ... -> (b n) ...", n = diffusion_batch_size),device = input_features["ref_pos"].device).normal_(generator = self.generator)
        pred_dense_atom_mask = input_features['pred_dense_atom_mask']
        if self.advanced_conversion:
            [aggregated_pred_dense_atom_mask], _ = aggregate_fn_advanced([pred_dense_atom_mask], pred_dense_atom_mask)
        else:
            [aggregated_pred_dense_atom_mask], _ = aggregate_fn([pred_dense_atom_mask], pred_dense_atom_mask)
        aggregated_pred_dense_atom_mask = einops.repeat(aggregated_pred_dense_atom_mask, 'b ... -> (b n) ...', n = diffusion_batch_size)

        for tau in range(1, T + 1):
            
            x = self.centre_random_augmentation(x, aggregated_pred_dense_atom_mask, generator = self.generator)
            
            c_tau = noise_schedule_c[tau]
            c_tau_minus_1 = noise_schedule_c[tau - 1]
            
            gamma = gamma_0 if c_tau > gamma_min else 0.0
            
            c_tau_minus_1 = einops.repeat(c_tau_minus_1, '-> b 1 1', b = x.shape[0])
            c_tau = einops.repeat(c_tau, '-> b 1 1', b = x.shape[0])
            
            t = c_tau_minus_1 * (gamma + 1)
            
            xi = noise_scale_lambda * ((t ** 2 - c_tau_minus_1 ** 2).sqrt()) * torch.empty_like(x,device = x.device).normal_(generator = self.generator)
            
            x_noisy = x + xi
            
            x_denoised = self.diffusion_edm_forward(
                x_noisy = x_noisy,
                t = t,
                input_features = input_features,
                s_inputs = s_inputs,
                s_trunk = s_trunk,
                z_trunk = z_trunk
            )
            
            delta = (x_noisy - x_denoised) / t
            
            dt = c_tau - t
            
            x = x_noisy + step_scale_eta * dt * delta

        
        return x
        
    def forward(self, input_features, diffusion_batch_size=1):
        """
        Args:
            input_features (dict): Dictionary of features, as outlined in Algorithm 5.
                        
            diffusion_batch_size: The augmentation batch size for the diffusion module.
            
        Returns:
            Output of the forward pass.
        """
        outputs = dict()
        
        # aggregate the ref_features
        aggregated_ref_keys = [key for key in input_features.keys() if 'ref_' in key]
        if self.advanced_conversion:
            aggregated_ref_features, reverse_fn = aggregate_fn_advanced([input_features[key] for key in aggregated_ref_keys], input_features['pred_dense_atom_mask'])
        else:
            aggregated_ref_features, reverse_fn = aggregate_fn([input_features[key] for key in aggregated_ref_keys], input_features['pred_dense_atom_mask'])

        # update the input_features with the aggregated ref_features
        input_features.update(dict(zip(aggregated_ref_keys, aggregated_ref_features)))
        
        input_features['molecule_atom_lens'] = input_features['pred_dense_atom_mask'].sum(dim = -1)
        
        if self.advanced_conversion:
            aggregated_output, _ = aggregate_fn_advanced([input_features['pred_dense_atom_mask']], input_features['pred_dense_atom_mask'])
        else:
            aggregated_output, _ = aggregate_fn([input_features['pred_dense_atom_mask']], input_features['pred_dense_atom_mask'])
        input_features['aggregated_pred_dense_atom_mask'] = aggregated_output[0].float() # use in the model
        
        # forward the backbone trunk
        backbone_outputs = self.backbone_trunk(input_features)
        outputs['distogram_logits'] = backbone_outputs['distogram_logits']        
        
        x_predicted = self.sample_diffusion(input_features,backbone_outputs['s_inputs'],backbone_outputs['s'],backbone_outputs['z'],diffusion_batch_size)
        x_predicted = reverse_fn([x_predicted])[0]
        outputs['x_predicted'] = x_predicted
                        
        confidence_outputs = self.confidence_head(backbone_outputs['s_inputs'],backbone_outputs['s'],backbone_outputs['z'],x_predicted,input_features)
        outputs.update(confidence_outputs)
    
        return outputs


class CentreRandomAugmentation(nn.Module):
    """
    The CentreRandomAugmentation module applies random rotation and translation to the input coordinates.
    
    Implements Algorithm 19
    """

    def __init__(self, s_trans = 1.0):
        super().__init__()
        self.s_trans = s_trans
        self.register_buffer('dummy', torch.tensor(0), persistent = False)

    def device(self):
        return self.dummy.device

    @torch.no_grad()
    @autocast("cuda",enabled=True, dtype=torch.float32)
    def forward(
        self,
        x,
        mask,
        generator = None
    ):
        """
        x: coordinates to be augmented
        """
        batch_size = x.shape[0]

        if mask is not None:
            x_mean = (torch.sum(x * mask.unsqueeze(-1), dim = -2, keepdim = True) /
                              (torch.sum(mask.float(), dim = -1, keepdim = True).unsqueeze(-1).clamp(min = 1.)))
            x = (x - x_mean) * mask.unsqueeze(-1)

        else:
            raise ValueError("mask is None")
            

        # Generate random rotation matrix
        R = self._random_rotation_matrix(batch_size, generator = generator)

        # Generate random translation vector
        t = self._random_translation_vector(batch_size, generator = generator)
        t = t.unsqueeze(-2)

        # Apply rotation and translation
        x = torch.einsum('b n i, b j i -> b n j', x, R) + t

        return x * mask.unsqueeze(-1)

    def _random_rotation_matrix(self, batch_size, generator = None):
        # Generate random rotation angles
        angles = torch.rand((batch_size, 3), device = self.device(), generator = generator) * 2 * torch.pi

        # Compute sine and cosine of angles
        sin_angles = torch.sin(angles)
        cos_angles = torch.cos(angles)

        # Construct rotation matrix
        eye = torch.eye(3, device = self.device())
        rotation_matrix = einops.repeat(eye, 'i j -> b i j', b = batch_size).clone()

        rotation_matrix[:, 0, 0] = cos_angles[:, 0] * cos_angles[:, 1]
        rotation_matrix[:, 0, 1] = cos_angles[:, 0] * sin_angles[:, 1] * sin_angles[:, 2] - sin_angles[:, 0] * cos_angles[:, 2]
        rotation_matrix[:, 0, 2] = cos_angles[:, 0] * sin_angles[:, 1] * cos_angles[:, 2] + sin_angles[:, 0] * sin_angles[:, 2]
        rotation_matrix[:, 1, 0] = sin_angles[:, 0] * cos_angles[:, 1]
        rotation_matrix[:, 1, 1] = sin_angles[:, 0] * sin_angles[:, 1] * sin_angles[:, 2] + cos_angles[:, 0] * cos_angles[:, 2]
        rotation_matrix[:, 1, 2] = sin_angles[:, 0] * sin_angles[:, 1] * cos_angles[:, 2] - cos_angles[:, 0] * sin_angles[:, 2]
        rotation_matrix[:, 2, 0] = -sin_angles[:, 1]
        rotation_matrix[:, 2, 1] = cos_angles[:, 1] * sin_angles[:, 2]
        rotation_matrix[:, 2, 2] = cos_angles[:, 1] * cos_angles[:, 2]

        return rotation_matrix

    def _random_translation_vector(self, batch_size, generator = None):
        # Generate random translation vector
        translation_vector = torch.randn((batch_size, 3), device = self.device(), generator = generator) * self.s_trans
        return translation_vector
