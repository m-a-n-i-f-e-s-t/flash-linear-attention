# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.power_attn.configuration_power_attn import PowerAttentionConfig
from fla.models.power_attn.modeling_power_attn import PowerAttentionForCausalLM, PowerAttentionModel

AutoConfig.register(PowerAttentionConfig.model_type, PowerAttentionConfig)
AutoModel.register(PowerAttentionConfig, PowerAttentionModel)
AutoModelForCausalLM.register(PowerAttentionConfig, PowerAttentionForCausalLM)


__all__ = ['PowerAttentionConfig', 'PowerAttentionForCausalLM', 'PowerAttentionModel']
