# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.og_sympow.configuration_og_sympow import OGSympowConfig
from fla.models.og_sympow.modeling_og_sympow import OGSympowForCausalLM, OGSympowModel

AutoConfig.register(OGSympowConfig.model_type, OGSympowConfig)
AutoModel.register(OGSympowConfig, OGSympowModel)
AutoModelForCausalLM.register(OGSympowConfig, OGSympowForCausalLM)


__all__ = ['OGSympowConfig', 'OGSympowForCausalLM', 'OGSympowModel']
