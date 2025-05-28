# Copyright 2023 Haotian Liu
#
# This file is part of the Vary project, originally located at:
# https://github.com/Ucas-HaoranWei/Vary-toy/blob/main/Vary-master/vary/model/vary_opt.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    OPTConfig,
    OPTForCausalLM,
    OPTModel,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from docling_ibm_models.code_formula_model.models.sam import build_sam_vit_b


class SamOptConfig(OPTConfig):
    model_type = "sam_opt"

    def __init__(
        self,
        sam_image_size=1024,
        sam_mm_projector_in=1024,
        sam_mm_projector_out=768,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sam_image_size = sam_image_size
        self.sam_mm_projector_in = sam_mm_projector_in
        self.sam_mm_projector_out = sam_mm_projector_out


class SamOPTModel(OPTModel):
    config_class = SamOptConfig  # type: ignore

    def __init__(self, config: OPTConfig):
        super(SamOPTModel, self).__init__(config)
        self.vision_tower = build_sam_vit_b(image_size=config.sam_image_size)

        self.mm_projector = nn.Linear(
            config.sam_mm_projector_in, config.sam_mm_projector_out
        )

    def embed_tokens(self, x):
        return self.get_input_embeddings()(x)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        vision_tower = getattr(self, "vision_tower", None)
        im_start_token = getattr(self.config, "im_start_token", -1)  # type: ignore

        if input_ids.shape[1] != 1 or self.training:  # type: ignore
            with torch.set_grad_enabled(self.training):  # type: ignore
                assert vision_tower is not None
                image_features = vision_tower(images)
                image_features = image_features.flatten(2).permute(0, 2, 1)
                image_features = self.mm_projector(image_features)

            new_input_embeds = []
            for cur_input_ids, cur_input_embeds, cur_image_features in zip(
                input_ids, inputs_embeds, image_features
            ):
                image_start_token_position = int(
                    torch.where(cur_input_ids == im_start_token)[0].item()
                )  # cast to int for mypy

                cur_image_features = cur_image_features.to(
                    device=cur_input_embeds.device
                )
                num_patches = cur_image_features.shape[0]
                cur_input_embeds = torch.cat(
                    (
                        cur_input_embeds[: image_start_token_position + 1],
                        cur_image_features,
                        cur_input_embeds[
                            image_start_token_position + num_patches + 1 :
                        ],
                    ),
                    dim=0,
                )

                new_input_embeds.append(cur_input_embeds)

            inputs_embeds = torch.stack(new_input_embeds, dim=0)  # type: ignore

        return super(SamOPTModel, self).forward(  # type: ignore
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class SamOPTForCausalLM(OPTForCausalLM):
    config_class = SamOptConfig  # type: ignore

    def __init__(self, config):
        super(OPTForCausalLM, self).__init__(config)
        self.model = SamOPTModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions  # type: ignore
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states  # type: ignore
        )

        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            images=images,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states).contiguous()

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        token_type_ids = kwargs.get("token_type_ids", None)
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs


AutoConfig.register("sam_opt", SamOptConfig)
AutoModelForCausalLM.register(SamOptConfig, SamOPTForCausalLM)
