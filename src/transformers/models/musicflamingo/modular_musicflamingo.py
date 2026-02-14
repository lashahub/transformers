# Copyright 2026 NVIDIA CORPORATION and the HuggingFace Inc. team. All rights
# reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from math import pi

import torch
from torch import Tensor, broadcast_tensors, einsum, nn
from torch.amp import autocast
from torch.nn import Module

from ...cache_utils import Cache
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import BaseModelOutput, CausalLMOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ..audioflamingo3.modeling_audioflamingo3 import (
    AudioFlamingo3Encoder,
    AudioFlamingo3ForConditionalGeneration,
    AudioFlamingo3MultiModalProjector,
    AudioFlamingo3PreTrainedModel,
)
from .configuration_musicflamingo import MusicFlamingoConfig


logger = logging.get_logger(__name__)


# rotary embedding helper functions
def rotate_half(x):
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


@autocast("cuda", enabled=False)
def apply_rotary_emb(freqs, t, start_index=0, scale=1.0, seq_dim=-2):
    ori_dtype = t.dtype
    embed_dtype = torch.float64
    t = t.to(embed_dtype)
    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        if freqs.ndim == 2:
            freqs = freqs[-seq_len:].to(t)
        else:
            freqs = freqs.to(t)

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert rot_dim <= t.shape[-1], (
        f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"
    )

    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    return torch.cat((t_left, t, t_right), dim=-1).to(ori_dtype)


# classes
class MusicFlamingoRotaryEmbedding(Module):
    def __init__(
        self,
        dim,
        max_time=7200,
    ):
        super().__init__()

        self.dim = dim
        self.max_time = max_time

        self.register_buffer("cached_freqs", None, persistent=False)

        theta = max_time / (2 * pi) if max_time is not None else 50000

        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

        self.freqs = nn.Parameter(freqs, requires_grad=False)

        # dummy for device

        self.register_buffer("dummy", torch.tensor(0), persistent=False)

    @property
    def device(self):
        return self.dummy.device

    def get_axial_freqs(self, *dims):
        Colon = slice(None)
        all_freqs = []

        for ind, dim in enumerate(dims):
            pos = torch.arange(dim, device=self.device)

            freqs = self.forward(pos, seq_len=dim)

            all_axis = [None] * len(dims)
            all_axis[ind] = Colon

            new_axis_slice = (Ellipsis, *all_axis, Colon)
            all_freqs.append(freqs[new_axis_slice])

        all_freqs = broadcast_tensors(*all_freqs)
        return torch.cat(all_freqs, dim=-1)

    @autocast("cuda", enabled=False)
    def forward(self, t: Tensor, seq_len=None, offset=0):
        should_cache = seq_len is not None

        if should_cache and self.cached_freqs is not None and (offset + seq_len) <= self.cached_freqs.shape[0]:
            return self.cached_freqs[offset : (offset + seq_len)].detach()

        freqs = self.freqs

        # Scale time to keep t * freq <= 2pi
        if hasattr(self, "max_time") and self.max_time is not None:
            t = t / self.max_time * (2 * pi)

        freqs = einsum("..., f -> ... f", t.type(freqs.dtype), freqs)
        freqs = torch.repeat_interleave(freqs, 2, dim=-1)

        if should_cache:
            self.register_buffer("cached_freqs", freqs.detach(), persistent=False)

        return freqs


class MusicFlamingoPreTrainedModel(AudioFlamingo3PreTrainedModel):
    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights for MusicFlamingo-specific modules."""
        if isinstance(module, MusicFlamingoRotaryEmbedding):
            # Reinitialize freqs parameter
            dim = module.dim
            max_time = module.max_time

            theta = max_time / (2 * pi) if max_time is not None else 50000

            # Generate freqs
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

            module.freqs.data = freqs

            # Reinitialize dummy buffer
            module.dummy.data = torch.tensor(0)
        else:
            # Delegate to parent class for other modules
            super()._init_weights(module)


@auto_docstring(
    custom_intro="""
    The audio model from MusicFlamingo without any head or projection on top.
    """
)
class MusicFlamingoEncoder(AudioFlamingo3Encoder):
    """
    MusicFlamingo encoder: Whisper encoder with rotary embeddings for time information.
    """

    def __init__(self, config: MusicFlamingoConfig):
        super().__init__(config)
        self.pos_emb = MusicFlamingoRotaryEmbedding(dim=256, max_time=1200.0)

    @can_return_tuple
    def forward(
        self,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor | None = None,
        audio_times: torch.Tensor | None = None,
        **kwargs,
    ):
        r"""
        Args:
            input_features (`torch.FloatTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Log-Mel features extracted from raw audio. Use the processor/feature extractor to compute and pad
                these features from waveform input.
            input_features_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            audio_times (`torch.FloatTensor` of shape `(batch_size,)`, *optional*):
                The start time of the audio segments in seconds.
        """
        seq_len = (input_features.shape[-1] - 1) // 2 + 1  # After conv2 downsampling
        input_features_lengths = input_features_mask.sum(-1)
        input_features_lengths = (input_features_lengths - 1) // 2 + 1  # conv2 downsampling
        input_features_mask = torch.arange(seq_len, device=input_features.device) < input_features_lengths[:, None]

        # Conv front-end
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))
        inputs_embeds = inputs_embeds.permute(0, 2, 1)

        # Add positions, dropout
        hidden_states = inputs_embeds + self.embed_positions.weight
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        attention_mask = create_bidirectional_mask(
            config=self.config,
            input_embeds=hidden_states,
            attention_mask=input_features_mask,
        )

        # Transformer stack
        for layer in self.layers:
            drop = self.training and torch.rand([]) < self.layerdrop
            if not drop:
                hidden_states = layer(hidden_states, attention_mask)[0]

        # AvgPool (time/2) + LayerNorm
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.avg_pooler(hidden_states).permute(0, 2, 1)
        hidden_states = self.layer_norm(hidden_states)

        if audio_times is not None:
            times = audio_times.to(hidden_states.device)
            freqs = self.pos_emb.get_axial_freqs(times.shape[0], hidden_states.shape[-2]).to(self.conv1.weight.device)
            angle = (-times * 2 * pi).to(self.conv1.weight.device)
            angle_expanded = angle.unsqueeze(2).expand(times.shape[0], hidden_states.shape[-2], freqs.shape[-1])
            freqs = freqs * angle_expanded

            hidden_states = apply_rotary_emb(freqs, hidden_states)

        return BaseModelOutput(last_hidden_state=hidden_states)


class MusicFlamingoMultiModalProjector(AudioFlamingo3MultiModalProjector):
    pass


@auto_docstring(
    custom_intro="""
    The MusicFlamingo model which consists of a fine-tuned Whisper encoder, a multi-modal projector and a Qwen2 language model.
    """
)
class MusicFlamingoForConditionalGeneration(AudioFlamingo3ForConditionalGeneration):
    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        input_features_mask: torch.Tensor,
        audio_times: torch.Tensor | None = None,
    ) -> torch.FloatTensor:
        # Encode audio with dtype conversion and audio_times
        input_features = input_features.to(dtype=self.audio_tower.conv1.weight.dtype)
        encoder_output = self.audio_tower(
            input_features, input_features_mask=input_features_mask, audio_times=audio_times
        )
        audio_embeds = self.multi_modal_projector(encoder_output.last_hidden_state)

        # Mask according to avg pooling
        post_lengths = (input_features_mask.sum(-1) - 2) // 2 + 1
        valid_mask = torch.arange(audio_embeds.shape[1], device=post_lengths.device)[None, :] < post_lengths[:, None]
        audio_embeds = audio_embeds[valid_mask.to(audio_embeds.device)]
        return audio_embeds

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        audio_times: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        input_features_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`, *optional*):
            Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        audio_times (`torch.FloatTensor` of shape `(batch_size,)`, *optional*):
            The start time of the audio segments in seconds.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import MusicFlamingoForConditionalGeneration, AutoProcessor

        >>> model_id = "nvidia/music-flamingo-hf"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> model = MusicFlamingoForConditionalGeneration.from_pretrained(model_id, device_map="auto")

        >>> conversations = [
        >>>     [
        >>>         {
        >>>             "role": "user",
        >>>             "content": [
        >>>                 {"type": "text", "text": "Transcribe the input speech."},
        >>>                 {
        >>>                     "type": "audio",
        >>>                     "path": "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/t_837b89f2-26aa-4ee2-bdf6-f73f0dd59b26.wav",
        >>>                 },
        >>>             ],
        >>>         }
        >>>     ],
        >>>     [
        >>>         {
        >>>             "role": "user",
        >>>             "content": [
        >>>                 {
        >>>                     "type": "text",
        >>>                     "text": "This track feels really peaceful and introspective. What elements make it feel so calming and meditative?",
        >>>                 },
        >>>                 {"type": "audio", "path": "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/FPSbCAANfbJLVSwD.mp3"},
        >>>             ],
        >>>         }
        >>>     ],
        >>> ]

        >>> inputs = processor.apply_chat_template(
        >>>     conversations,
        >>>     tokenize=True,
        >>>     add_generation_prompt=True,
        >>>     return_dict=True,
        >>> ).to(model.device)

        >>> outputs = model.generate(**inputs, max_new_tokens=500)

        >>> decoded_outputs = processor.batch_decode(
        >>>     outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        >>> )
        >>> print(decoded_outputs)
        ["The spoken content of the audio is...", "The track's calming and meditative feel can be attributed to..."]
        ```"""

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if input_features is not None and input_ids is not None:
            audio_embeds = self.get_audio_features(input_features, input_features_mask, audio_times=audio_times)

            # replace text-audio token placeholders with audio embeddings
            audio_token_mask = (input_ids == self.config.audio_token_id).unsqueeze(-1)
            inputs_embeds = inputs_embeds.masked_scatter(
                audio_token_mask.to(inputs_embeds.device), audio_embeds.to(inputs_embeds.device)
            )

        outputs: CausalLMOutputWithPast = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )
        return outputs

    def prepare_inputs_for_generation(self, *args, **kwargs):
        # Overwritten -- we should not pass input_features when we are in cached decoding stage

        input_features = kwargs.pop("input_features", None)
        input_features_mask = kwargs.pop("input_features_mask", None)
        audio_times = kwargs.pop("audio_times", None)
        cache_position = kwargs.get("cache_position")

        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)

        if cache_position is not None and cache_position[0] == 0:
            # input_features should only be passed when we are not in cached decoding stage
            if input_features is not None:
                model_inputs["input_features"] = input_features
            if input_features_mask is not None:
                model_inputs["input_features_mask"] = input_features_mask
            if audio_times is not None:
                model_inputs["audio_times"] = audio_times

        return model_inputs


__all__ = ["MusicFlamingoForConditionalGeneration", "MusicFlamingoPreTrainedModel", "MusicFlamingoEncoder"]
