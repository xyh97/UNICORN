"""
Requires Transformer 4.28 and above, implementation may change according the Llama implementation
"""
import logging
import os
import time
import string
from packaging import version

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import einops
import bitsandbytes as bnb
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model
from transformers.generation import GenerationConfig


import transformers
from lavis.models.blip2_models.pos_embed import get_1d_sincos_pos_embed_from_grid

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


@registry.register_model("blip2_vicuna_instruct_video")
class Blip2VicunaInstructVideo(Blip2Base):
    """
    BLIP2 Vicuna model.
    Supported model types:
        - vicuna7b
        - vicuna13b
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_vicuna_instruct", "vicuna7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "vicuna7b": "configs/models/blip2/blip2_instruct_vicuna7b_video.yaml",
        "vicuna7b_finetuned": "configs/models/blip2/blip2_instruct_vicuna7b_finetuned_video.yaml",
        "vicuna13b": "configs/models/blip2/blip2_instruct_vicuna13b.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        num_frame=32,
        llm_model="",
        prompt="",
        max_txt_len=512,
        max_output_txt_len=512,
        apply_lemmatizer=False,
        qformer_text_input=True,
        mode="mr",
        projection="post",
        use_lora=True,
        temporal_modeling=True
    ):
        super().__init__()
        transformers_version = version.parse(transformers.__version__)
        assert transformers_version >= version.parse("4.28"), "BLIP-2 Vicuna requires transformers>=4.28"        
        from transformers import LlamaTokenizer
        from lavis.models.blip2_models.modeling_llama import LlamaForCausalLM
        
        self.projection = projection
        self.tokenizer = self.init_tokenizer(truncation_side="left")
        self.use_lora = use_lora
        self.temporal_modeling = temporal_modeling

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        
        if not qformer_text_input:
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None

        self.llm_tokenizer = LlamaTokenizer.from_pretrained(llm_model, use_fast=False, truncation_side="left")
        self.generation_config = GenerationConfig.from_pretrained(llm_model)
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 1)}
        self.llm_model = LlamaForCausalLM.from_pretrained(
            llm_model, 
            # torch_dtype=torch.float16,
            # load_in_8bit=True, 
            # device_map=device_map,
        )
        self.llm_model.to(torch.float16)
        # print(self.llm_model.config.tie_word_embeddings)
        # self.llm_model.config.use_cache = False
        # self.llm_model = self.llm_model.to_bettertransformer()
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
        # self.llm_tokenizer.pad_token = self.llm_tokenizer.unk_token

        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        # self.eos_token_id = self.llm_tokenizer(
        #     self.llm_tokenizer.eos_token, add_special_tokens=False
        # ).input_ids[0]

        for name, param in self.llm_model.named_parameters():
            param.requires_grad = False
        
        if self.projection == "post":
            hidden_size = self.Qformer.config.hidden_size
            num_attention_heads = 12
        else:
            hidden_size = self.llm_model.config.hidden_size
            num_attention_heads = 16
        if self.temporal_modeling:
            self.video_Qformer = self.init_video_Qformer(num_frame, hidden_size=hidden_size, num_attention_heads=num_attention_heads, num_hidden_layers=2)
            self.video_Qformer.cls = None
            self.video_Qformer.bert.embeddings.word_embeddings = None
            self.video_Qformer.bert.embeddings.position_embeddings = None
            for layer in self.video_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
 
        if self.temporal_modeling:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_frame, self.Qformer.config.hidden_size), requires_grad=False)
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_frame, self.llm_model.config.hidden_size), requires_grad=False)
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], num_frame)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.llm_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llm_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self.prompt = prompt
        prompt_tokens = self.llm_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

        self._lemmatizer = None

        self.qformer_text_input = qformer_text_input
        self.mode = mode

        if self.use_lora:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=8,
                target_modules=find_all_linear_names(self.llm_model),
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llm_model = get_peft_model(self.llm_model, lora_config)

    def add_time_tokens(self):
        num_bins = 75
        time_tokens = []
        for bin in range(num_bins):
            time_token = f"<t{bin}>"
            # time_token = f"<{bin}>"
            time_tokens.append(time_token)
        self.tokenizer.add_tokens(time_tokens, special_tokens=True)
        num_new_tokens = self.llm_tokenizer.add_tokens(time_tokens, special_tokens=True)
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
        # print(time_tokens)
        # print(len(self.llm_tokenizer))
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
        # avg initialization
        input_embeddings = self.Qformer.get_input_embeddings().weight.data
        output_embeddings = self.Qformer.get_output_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        input_embeddings = self.llm_model.get_input_embeddings().weight.data
        output_embeddings = self.llm_model.get_output_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens-1].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens-1].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        self.Qformer.cls = None
        self.llm_model = prepare_model_for_int8_training(self.llm_model, use_gradient_checkpointing=False)
        self.llm_model.model.embed_tokens.weight.requires_grad = True
        self.llm_model.lm_head.weight.requires_grad = True
        # logging.info(self.llm_model.lm_head.weight.shape)
        logging.info("add time tokens")

    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][1:],
                    input_ids[i][this_input_ones:]
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][1:],
                    input_atts[i][this_input_ones:]
                ])
            )
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len

    def qformer_video(self, batch_size, time_length, image_embeds, image_atts, query_tokens, text, video_mask=None):
        # batch_size, _, time_length, _, _ = video.size()
        # video = einops.rearrange(video, 'b c t h w -> (b t) c h w')
        # with self.maybe_autocast():
        #     image_embeds = self.ln_vision(self.visual_encoder(video))
        # image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(video.device)
        # query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                text,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image_embeds.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image_embeds.device)
            Qformer_atts = torch.cat([query_atts, torch.repeat_interleave(text_Qformer.attention_mask, time_length, dim=0)], dim=1)
            query_output = self.Qformer.bert(
                torch.repeat_interleave(text_Qformer.input_ids, time_length, dim=0),
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        if not self.temporal_modeling:
            inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
            inputs_llm = einops.rearrange(inputs_llm, '(b t) q h -> b t q h',b=batch_size,t=time_length)
            inputs_llm = torch.mean(inputs_llm, dim=2)
            inputs_llm = inputs_llm + self.pos_embed
            return inputs_llm
        else:
            inputs_llm = query_output.last_hidden_state[:,:query_tokens.size(1),:]
            inputs_llm = einops.rearrange(inputs_llm, '(b t) q h -> b t q h',b=batch_size,t=time_length)
            inputs_llm = torch.mean(inputs_llm, dim=2)
            if self.projection == "post":
                inputs_llm = inputs_llm + self.pos_embed
                video_output = self.video_Qformer.bert(
                        query_embeds=inputs_llm,
                        return_dict=True,
                        attention_mask=video_mask,
                        )
                video_hidden = video_output.last_hidden_state
                inputs_llm = self.llm_proj(video_hidden)
            else:
                inputs_llm = self.llm_proj(inputs_llm)
                inputs_llm = inputs_llm + self.pos_embed
                video_output = self.video_Qformer.bert(
                        query_embeds=inputs_llm,
                        return_dict=True,
                        attention_mask=video_mask,
                        )
                inputs_llm = video_output.last_hidden_state
            return inputs_llm
        
    def forward(self, samples):
        # print('-----------------')
        # print(samples["text_input"])
        # print(samples["text_output"])
        # print('-----------------')

        # get llm inputs from video
        image = samples["video"]
        if "video_mask" in samples:
            video_mask = samples["video_mask"]
        else:
            video_mask = None

        # print("reverse input:", samples['reverse_input'])
        # print("reverse output:", samples['reverse_output'])
        batch_size, _, time_length, _, _ = image.size()
        image = einops.rearrange(image, 'b c t h w -> (b t) c h w')
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        if "dvc" in self.mode:
            # print("dvc")
            # print("vid:", samples["vid"])
            # print("input:", samples['text_input'])
            # print("output:", samples['text_output'])
            inputs_llm = self.qformer_video(batch_size, time_length, image_embeds, image_atts, query_tokens, samples["text_input"], video_mask)
            if video_mask is None:
                atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)
            else:
                atts_llm = video_mask
            self.llm_tokenizer.padding_side = "right"
            self.llm_tokenizer.truncation_side = 'left'
            # temporal grounding input
            text_input_tokens = self.llm_tokenizer(
                samples['text_input'],
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image.device)

            self.llm_tokenizer.truncation_side = 'right'
            text_output_tokens = self.llm_tokenizer(
                [t + self.llm_tokenizer.eos_token for t in samples['text_output']],
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_output_txt_len,
            ).to(image.device)

            llm_tokens, input_part_targets_len = self.concat_text_input_output(
                text_input_tokens.input_ids,
                text_input_tokens.attention_mask,
                text_output_tokens.input_ids,
                text_output_tokens.attention_mask,
            )

            # do not apply loss to the padding
            targets = llm_tokens['input_ids'].masked_fill(
                llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
            )

            # do not apply loss to the text input (i.e., instruction)
            for i, l in enumerate(input_part_targets_len):
                targets[i][:l] = -100

            # do not apply loss to the query tokens
            empty_targets = (
                torch.ones(atts_llm.size(), dtype=torch.long).to(image.device).fill_(-100)
            )
            targets = torch.cat([empty_targets, targets], dim=1)

            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
            inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm, llm_tokens['attention_mask']], dim=1)


            with self.maybe_autocast():
                outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
            loss = outputs.loss
            # return {"loss": loss}
        # reverse input
        if "mr" in self.mode:
            # print("mr")
            # print(samples['reverse_input'])
            # print(samples['reverse_output'])
            reverse_inputs_llm = self.qformer_video(batch_size, time_length, image_embeds, image_atts, query_tokens, samples["reverse_input"])
            atts_llm = torch.ones(reverse_inputs_llm.size()[:-1], dtype=torch.long).to(image.device)
            self.llm_tokenizer.padding_side = "right"
            self.llm_tokenizer.truncation_side = 'left'
            reverse_input_tokens = self.llm_tokenizer(
                samples['reverse_input'],
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image.device)

            self.llm_tokenizer.truncation_side = 'right'
            reverse_output_tokens = self.llm_tokenizer(
                [t + self.llm_tokenizer.eos_token for t in samples['reverse_output']],
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_output_txt_len,
            ).to(image.device)

            reverse_llm_tokens, reverse_input_part_targets_len = self.concat_text_input_output(
                reverse_input_tokens.input_ids,
                reverse_input_tokens.attention_mask,
                reverse_output_tokens.input_ids,
                reverse_output_tokens.attention_mask,
            )

            # do not apply loss to the padding
            reverse_targets = reverse_llm_tokens['input_ids'].masked_fill(
                reverse_llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
            )

            # do not apply loss to the text input (i.e., instruction)
            for i, l in enumerate(reverse_input_part_targets_len):
                reverse_targets[i][:l] = -100

            # do not apply loss to the query tokens
            empty_targets = (
                torch.ones(atts_llm.size(), dtype=torch.long).to(image.device).fill_(-100)
            )
            reverse_targets = torch.cat([empty_targets, reverse_targets], dim=1)

            reverse_inputs_embeds = self.llm_model.get_input_embeddings()(reverse_llm_tokens['input_ids'])
            reverse_inputs_embeds = torch.cat([reverse_inputs_llm, reverse_inputs_embeds], dim=1)
            reverse_attention_mask = torch.cat([atts_llm, reverse_llm_tokens['attention_mask']], dim=1)


            with self.maybe_autocast():
                reverse_outputs = self.llm_model(
                    inputs_embeds=reverse_inputs_embeds,
                    attention_mask=reverse_attention_mask,
                    return_dict=True,
                    labels=reverse_targets,
                )
            reverse_loss = reverse_outputs.loss
            # return {"loss": reverse_loss}
        if self.mode == "dvc":
            return {"loss": loss}
        elif self.mode == "mr":
            return {"loss": reverse_loss}
        elif self.mode == "mr+dvc":
            return {"loss": loss + reverse_loss, "mr": reverse_loss, "dvc": loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1,
        num_captions=5,
        temperature=1,
    ):
        self.llm_tokenizer.padding_side = "left"

        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        elif "text_input" in samples.keys():
            prompt = samples["text_input"]
        else:
            prompt = self.prompt

        image = samples["video"]
        if "video_mask" in samples:
            video_mask = samples["video_mask"]
        else:
            video_mask = None

        bs = image.size(0)

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

        # For TextCaps
        if "ocr_tokens" in samples.keys() and "{}" in prompt[0]:
            prompt = [p.format(', '.join(samples['ocr_tokens'][i][:30])) for i, p in enumerate(prompt)]

        # inputs_llm = self.qformer_video(samples["video"], prompt)
        batch_size, _, time_length, _, _ = image.size()
        image = einops.rearrange(image, 'b c t h w -> (b t) c h w')
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        inputs_llm = self.qformer_video(batch_size, time_length, image_embeds, image_atts, query_tokens, prompt, video_mask)
        if video_mask is None:
            atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)
        else:
            atts_llm = video_mask

        llm_tokens = self.llm_tokenizer(
            prompt,
            padding="longest",
            return_tensors="pt"
        ).to(image.device)
        
        # gen_cfg = GenerationConfig.from_model_config(self.llm_model.config)
        
        # print(self.llm_model.config)
        # print(len(self.llm_tokenizer))
        with self.maybe_autocast():
            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)

            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                # top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                # eos_token_id=self.eos_token_id,
                # pad_token_id=self.llm_tokenizer.pad_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
                generation_config=self.generation_config
            )
            # print(self.llm_model.pad_token_id)

        outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
        # print(outputs)
        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]
        num_samples = len(output_text) // num_captions
        new_output = []
        for i in range(num_samples):
            new_output.append(output_text[i*num_captions:(i+1)*num_captions])

        # return output_text
        return new_output

    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=0,
        **kwargs
    ):
        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]

        if prompt:
            if prompt.count("{}") == 2:
                if 'ocr_tokens' in samples:
                    text_input = [
                        prompt.format(', '.join(samples['ocr_tokens'][i][:30]), samples["text_input"][i])
                    for i in range(len(samples["text_input"]))]
                elif 'choices' in samples:
                    text_input = []
                    for i in range(len(samples["text_input"])):
                        this_choices = [f"({string.ascii_lowercase[j]}) {ch}" for j, ch in enumerate(samples["choices"][i])]
                        this_choices = " ".join(this_choices)
                        text_input.append(prompt.format(samples["text_input"][i], this_choices))
            else:
                text_input = [prompt.format(question) for question in samples["text_input"]]
        else:
            text_input = samples["text_input"]

        samples["prompt"] = text_input

        output_text = self.generate(
            samples,
            num_beams=num_beams,
            max_length=max_len,
            min_length=min_len,
            length_penalty=length_penalty
        )

        if "apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]:
            output_text = self._lemmatize(output_text)

        return output_text

    def predict_class(
        self,
        samples,
        candidates,
        n_segments=1,
    ):
        self.llm_tokenizer.padding_side = "left"

        # If candidates is a list of lists, each sample has its candidates, then we need to iterate one by one
        if type(candidates[0]) == list:
            results = []

            for i in range(samples["image"].size(0)):
                this_sample = {
                    "image": samples["image"][i].unsqueeze(0),
                    "prompt": samples["prompt"],
                }

                if "text_input" in samples.keys():
                    this_sample["text_input"] = [samples["text_input"][i]]

                if 'context' in samples.keys():
                    this_sample['context'] = [samples["context"][i]]

                if 'history' in samples.keys():
                    this_sample['history'] = [samples["history"][i]]

                if 'caption' in samples.keys():
                    this_sample['caption'] = [samples["caption"][i]]

                this_result = self._predict_class(this_sample, candidates[i], n_segments)
                results.append(this_result)

            try:
                results = torch.cat(results, dim=0)
            except:
                results = [res.tolist()[0] for res in results]

            return results

        return self._predict_class(samples, candidates, n_segments)

    def _predict_class(
        self,
        samples,
        candidates,
        n_segments=1,
    ):
        image = samples["image"]
        prompt = samples["prompt"]

        bs = image.size(0)

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

        if "text_input" in samples.keys():
            if type(samples["text_input"][0]) == list:
                prompt = [prompt[i].format(*samples["text_input"][i]) for i in range(len(prompt))]
            else:
                prompt = [prompt[i].format(samples["text_input"][i]) for i in range(len(prompt))]

        # scienceqa
        if 'context' in samples.keys() and samples['context'] != '':
            prompt = [f'context: {samples["context"][i]}. {prompt[i]}' for i in range(len(prompt))]

        # visual dialog
        if 'history' in samples.keys() and samples['history'][0] != '':
            prompt = [f'dialog history: {samples["history"][i]}\n{prompt[i]}' for i in range(len(prompt))]

        if 'caption' in samples.keys() and samples['caption'][0] != '':
            prompt = [f'This image has the caption "{samples["caption"][i]}". {prompt[i]}' for i in range(len(prompt))]

        query_tokens = self.query_tokens.expand(bs, -1, -1)
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                prompt,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt"
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

        if image.dim() == 5:
            inputs_llm, atts_llm = [], []
            for j in range(image.size(2)):
                this_frame = image[:,:,j,:,:]
                with self.maybe_autocast():
                    frame_embeds = self.ln_vision(self.visual_encoder(this_frame))
                    frame_atts = torch.ones(frame_embeds.size()[:-1], dtype=torch.long).to(image.device)

                if self.qformer_text_input:
                    frame_query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )
                else:
                    frame_query_output = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )

                frame_inputs_llm = self.llm_proj(frame_query_output.last_hidden_state[:,:query_tokens.size(1),:])
                frame_atts_llm = torch.ones(frame_inputs_llm.size()[:-1], dtype=torch.long).to(image.device)
                inputs_llm.append(frame_inputs_llm)
                atts_llm.append(frame_atts_llm)
            inputs_llm = torch.cat(inputs_llm, dim=1)
            atts_llm = torch.cat(atts_llm, dim=1)
        else:
            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            if self.qformer_text_input:
                query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            else:
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

            inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
            atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'
        text_input_tokens = self.llm_tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            # truncation=True,
            # max_length=self.max_txt_len,
        ).to(image.device)

        empty_targets = torch.ones(atts_llm.size(), dtype=torch.long).to(image.device).fill_(-100)

        # self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'right'
        n_cands = len(candidates)
        with self.maybe_autocast(dtype=torch.bfloat16):
            all_losses = []
            for n in range(n_segments):
                seg_len = n_cands // n_segments
                if n == (n_segments - 1):
                    seg_len = n_cands - seg_len * (n_segments - 1)

                start_i = n * (n_cands // n_segments)
                end_i = start_i + seg_len

                this_output_tokens = self.llm_tokenizer(
                    candidates[start_i:end_i],
                    return_tensors="pt",
                    padding="longest",
                    # truncation=True,
                    # max_length=self.max_output_txt_len,
                ).to(image.device)

                this_input_tokens_ids = text_input_tokens.input_ids.repeat_interleave(seg_len, dim=0)
                this_input_tokens_atts = text_input_tokens.attention_mask.repeat_interleave(seg_len, dim=0)

                this_output_tokens_ids = this_output_tokens.input_ids.repeat(bs, 1)
                this_output_tokens_atts = this_output_tokens.attention_mask.repeat(bs, 1)

                this_llm_tokens, this_input_targets_len = self.concat_text_input_output(
                    this_input_tokens_ids,
                    this_input_tokens_atts,
                    this_output_tokens_ids,
                    this_output_tokens_atts
                )

                this_llm_input_ids = this_llm_tokens['input_ids']
                this_llm_atts = this_llm_tokens['attention_mask']
                # this_llm_input_ids = torch.cat([this_input_tokens_ids, this_output_tokens_ids], dim=1)
                # this_llm_atts = torch.cat([this_input_tokens_atts, this_output_tokens_atts], dim=1)

                inputs_embeds = self.llm_model.get_input_embeddings()(this_llm_input_ids)
                inputs_embeds = torch.cat([inputs_llm.repeat_interleave(seg_len, dim=0), inputs_embeds], dim=1)
                attention_mask = torch.cat([atts_llm.repeat_interleave(seg_len, dim=0), this_llm_atts], dim=1)

                this_targets = this_llm_input_ids.masked_fill(this_llm_input_ids == self.llm_tokenizer.pad_token_id, -100)
                # this_targets[:, :this_input_tokens_ids.size(1)] = -100
                for i, l in enumerate(this_input_targets_len):
                    this_targets[i][:l] = -100

                this_targets = torch.cat([empty_targets.repeat_interleave(seg_len, dim=0), this_targets], dim=1)

                outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=this_targets,
                    reduction="none",
                )

                loss = outputs.loss

                loss = loss.reshape(bs, seg_len)
                # output_class_ranks = torch.argsort(loss, dim=-1)
                all_losses.append(loss)

            all_losses = torch.cat(all_losses, dim=-1)
            output_class_ranks = torch.argsort(all_losses, dim=-1)

        return output_class_ranks

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        num_frame = cfg.get("num_frame")
        llm_model = cfg.get("llm_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        mode = cfg.get("mode", "mr")
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 200)
        max_output_txt_len = cfg.get("max_output_txt_len", 256)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        qformer_text_input = cfg.get("qformer_text_input", True)
        projection = cfg.get("projection", "post")
        use_lora = cfg.get("use_lora", True)
        temporal_modeling = cfg.get("temporal_modeling", True)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            num_frame=num_frame,
            llm_model=llm_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            max_output_txt_len=max_output_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            qformer_text_input=qformer_text_input,
            mode=mode,
            projection=projection,
            use_lora=use_lora,
            temporal_modeling=temporal_modeling
        )

        model.load_checkpoint_from_config(cfg)

        return model
