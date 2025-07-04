# Copyright (c) 2025 ZO2 Implementation for Mixtral
# Licensed under the Apache License, Version 2.0
import sys
sys.path.append("../")

from transformers import MixtralForCausalLM, MixtralConfig
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union
from dataclasses import dataclass
import math

from zo2 import ZOConfig
from zo2.model.base import BaseZOModel
from zo2.optimizer.mezo_sgd.zo2 import MeZO2SGD
from zo2.config.mezo_sgd import MeZOSGDConfig

# 从transformers导入Mixtral相关组件
from transformers.models.mixtral.modeling_mixtral import (
    MixtralConfig,
    MixtralPreTrainedModel,
    MixtralModel,
    MixtralForCausalLM,
    MixtralDecoderLayer,
    MixtralSparseMoeBlock,
    MixtralBlockSparseTop2MLP,
    MixtralAttention,
    MixtralRMSNorm,
    MixtralRotaryEmbedding,
    BaseModelOutputWithPast,
    MoeCausalLMOutputWithPast,
    load_balancing_loss_func
)

# ======================= 工具函数 =======================

def get_shift_logits(logits):
    """从logits中提取shift_logits用于语言模型损失计算"""
    return logits[..., :-1, :].contiguous()

def get_shift_labels(labels):
    """从labels中提取shift_labels用于语言模型损失计算"""
    return labels[..., 1:].contiguous()

def get_moe_expert_outputs(expert_outputs, routing_weights, selected_experts, final_hidden_states):
    """处理MoE expert输出并应用routing权重"""
    # 这个函数需要根据MoE的具体实现来调整
    return expert_outputs * routing_weights

def get_hidden_states_from_outputs(outputs):
    """从模型输出中提取hidden_states"""
    if hasattr(outputs, 'last_hidden_state'):
        return outputs.last_hidden_state
    elif isinstance(outputs, tuple):
        return outputs[0]
    else:
        return outputs

# ======================= MixtralZO2 优化器 =======================

class MixtralZO2Optimizer(MeZO2SGD):
    """
    专门为Mixtral模型设计的ZO2优化器
    重点优化MoE中gate选定的experts
    """
    
    def init_zo2_upload(self):
        """初始化上传关键组件到GPU"""
        print("Upload embeddings and head to cuda...")
        
        # 上传embedding层
        self.model.model.embed_tokens = self.model.model.embed_tokens.to(self.device)
        
        # 上传最终层归一化和语言模型头
        self.model.model.norm = self.model.model.norm.to(self.device)
        self.model.lm_head = self.model.lm_head.to(self.device)
        
        # 上传旋转位置编码
        self.model.model.rotary_emb = self.model.model.rotary_emb.to(self.device)
        
        # 处理decoder层的上传/卸载
        self.num_layers = len(self.model.model.layers)
        if self.offloading_blocks is not None:
            self.offloading_blocks = self.offloading_blocks
        else:
            self.offloading_blocks = list(range(self.num_layers))
        
        print(f"Decoder layers {self.offloading_blocks} will be offloaded to {self.offloading_device}")
        
        # 初始化时保持某些层在GPU上
        for i in range(self.num_layers):
            if i not in self.offloading_blocks:
                self.model.model.layers[i] = self.model.model.layers[i].to(self.device)
                print(f"Keep layer {i} on {self.device}")
    
    @torch.inference_mode()
    def inner_zo_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MixtralZO2的前向传播实现
        返回两个损失值用于梯度估计
        """
        
        # 1. 处理输入embedding
        if inputs_embeds is None:
            inputs_embeds1, inputs_embeds2 = self.task_compute_module(
                self.model.model.embed_tokens,
                inputs1={"input_ids": input_ids},
                inputs2={"input_ids": input_ids},
                grad=self.projected_grad
            )
        else:
            inputs_embeds1 = inputs_embeds2 = inputs_embeds
        
        # 2. 计算位置编码
        if position_ids is None:
            batch_size, seq_len = inputs_embeds1.shape[:2]
            position_ids = torch.arange(seq_len, device=inputs_embeds1.device).unsqueeze(0).expand(batch_size, -1)
        
        # 获取旋转位置编码
        position_embeddings1, position_embeddings2 = self.task_compute_module(
            self.model.model.rotary_emb,
            inputs1={"x": inputs_embeds1, "position_ids": position_ids},
            inputs2={"x": inputs_embeds2, "position_ids": position_ids},
            grad=self.projected_grad,
            compute_sync=False
        )
        
        # 3. 处理注意力掩码
        if attention_mask is None:
            batch_size, seq_len = inputs_embeds1.shape[:2]
            attention_mask = torch.ones(batch_size, seq_len, device=inputs_embeds1.device)
        
        # 创建因果掩码
        causal_mask1, causal_mask2 = self.task_compute_function(
            self._create_causal_mask,
            inputs1={"attention_mask": attention_mask, "input_shape": inputs_embeds1.shape},
            inputs2={"attention_mask": attention_mask, "input_shape": inputs_embeds2.shape},
            compute_sync=False
        )
        
        # 4. 初始化隐藏状态
        hidden_states1, hidden_states2 = inputs_embeds1, inputs_embeds2
        
        # 5. 上传第一层decoder
        if 0 in self.offloading_blocks:
            self.model.model.layers[0] = self.task_upload(
                module=self.model.model.layers[0],
                device=self.device
            )
        
        # 6. 逐层处理decoder
        N = len(self.model.model.layers)
        for i in range(1, N):
            # 卸载前面的层
            if i != 1 and (i-2) in self.offloading_blocks:
                self.model.model.layers[i-2] = self.task_offload(
                    module=self.model.model.layers[i-2],
                    device=self.offloading_device
                )
            
            # 计算当前层
            hidden_states1, hidden_states2 = self.task_compute_decoder_layer(
                layer=self.model.model.layers[i-1],
                hidden_states1=hidden_states1,
                hidden_states2=hidden_states2,
                attention_mask=causal_mask1,  # 两个mask应该相同
                position_embeddings=position_embeddings1,  # 位置编码也相同
                grad=self.projected_grad,
                output_router_logits=output_router_logits
            )
            
            # 上传下一层
            if i in self.offloading_blocks:
                self.model.model.layers[i] = self.task_upload(
                    module=self.model.model.layers[i],
                    device=self.device
                )
        
        # 7. 处理最后一层
        if N-2 in self.offloading_blocks:
            self.model.model.layers[N-2] = self.task_offload(
                module=self.model.model.layers[N-2],
                device=self.offloading_device
            )
        
        # 计算最后一层
        hidden_states1, hidden_states2 = self.task_compute_decoder_layer(
            layer=self.model.model.layers[N-1],
            hidden_states1=hidden_states1,
            hidden_states2=hidden_states2,
            attention_mask=causal_mask1,
            position_embeddings=position_embeddings1,
            grad=self.projected_grad,
            output_router_logits=output_router_logits
        )
        
        # 卸载最后一层
        if N-1 in self.offloading_blocks:
            self.model.model.layers[N-1] = self.task_offload(
                module=self.model.model.layers[N-1],
                device=self.offloading_device
            )
        
        # 8. 最终层归一化
        hidden_states1, hidden_states2 = self.task_compute_module(
            self.model.model.norm,
            inputs1={"hidden_states": hidden_states1},
            inputs2={"hidden_states": hidden_states2},
            grad=self.projected_grad,
            weight_decay=0.0  # 通常LayerNorm不应用weight decay
        )
        
        # 9. 语言模型头
        logits1, logits2 = self.task_compute_module(
            self.model.lm_head,
            inputs1={"hidden_states": hidden_states1},
            inputs2={"hidden_states": hidden_states2},
            grad=self.projected_grad
        )
        
        # 10. 计算损失
        if labels is not None:
            # 计算语言模型损失
            shift_logits1, shift_logits2 = self.task_compute_function(
                get_shift_logits,
                inputs1={"logits": logits1},
                inputs2={"logits": logits2}
            )
            
            shift_labels1, shift_labels2 = self.task_compute_function(
                get_shift_labels,
                inputs1={"labels": labels},
                inputs2={"labels": labels}
            )
            
            loss1, loss2 = self.task_compute_function(
                F.cross_entropy,
                inputs1={
                    "input": shift_logits1.view(-1, shift_logits1.size(-1)),
                    "target": shift_labels1.view(-1)
                },
                inputs2={
                    "input": shift_logits2.view(-1, shift_logits2.size(-1)),
                    "target": shift_labels2.view(-1)
                }
            )
            
            return loss1, loss2
        else:
            # 如果没有标签，返回假的损失用于测试
            dummy_loss1 = torch.tensor(0.0, device=logits1.device)
            dummy_loss2 = torch.tensor(0.0, device=logits2.device)
            return dummy_loss1, dummy_loss2
    
    def task_compute_decoder_layer(
        self,
        layer: MixtralDecoderLayer,
        hidden_states1: torch.Tensor,
        hidden_states2: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor,
        grad: float,
        output_router_logits: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算单个decoder层，特别处理MoE部分
        """
        # 这里需要手动实现decoder层的前向传播以支持双路径
        # 由于MoE的复杂性，我们需要分解每个组件
        
        # 1. 输入层归一化
        normed_hidden_states1, normed_hidden_states2 = self.task_compute_module(
            layer.input_layernorm,
            inputs1={"hidden_states": hidden_states1},
            inputs2={"hidden_states": hidden_states2},
            grad=grad,
            weight_decay=0.0
        )
        
        # 2. 自注意力机制
        attn_output1, attn_output2 = self.task_compute_attention(
            attention_layer=layer.self_attn,
            hidden_states1=normed_hidden_states1,
            hidden_states2=normed_hidden_states2,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            grad=grad
        )
        
        # 3. 残差连接
        hidden_states1, hidden_states2 = self.task_compute_function(
            torch.add,
            inputs1={"input": hidden_states1, "other": attn_output1},
            inputs2={"input": hidden_states2, "other": attn_output2}
        )
        
        # 4. 后注意力层归一化
        normed_hidden_states1, normed_hidden_states2 = self.task_compute_module(
            layer.post_attention_layernorm,
            inputs1={"hidden_states": hidden_states1},
            inputs2={"hidden_states": hidden_states2},
            grad=grad,
            weight_decay=0.0
        )
        
        # 5. MoE块 - 这是关键部分
        moe_output1, moe_output2 = self.task_compute_moe_block(
            moe_block=layer.block_sparse_moe,
            hidden_states1=normed_hidden_states1,
            hidden_states2=normed_hidden_states2,
            grad=grad,
            output_router_logits=output_router_logits
        )
        
        # 6. 最终残差连接
        hidden_states1, hidden_states2 = self.task_compute_function(
            torch.add,
            inputs1={"input": hidden_states1, "other": moe_output1},
            inputs2={"input": hidden_states2, "other": moe_output2}
        )
        
        return hidden_states1, hidden_states2
    
    def task_compute_attention(
        self,
        attention_layer: MixtralAttention,
        hidden_states1: torch.Tensor,
        hidden_states2: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor,
        grad: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算注意力层
        """
        # 由于注意力机制的复杂性，这里简化处理
        # 在实际实现中，你可能需要进一步分解Q、K、V的计算
        
        return self.task_compute_module(
            attention_layer,
            inputs1={
                "hidden_states": hidden_states1,
                "attention_mask": attention_mask,
                "position_embeddings": position_embeddings
            },
            inputs2={
                "hidden_states": hidden_states2,
                "attention_mask": attention_mask,
                "position_embeddings": position_embeddings
            },
            grad=grad
        )
    
    def task_compute_moe_block(
        self,
        moe_block: MixtralSparseMoeBlock,
        hidden_states1: torch.Tensor,
        hidden_states2: torch.Tensor,
        grad: float,
        output_router_logits: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算MoE块 - 这是ZO2的核心优化目标
        
        关键思路：
        1. 首先计算routing权重（gate）
        2. 选择top-k experts
        3. 只对选中的experts应用ZO2优化
        4. 未选中的experts保持原始权重
        """
        
        # 1. 计算gate logits
        gate_logits1, gate_logits2 = self.task_compute_module(
            moe_block.gate,
            inputs1={"hidden_states": hidden_states1},
            inputs2={"hidden_states": hidden_states2},
            grad=grad
        )
        
        # 2. 计算routing权重和选择experts
        # 注意：routing逻辑对两个路径应该是相同的
        routing_weights1, selected_experts1 = self.task_compute_function(
            self._compute_routing,
            inputs1={"gate_logits": gate_logits1, "top_k": moe_block.top_k},
            inputs2=None  # 第二个路径使用相同的routing
        )
        
        # 3. 为选中的experts计算输出
        expert_outputs1, expert_outputs2 = self.task_compute_selected_experts(
            experts=moe_block.experts,
            hidden_states1=hidden_states1,
            hidden_states2=hidden_states2,
            selected_experts=selected_experts1,
            routing_weights=routing_weights1,
            grad=grad
        )
        
        return expert_outputs1, expert_outputs2
    
    def task_compute_selected_experts(
        self,
        experts: nn.ModuleList,
        hidden_states1: torch.Tensor,
        hidden_states2: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        grad: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        只对选中的experts进行ZO2计算
        """
        batch_size, seq_len, hidden_dim = hidden_states1.shape
        hidden_states1_flat = hidden_states1.view(-1, hidden_dim)
        hidden_states2_flat = hidden_states2.view(-1, hidden_dim)
        
        final_output1 = torch.zeros_like(hidden_states1_flat)
        final_output2 = torch.zeros_like(hidden_states2_flat)
        
        # 创建expert mask
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=len(experts)
        ).permute(2, 1, 0)
        
        # 只处理被选中的experts
        for expert_idx in range(len(experts)):
            expert_mask_current = expert_mask[expert_idx]
            if expert_mask_current.sum() > 0:  # 如果这个expert被选中
                # 找到需要这个expert处理的token
                token_indices = expert_mask_current.nonzero(as_tuple=True)
                
                if len(token_indices[0]) > 0:
                    # 提取相关的hidden states
                    expert_input1 = hidden_states1_flat[token_indices[0]]
                    expert_input2 = hidden_states2_flat[token_indices[0]]
                    
                    # 对选中的expert应用ZO2
                    expert_output1, expert_output2 = self.task_compute_module(
                        experts[expert_idx],
                        inputs1={"hidden_states": expert_input1},
                        inputs2={"hidden_states": expert_input2},
                        grad=grad
                    )
                    
                    # 应用routing权重
                    weights = routing_weights[token_indices]
                    expert_output1 = expert_output1 * weights.unsqueeze(-1)
                    expert_output2 = expert_output2 * weights.unsqueeze(-1)
                    
                    # 累积到最终输出
                    final_output1[token_indices[0]] += expert_output1
                    final_output2[token_indices[0]] += expert_output2
        
        # 重塑回原始形状
        final_output1 = final_output1.view(batch_size, seq_len, hidden_dim)
        final_output2 = final_output2.view(batch_size, seq_len, hidden_dim)
        
        return final_output1, final_output2
    
    def _compute_routing(self, gate_logits, top_k):
        """计算routing权重和选择的experts"""
        routing_weights = F.softmax(gate_logits, dim=-1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        return routing_weights.to(gate_logits.dtype), selected_experts
    
    def _create_causal_mask(self, attention_mask, input_shape):
        """创建因果注意力掩码"""
        batch_size, seq_len = input_shape[:2]
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=attention_mask.device))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            causal_mask = causal_mask * attention_mask
        
        return causal_mask
    
    @torch.inference_mode()
    def inner_zo_eval_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        """
        评估时的前向传播
        """
        # 添加通信hooks来管理层的上传和卸载
        handles = self.add_zo2_eval_comm_hooks(self.model.model.layers)
        
        # 调用原始的前向传播
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            cache_position=cache_position,
            **kwargs
        )
        
        # 清理hooks
        self.clear_zo2_eval_comm_hooks(handles)
        
        return outputs


# ======================= ZO2 Mixtral模型 =======================

class ZO2MixtralForCausalLM(BaseZOModel):
    """
    集成ZO2优化的Mixtral因果语言模型
    """
    
    def __init__(self, config: MixtralConfig):
        # 首先初始化标准的Mixtral模型
        self.model = MixtralForCausalLM(config)
        
        # 初始化BaseZOModel
        BaseZOModel.__init__(self)
        
        # 存储配置，稍后初始化ZO2优化器
        self.config = config
        self.zo_config = None
        self.opt = None
        
        print(f"ZO2MixtralForCausalLM initialized with {config.num_local_experts} experts")
    
    def init_zo2_optimizer(self, zo_config: MeZOSGDConfig):
        """初始化ZO2优化器"""
        self.zo_config = zo_config
        self.opt = MixtralZO2Optimizer(model=self, config=zo_config)
        print("ZO2 optimizer initialized")
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, zo_config: MeZOSGDConfig = None, **kwargs):
        """
        从预训练模型加载并初始化ZO2版本
        """
        print(f"Loading pretrained model from {pretrained_model_name_or_path}")
        
        # 首先加载标准的Mixtral模型
        standard_model = MixtralForCausalLM.from_pretrained(
            pretrained_model_name_or_path, 
            **kwargs
        )
        
        # 创建ZO2版本的模型
        zo2_model = cls(standard_model.config)
        
        # 复制权重
        zo2_model.model.load_state_dict(standard_model.state_dict())
        
        # 如果提供了zo_config，初始化ZO2优化器
        if zo_config is not None:
            zo2_model.init_zo2_optimizer(zo_config)
        
        print("✓ ZO2 model loaded from pretrained successfully")
        return zo2_model
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        """
        前向传播，根据训练模式选择ZO2或常规前向传播
        """
        if self.zo_training and self.opt is not None:
            # ZO2训练模式
            return self.opt.zo_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                output_router_logits=output_router_logits,
                cache_position=cache_position,
                **kwargs
            )
        else:
            # 标准前向传播
            return self.model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                output_router_logits=output_router_logits,
                cache_position=cache_position,
                **kwargs
            )
    
    def __getattr__(self, name):
        """代理到内部模型的属性访问"""
        if name in ['model', 'config', 'zo_config', 'opt', 'zo_training']:
            return super().__getattribute__(name)
        return getattr(self.model, name)


# ======================= 配置和工厂函数 =======================

@dataclass
class MixtralZO2Config:
    """Mixtral ZO2特定配置"""
    # 模型配置
    model_name: str = "mixtral-8x7b"
    
    # ZO2配置
    zo2: bool = True
    learning_rate: float = 1e-4
    weight_decay: float = 1e-1
    zo_eps: float = 1e-3
    
    # 设备配置
    working_device: str = "cuda:0"
    offloading_device: str = "cpu"
    
    # 内存优化
    offloading_blocks: Optional[List[int]] = None
    overlap: bool = True
    
    # 精度配置
    amp: bool = False
    amp_precision: torch.dtype = torch.float16
    precision_on_working_device: torch.dtype = torch.float32
    precision_on_offloading_device: torch.dtype = torch.float16
    
    # 优化方法
    compute_module_optimize_method: str = ""
    compute_function_optimize_method: str = ""
    communicate_optimize_method: str = ""
    
    # MoE特定配置
    expert_parallel: bool = False  # 是否并行处理experts
    selective_expert_training: bool = True  # 是否只训练选中的experts


def create_zo2_mixtral_model(
    pretrained_model_path: str,
    zo2_config: MixtralZO2Config,
    **kwargs
) -> ZO2MixtralForCausalLM:
    """
    创建ZO2 Mixtral模型的工厂函数
    
    Args:
        pretrained_model_path: 预训练模型路径
        zo2_config: ZO2优化配置
        **kwargs: 传递给from_pretrained的额外参数
    
    Returns:
        ZO2MixtralForCausalLM: 配置好的ZO2 Mixtral模型
    """
    
    # 转换为MeZOSGDConfig
    mezo_config = MeZOSGDConfig(
        zo_method="mezo-sgd",
        zo2=zo2_config.zo2,
        lr=zo2_config.learning_rate,
        weight_decay=zo2_config.weight_decay,
        eps=zo2_config.zo_eps,
        working_device=zo2_config.working_device,
        offloading_device=zo2_config.offloading_device,
        offloading_blocks=zo2_config.offloading_blocks,
        overlap=zo2_config.overlap,
        amp=zo2_config.amp,
        amp_precision=zo2_config.amp_precision,
        precision_on_working_device=zo2_config.precision_on_working_device,
        precision_on_offloading_device=zo2_config.precision_on_offloading_device,
        compute_module_optimize_method=zo2_config.compute_module_optimize_method,
        compute_function_optimize_method=zo2_config.compute_function_optimize_method,
        communicate_optimize_method=zo2_config.communicate_optimize_method
    )
    
    # 创建模型
    model = ZO2MixtralForCausalLM.from_pretrained(
        pretrained_model_path,
        zo_config=mezo_config,
        **kwargs
    )
    
    return model


# ======================= 使用示例 =======================

def example_usage():
    """使用示例"""
    
    # 检查设备
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 创建ZO2配置
    zo2_config = MixtralZO2Config(
        model_name="mixtral-8x7b",
        zo2=True,
        learning_rate=1e-4,
        weight_decay=1e-1,
        zo_eps=1e-3,
        working_device=device,
        offloading_device="cpu",
        offloading_blocks=list(range(16, 32)),  # 后半部分层卸载到CPU
        overlap=True,
        amp=False,
        selective_expert_training=True
    )
    
    # 创建模型
    try:
        print("Creating ZO2 Mixtral model...")
        model = create_zo2_mixtral_model(
            pretrained_model_path="/data2/fhe/models/Mixtral-8x7B-v0.1",
            zo2_config=zo2_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        print("✓ Model created successfully")
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        return
    
    # 准备训练数据
    batch_size = 1
    seq_len = 64  # 较小的序列长度
    vocab_size = model.config.vocab_size
    
    print(f"Preparing training data: batch_size={batch_size}, seq_len={seq_len}")
    
    # 随机生成输入数据用于测试
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len + 1))
    inputs = input_ids[:, :-1]
    labels = input_ids[:, 1:]
    
    # 训练步骤
    print("Starting ZO2 training...")
    model.zo_train()  # 设置为ZO2训练模式
    
    try:
        for step in range(3):  # 少量步骤测试
            print(f"Step {step + 1}/3...")
            loss = model(input_ids=inputs, labels=labels)
            print(f"Step {step + 1}, Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"✗ Error during training: {e}")
        return
    
    print("✓ ZO2 Mixtral training completed!")


if __name__ == "__main__":
    example_usage()