import torch
import torch.nn as nn
import torch_npu
from typing import Optional, Set, List, Union
from functools import wraps
import weakref


# ============== FP8 量化核心函数 ==============


def _fp8_quant_matmul_op(
    x: torch.Tensor, 
    w: torch.Tensor, 
    bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """FP8 量化矩阵乘法"""

    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    if w.dtype != torch.bfloat16:
        w = w.to(torch.bfloat16)
    
    # 动态量化输入和权重
    x_fp8, x_scale = torch_npu.npu_dynamic_mx_quant(
        x, dst_type=torch_npu.float8_e4m3fn
    )
    w_fp8, w_scale = torch_npu.npu_dynamic_mx_quant(
        w, dst_type=torch_npu.float8_e4m3fn
    )

    # 处理 bias
    # TODO: BIAS MAY BE BFLOAT16
    # https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/torchnpuCustomsapi/docs/context/torch_npu-npu_quant_matmul.md 
    if bias is not None and bias.dtype != torch.float32:
        bias = bias.to(torch.float32)
    
    output = torch_npu.npu_quant_matmul(
        x_fp8,
        w_fp8.T,
        w_scale,
        scale_dtype=torch_npu.float8_e8m0fnu,
        pertoken_scale=x_scale,
        pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
        bias=bias,
        output_dtype=torch.bfloat16,
        group_sizes=[1, 1, 32],
    )
    return output


def fp8_quant_matmul(
    x: torch.Tensor, 
    w: torch.Tensor, 
    bias: Optional[torch.Tensor] = None
):
    if x.ndim >= 3:
        x_reshpe = x.reshape(x.shape[:-1].numel(), -1)
    
    output = _fp8_quant_matmul_op(
        x_reshpe,
        w,
        bias
    )
    
    if x.ndim >= 3:
        new_size = list(x.shape)[:-1]
        new_size.append(output.shape[1])
        output = output.view(*new_size)
    
    return output


# ============== Linear FP8 Monkey Patch 管理器 ==============


class FP8LinearPatcher:
    """
    FP8 量化 Linear 层的 Monkey Patch 管理器
    
    支持功能：
    - 全局替换所有 nn.Linear
    - 选择性替换指定模块
    - 白名单/黑名单模式
    - 动态启用/禁用
    - 统计量化层信息
    """
    
    _instance = None
    _original_forward = None
    _is_patched = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_state()
        return cls._instance
    
    def _init_state(self):
        """初始化状态"""
        self.enabled = True
        self.patched_modules: Set[int] = set()  # 存储已 patch 的模块 id
        self.whitelist: Optional[Set[str]] = None  # 白名单（模块名）
        self.blacklist: Set[str] = set()  # 黑名单（模块名）
        self.min_features = 0  # 最小特征维度（太小的不量化）
        self.stats = {
            "total_calls": 0,
            "quantized_calls": 0,
            "skipped_calls": 0,
        }
        self._module_names: dict = {}  # 模块到名称的映射
    
    @classmethod
    def get_instance(cls) -> 'FP8LinearPatcher':
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def _create_quantized_forward(self, original_forward):
        """创建量化版本的 forward 方法"""
        patcher = self
        
        @wraps(original_forward)
        def quantized_forward(self_module, input: torch.Tensor) -> torch.Tensor:
            patcher.stats["total_calls"] += 1
            
            # 检查是否应该使用量化
            should_quantize = (
                patcher.enabled
                and patcher._should_quantize_module(self_module)
                and input.is_npu  # 确保在 NPU 上
            )
            
            if should_quantize:
                patcher.stats["quantized_calls"] += 1
                return fp8_quant_matmul(input, self_module.weight, self_module.bias)
            else:
                patcher.stats["skipped_calls"] += 1
                return original_forward(self_module, input)
        
        return quantized_forward
    
    def _should_quantize_module(self, module: nn.Linear) -> bool:
        """判断模块是否应该被量化"""
        module_id = id(module)
        module_name = self._module_names.get(module_id, "")
        
        # 检查最小特征维度
        if module.in_features < self.min_features or module.out_features < self.min_features:
            return False
        
        # 白名单模式：只量化白名单中的模块
        if self.whitelist is not None:
            return any(name in module_name for name in self.whitelist)
        
        # 黑名单模式：不量化黑名单中的模块
        if self.blacklist:
            return not any(name in module_name for name in self.blacklist)
        
        return True
    
    def patch(self):
        """应用 monkey patch 到 nn.Linear"""
        if FP8LinearPatcher._is_patched:
            print("Warning: nn.Linear is already patched")
            return self
        
        # 保存原始 forward 方法
        FP8LinearPatcher._original_forward = nn.Linear.forward
        
        # 创建新的 forward 方法
        new_forward = self._create_quantized_forward(FP8LinearPatcher._original_forward)
        
        # 替换 forward 方法
        nn.Linear.forward = new_forward
        
        FP8LinearPatcher._is_patched = True
        print("✓ nn.Linear has been patched with FP8 quantization")
        
        return self
    
    def unpatch(self):
        """移除 monkey patch，恢复原始 forward"""
        if not FP8LinearPatcher._is_patched:
            print("Warning: nn.Linear is not patched")
            return self
        
        if FP8LinearPatcher._original_forward is not None:
            nn.Linear.forward = FP8LinearPatcher._original_forward
            FP8LinearPatcher._is_patched = False
            print("✓ nn.Linear has been restored to original")
        
        return self
    
    def register_model(self, model: nn.Module, prefix: str = ""):
        """
        注册模型，建立模块名称映射
        这样可以支持按名称过滤
        """
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                full_name = f"{prefix}.{name}" if prefix else name
                self._module_names[id(module)] = full_name
                self.patched_modules.add(id(module))
        
        print(f"✓ Registered {len(self.patched_modules)} Linear layers")
        return self
    
    def set_whitelist(self, patterns: List[str]):
        """设置白名单（只有匹配的层会被量化）"""
        self.whitelist = set(patterns)
        print(f"✓ Whitelist set: {patterns}")
        return self
    
    def set_blacklist(self, patterns: List[str]):
        """设置黑名单（匹配的层不会被量化）"""
        self.blacklist = set(patterns)
        print(f"✓ Blacklist set: {patterns}")
        return self
    
    def set_min_features(self, min_features: int):
        """设置最小特征维度"""
        self.min_features = min_features
        print(f"✓ Min features set to: {min_features}")
        return self
    
    def enable(self):
        """启用量化"""
        self.enabled = True
        print("✓ FP8 quantization enabled")
        return self
    
    def disable(self):
        """禁用量化（使用原始 forward）"""
        self.enabled = False
        print("✓ FP8 quantization disabled")
        return self
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            "total_calls": 0,
            "quantized_calls": 0,
            "skipped_calls": 0,
        }
        return self
    
    def print_stats(self):
        """打印统计信息"""
        print("\n" + "="*50)
        print(" FP8 Quantization Statistics")
        print("="*50)
        print(f"  Total forward calls:     {self.stats['total_calls']}")
        print(f"  Quantized calls:         {self.stats['quantized_calls']}")
        print(f"  Skipped calls:           {self.stats['skipped_calls']}")
        if self.stats['total_calls'] > 0:
            ratio = self.stats['quantized_calls'] / self.stats['total_calls'] * 100
            print(f"  Quantization ratio:      {ratio:.1f}%")
        print("="*50 + "\n")
        return self
    
    def print_registered_modules(self):
        """打印已注册的模块"""
        print("\n" + "="*50)
        print(" Registered Linear Modules")
        print("="*50)
        for module_id, name in self._module_names.items():
            status = "✓" if self._should_quantize_module_by_name(name) else "✗"
            print(f"  {status} {name}")
        print("="*50 + "\n")
        return self
    
    def _should_quantize_module_by_name(self, name: str) -> bool:
        """根据名称判断是否量化"""
        if self.whitelist is not None:
            return any(pattern in name for pattern in self.whitelist)
        if self.blacklist:
            return not any(pattern in name for pattern in self.blacklist)
        return True


# ============== 便捷函数 ==============

def patch_linear_with_fp8():
    """便捷函数：启用 FP8 量化"""
    return FP8LinearPatcher.get_instance().patch()


def unpatch_linear():
    """便捷函数：禁用 FP8 量化"""
    return FP8LinearPatcher.get_instance().unpatch()


def get_fp8_patcher() -> FP8LinearPatcher:
    """获取 patcher 实例"""
    return FP8LinearPatcher.get_instance()


# ============== 上下文管理器 ==============

class fp8_quantization:
    """
    上下文管理器，用于临时启用/禁用 FP8 量化
    
    Usage:
        with fp8_quantization(enabled=True):
            output = model(input)
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.patcher = FP8LinearPatcher.get_instance()
        self.previous_state = None
    
    def __enter__(self):
        self.previous_state = self.patcher.enabled
        self.patcher.enabled = self.enabled
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.patcher.enabled = self.previous_state
        return False


# ============== 使用示例 ==============

if __name__ == "__main__":
    # 设置设备
    device = "npu"
    torch_npu.npu.set_device(0)
    
    # =====================
    # 示例 1: 基础使用
    # =====================
    print("\n" + "="*60)
    print(" 示例 1: 基础使用")
    print("="*60)
    
    # 创建一个简单的模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(256, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 10)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    model = SimpleModel().to(torch.bfloat16).to(device)
    x = torch.randn(2, 256, dtype=torch.bfloat16, device=device)
    
    # 应用 patch
    patcher = patch_linear_with_fp8()
    patcher.register_model(model)
    
    # 前向传播（使用 FP8 量化）
    with torch.no_grad():
        output_quant = model(x)
    
    print(f"Output shape: {output_quant.shape}")
    patcher.print_stats()
    
    # =====================
    # 示例 2: 白名单模式
    # =====================
    print("\n" + "="*60)
    print(" 示例 2: 白名单模式（只量化 fc1 和 fc2）")
    print("="*60)
    
    patcher.reset_stats()
    patcher.set_whitelist(["fc1", "fc2"])
    patcher.print_registered_modules()
    
    with torch.no_grad():
        output_whitelist = model(x)
    
    patcher.print_stats()
    
    # =====================
    # 示例 3: 黑名单模式
    # =====================
    print("\n" + "="*60)
    print(" 示例 3: 黑名单模式（不量化 fc3）")
    print("="*60)
    
    patcher.whitelist = None  # 清除白名单
    patcher.reset_stats()
    patcher.set_blacklist(["fc3"])
    patcher.print_registered_modules()
    
    with torch.no_grad():
        output_blacklist = model(x)
    
    patcher.print_stats()
    
    # =====================
    # 示例 4: 上下文管理器
    # =====================
    print("\n" + "="*60)
    print(" 示例 4: 使用上下文管理器临时禁用量化")
    print("="*60)
    
    patcher.blacklist = set()  # 清除黑名单
    patcher.reset_stats()
    
    # 临时禁用量化
    with fp8_quantization(enabled=False):
        with torch.no_grad():
            output_fp = model(x)
        print("在上下文中禁用了量化")
    
    # 上下文外恢复量化
    with torch.no_grad():
        output_quant2 = model(x)
    print("上下文外恢复了量化")
    
    patcher.print_stats()
    
    # =====================
    # 示例 5: 精度对比
    # =====================
    print("\n" + "="*60)
    print(" 示例 5: FP8 量化 vs 原始浮点精度对比")
    print("="*60)
    
    # 禁用量化，获取浮点结果
    patcher.disable()
    with torch.no_grad():
        ref_output = model(x)
    
    # 启用量化，获取量化结果
    patcher.enable()
    patcher.reset_stats()
    with torch.no_grad():
        quant_output = model(x)
    
    # 计算精度差异
    diff = (ref_output.float() - quant_output.float()).abs()
    cosine_sim = torch.nn.functional.cosine_similarity(
        ref_output.float().flatten().unsqueeze(0),
        quant_output.float().flatten().unsqueeze(0)
    )
    
    print(f"  最大绝对误差: {diff.max().item():.6e}")
    print(f"  平均绝对误差: {diff.mean().item():.6e}")
    print(f"  余弦相似度:   {cosine_sim.item():.6f}")
    
    # =====================
    # 示例 6: 移除 patch
    # =====================
    print("\n" + "="*60)
    print(" 示例 6: 移除 Patch")
    print("="*60)
    
    unpatch_linear()
    
    # 验证已恢复原始行为
    with torch.no_grad():
        output_original = model(x)
    
    print("nn.Linear 已恢复原始行为")


# ============== 针对 Transformer 模型的高级用法 ==============

def patch_transformer_model(
    model: nn.Module,
    quantize_qkv: bool = True,
    quantize_proj: bool = True,
    quantize_mlp: bool = True,
    quantize_lm_head: bool = False,
    min_features: int = 128,
):
    """
    针对 Transformer 模型的智能 patch
    
    Args:
        model: Transformer 模型
        quantize_qkv: 是否量化 QKV 投影层
        quantize_proj: 是否量化输出投影层
        quantize_mlp: 是否量化 MLP 层
        quantize_lm_head: 是否量化 LM Head
        min_features: 最小特征维度
    """
    patcher = patch_linear_with_fp8()
    patcher.register_model(model)
    patcher.set_min_features(min_features)
    
    # 构建黑名单
    blacklist = []
    
    if not quantize_qkv:
        blacklist.extend(["q_proj", "k_proj", "v_proj", "query", "key", "value"])
    
    if not quantize_proj:
        blacklist.extend(["o_proj", "out_proj", "dense"])
    
    if not quantize_mlp:
        blacklist.extend(["mlp", "fc1", "fc2", "gate_proj", "up_proj", "down_proj"])
    
    if not quantize_lm_head:
        blacklist.extend(["lm_head", "output", "classifier"])
    
    if blacklist:
        patcher.set_blacklist(blacklist)
    
    patcher.print_registered_modules()
    
    return patcher
