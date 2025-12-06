#!/usr/bin/env python3
"""
应用 render_nerfacc 修复
替换 submodules/nerf_pytorch/run_nerf_mod.py 中的 render_nerfacc 函数
"""

def main():
    import re

    # 读取原文件
    with open('submodules/nerf_pytorch/run_nerf_mod.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 读取修复后的函数
    with open('render_nerfacc_fixed.py', 'r', encoding='utf-8') as f:
        fixed_content = f.read()

    # 提取修复后的函数（去掉注释和导入）
    fixed_func = '\n'.join([line for line in fixed_content.split('\n') if not line.startswith('#') or line.startswith('    #')])

    # 找到原始函数的开始和结束
    # 开始：def render_nerfacc(
    # 结束：return [rgb_final, disp_final, acc_final, extras]

    pattern = r'(def render_nerfacc\(.*?\n)(.*?)(    return \[rgb_final, disp_final, acc_final, extras\])'

    # 使用 DOTALL 模式匹配跨行
    match = re.search(pattern, content, re.DOTALL)

    if not match:
        print("❌ 未找到 render_nerfacc 函数")
        return False

    print(f"✓ 找到 render_nerfacc 函数")
    print(f"  原始函数大小: {len(match.group(2))} 字符")

    # 替换函数体
    new_content = content[:match.start()] + fixed_func + content[match.end():]

    # 写入文件
    with open('submodules/nerf_pytorch/run_nerf_mod.py', 'w', encoding='utf-8') as f:
        f.write(new_content)

    print("✓ 修复已应用到 run_nerf_mod.py")
    print("\n关键改动:")
    print("  1. estimator.sampling() 调用次数: batch_size 次 → 1 次")
    print("  2. 使用统一的 sigma_fn_unified 处理所有 rays")
    print("  3. 在 batch 循环中筛选对应的采样点")
    print("\n预期加速: 2-3×")
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
