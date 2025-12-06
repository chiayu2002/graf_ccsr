"""
Blender 数据验证脚本
检查数据目录结构、文件命名、图像是否符合 GRAF 要求
"""
import os
import glob
import re
from collections import defaultdict

def check_blender_data(data_dir='data/RS307_n'):
    """验证 Blender 生成的数据是否符合要求"""

    print("=" * 70)
    print("Blender 数据验证工具")
    print("=" * 70)

    # 1. 检查目录是否存在
    print(f"\n[1] 检查数据目录: {data_dir}")
    if not os.path.exists(data_dir):
        print(f"  ✗ 错误: 数据目录不存在！")
        print(f"  请创建目录: mkdir -p {data_dir}")
        print(f"  然后从 Blender 导出图像到此目录")
        return False
    else:
        print(f"  ✓ 目录存在")

    # 2. 检查图像文件
    print(f"\n[2] 检查图像文件")
    image_files = (glob.glob(f'{data_dir}/*.jpg') +
                   glob.glob(f'{data_dir}/*.png') +
                   glob.glob(f'{data_dir}/*.JPG') +
                   glob.glob(f'{data_dir}/*.PNG'))

    if not image_files:
        print(f"  ✗ 错误: 没有找到任何图像文件！")
        print(f"  支持格式: .jpg, .png")
        return False
    else:
        print(f"  ✓ 找到 {len(image_files)} 张图像")

    # 3. 检查文件命名格式
    print(f"\n[3] 检查文件命名格式")

    # 预期的 category 前缀
    expected_categories = {
        "0": 0,      # v=0.5
        "0.5": 1,    # v=0.4
        "1": 2,      # v=0.3
        "1.5": 3     # v=0.2
    }

    # 文件名格式: {category}_{angle}.{ext}
    pattern = re.compile(r'^([\d.]+)_(\d+)\.(jpg|png|JPG|PNG)$')

    valid_files = defaultdict(list)
    invalid_files = []

    for filepath in image_files:
        filename = os.path.basename(filepath)
        match = pattern.match(filename)

        if match:
            category = match.group(1)
            angle = int(match.group(2))

            if category in expected_categories:
                valid_files[category].append(angle)
            else:
                invalid_files.append((filename, f"未知的 category: {category}"))
        else:
            invalid_files.append((filename, "文件名格式不正确"))

    # 报告结果
    print(f"  ✓ 有效文件: {sum(len(v) for v in valid_files.values())} 张")

    if invalid_files:
        print(f"  ⚠ 无效文件: {len(invalid_files)} 张")
        print(f"\n  前 10 个无效文件:")
        for filename, reason in invalid_files[:10]:
            print(f"    - {filename}: {reason}")
        if len(invalid_files) > 10:
            print(f"    ... 还有 {len(invalid_files) - 10} 个")

    # 4. 检查每个 category 的覆盖范围
    print(f"\n[4] 检查视角覆盖范围")

    for category in expected_categories.keys():
        if category in valid_files:
            angles = sorted(valid_files[category])
            v_val = [0.5, 0.4, 0.3, 0.2][expected_categories[category]]

            print(f"\n  Category '{category}' (v={v_val}):")
            print(f"    - 图像数量: {len(angles)}")
            print(f"    - 角度范围: {min(angles)} - {max(angles)}°")
            print(f"    - 是否连续: ", end="")

            # 检查是否有缺失角度
            expected_angles = set(range(min(angles), max(angles) + 1))
            actual_angles = set(angles)
            missing = expected_angles - actual_angles

            if not missing:
                print(f"✓")
            else:
                print(f"✗ (缺失 {len(missing)} 个角度)")
                if len(missing) <= 20:
                    print(f"      缺失角度: {sorted(missing)}")
                else:
                    sample = sorted(missing)[:10]
                    print(f"      缺失角度示例: {sample}... (共 {len(missing)} 个)")

            # 检查重复
            duplicates = [angle for angle in set(angles) if angles.count(angle) > 1]
            if duplicates:
                print(f"    ⚠ 重复角度: {duplicates}")

        else:
            print(f"\n  Category '{category}':")
            print(f"    ✗ 没有找到任何图像")

    # 5. 检查图像尺寸（如果 PIL 可用）
    try:
        from PIL import Image

        print(f"\n[5] 检查图像尺寸")

        # 随机抽样检查
        sample_files = image_files[:min(10, len(image_files))]
        sizes = set()

        for filepath in sample_files:
            try:
                img = Image.open(filepath)
                sizes.add(img.size)
            except Exception as e:
                print(f"  ⚠ 无法打开 {os.path.basename(filepath)}: {e}")

        if len(sizes) == 1:
            size = list(sizes)[0]
            print(f"  ✓ 图像尺寸一致: {size[0]}x{size[1]}")

            # 检查是否为正方形
            if size[0] != size[1]:
                print(f"  ⚠ 警告: 图像不是正方形！GRAF 预期正方形图像")

            # 检查是否匹配配置
            expected_size = 256  # from config
            if size[0] != expected_size or size[1] != expected_size:
                print(f"  ⚠ 警告: 图像尺寸 {size[0]}x{size[1]} 不匹配配置 {expected_size}x{expected_size}")
                print(f"      图像会被 resize，可能影响质量")
        elif len(sizes) > 1:
            print(f"  ✗ 图像尺寸不一致!")
            print(f"    发现的尺寸: {sizes}")
        else:
            print(f"  ⚠ 无法检查图像尺寸")

    except ImportError:
        print(f"\n[5] 跳过图像尺寸检查 (需要 PIL/Pillow)")

    # 6. 总结和建议
    print(f"\n" + "=" * 70)
    print("总结")
    print("=" * 70)

    total_valid = sum(len(v) for v in valid_files.values())
    total_expected = 4 * 360  # 4 categories × 360 angles

    print(f"\n总计:")
    print(f"  - 有效图像: {total_valid} 张")
    print(f"  - 理论完整: {total_expected} 张")
    print(f"  - 完成度: {total_valid / total_expected * 100:.1f}%")

    if total_valid < 100:
        print(f"\n⚠ 警告: 图像数量太少，建议至少 144 张（4 categories × 36 angles）")
        print(f"  当前只有 {total_valid} 张，可能影响训练效果")

    if len(valid_files) < 4:
        print(f"\n⚠ 警告: 缺少某些 category 的数据")
        missing_cats = set(expected_categories.keys()) - set(valid_files.keys())
        print(f"  缺失的 categories: {missing_cats}")

    print(f"\n建议:")
    if invalid_files:
        print(f"  1. 修正 {len(invalid_files)} 个无效文件的命名")
    if total_valid < total_expected:
        print(f"  2. 从 Blender 生成更多视角的图像")
    print(f"  3. 运行 'python check_data.py' 测试数据加载")

    print(f"\n" + "=" * 70)
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='验证 Blender 生成的数据')
    parser.add_argument('--data_dir', default='data/RS307_n',
                        help='数据目录路径 (默认: data/RS307_n)')
    args = parser.parse_args()

    check_blender_data(args.data_dir)
