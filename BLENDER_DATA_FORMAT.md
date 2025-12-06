# Blender 数据格式要求

## 🚨 关键发现
配置文件中指定的数据目录 `data/RS307_n/` **不存在**！这可能是训练质量差的根本原因。

## 预期的数据结构

### 目录结构
```
graf_ccsr/
└── data/
    └── RS307_n/           # 配置中指定的目录
        ├── 0_0.jpg        # 第一个视角 (v=0.5, angle=0°)
        ├── 0_1.jpg        # (v=0.5, angle=1°)
        ├── 0_2.jpg        # (v=0.5, angle=2°)
        ...
        ├── 0_359.jpg      # (v=0.5, angle=359°)
        ├── 0.5_0.jpg      # 第二个视角 (v=0.4, angle=0°)
        ├── 0.5_1.jpg      # (v=0.4, angle=1°)
        ...
        ├── 1_0.jpg        # 第三个视角 (v=0.3, angle=0°)
        ...
        └── 1.5_359.jpg    # 第四个视角 (v=0.2, angle=359°)
```

### 文件命名规则

根据 `graf/datasets.py` 的代码，文件名必须遵循以下格式：

```
{category_prefix}_{angle_index}.jpg
```

**Category 映射**（对应不同的俯仰角）：
- `0_` → category 0 → v = 0.5 （俯仰角约 60°）
- `0.5_` → category 1 → v = 0.4 （俯仰角约 66°）
- `1_` → category 2 → v = 0.3 （俯仰角约 73°）
- `1.5_` → category 3 → v = 0.2 （俯仰角约 78°）

**Angle index**：0-359（方位角，环绕物体一周）

### Label 格式

每张图片对应一个 label `[dir_idx, category_idx, file_idx]`：
- `dir_idx`: 数据目录索引（如果有多个数据目录）
- `category_idx`: 俯仰角类别 (0-3)
- `file_idx`: 方位角索引 (0-359)

例如：
- `0_45.jpg` → label = `[0, 0, 45]` （第0个数据集，category 0，角度45°）
- `1_180.jpg` → label = `[0, 2, 180]` （第0个数据集，category 2，角度180°）

## Blender 导出要求

### 1. 相机参数设置

根据 `configs/default.yaml`：
```yaml
fov: 20          # 视场角 20 度（CRITICAL！）
radius: 2.5      # 相机距离中心 2.5 米
near: 1.5        # 最近渲染距离
far: 4.5         # 最远渲染距离
imsize: 256      # 图像分辨率 256x256
```

**Blender 相机设置**：
- Type: **Perspective**（透视）
- Focal Length: 根据 FOV=20° 计算
  - 对于 256x256 图像：focal_length = (sensor_width / 2) / tan(10°)
  - 如果 sensor_width = 36mm，则 focal_length ≈ 102mm
- 或者直接设置 **Camera FOV = 20°**

### 2. 相机位置计算

相机应放置在半球面上，使用球坐标系：
- **Radius (r)**: 2.5 米
- **方位角 (azimuth, u)**: 0° - 360°（水平旋转）
- **俯仰角 (elevation, v)**: 对应 4 个类别

**v 值到实际角度的转换**：
```python
# v 是球坐标中的归一化参数 (0-1)
# 实际俯仰角 theta = acos(1 - 2*v)

v = 0.5 → theta ≈ 60°  (category 0_)
v = 0.4 → theta ≈ 66°  (category 0.5_)
v = 0.3 → theta ≈ 73°  (category 1_)
v = 0.2 → theta ≈ 78°  (category 1.5_)
```

**笛卡尔坐标转换**：
```python
import numpy as np

def to_sphere(u, v):
    theta = 2 * np.pi * u  # 方位角 (azimuth)
    phi = np.arccos(1 - 2 * v)  # 俯仰角 (elevation from z-axis)

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    return np.array([x, y, z])

# 相机位置
radius = 2.5
camera_position = to_sphere(u, v) * radius
```

### 3. Blender 坐标系注意事项

**GRAF 坐标系** vs **Blender 坐标系**：
- GRAF: Y-up（可能）
- Blender: Z-up

**可能需要坐标转换**：
```python
# Blender → GRAF
position_graf = [position_blender[0],
                 position_blender[2],   # Z → Y
                 -position_blender[1]]  # Y → -Z (可能需要反转)
```

### 4. 建议的 Blender Python 脚本

创建一个 Python 脚本在 Blender 中生成所有视角：

```python
import bpy
import math
import os
import numpy as np

# 配置
output_dir = "/path/to/graf_ccsr/data/RS307_n/"
os.makedirs(output_dir, exist_ok=True)

# 参数（对应 GRAF config）
radius = 2.5
fov = 20  # 度
resolution = 256

# 设置渲染
scene = bpy.context.scene
scene.render.resolution_x = resolution
scene.render.resolution_y = resolution
scene.render.resolution_percentage = 100

# 设置相机
camera = bpy.data.objects['Camera']
camera.data.type = 'PERSP'
camera.data.lens_unit = 'FOV'
camera.data.angle = math.radians(fov)

# v 值和对应的 category 前缀
v_values = [
    (0.5, "0"),
    (0.4, "0.5"),
    (0.3, "1"),
    (0.2, "1.5")
]

def to_sphere(u, v):
    """转换为球坐标"""
    theta = 2 * math.pi * u
    phi = math.acos(1 - 2 * v)

    x = math.sin(phi) * math.cos(theta)
    y = math.sin(phi) * math.sin(theta)
    z = math.cos(phi)

    return np.array([x, y, z])

# 生成所有视角
for v_val, category_prefix in v_values:
    for angle_idx in range(360):
        u = angle_idx / 360.0

        # 计算相机位置
        pos = to_sphere(u, v_val) * radius

        # 设置相机位置（Blender 是 Z-up）
        camera.location = (pos[0], pos[1], pos[2])

        # 让相机看向原点
        direction = -pos
        rot_quat = direction.to_track_quat('-Z', 'Y')
        camera.rotation_euler = rot_quat.to_euler()

        # 渲染
        filename = f"{category_prefix}_{angle_idx}.png"
        scene.render.filepath = os.path.join(output_dir, filename)
        bpy.ops.render.render(write_still=True)

        print(f"Rendered: {filename}")

print("完成！")
```

### 5. 数据验证

生成数据后，运行验证脚本：
```bash
python check_blender_data.py
```

## 常见问题

### Q1: 我的 Blender 数据已经生成，但格式不同怎么办？
A: 创建一个转换脚本，将你的数据重命名为上述格式。

### Q2: 我的相机参数和 config 不匹配怎么办？
A: 运行 `python check_camera_params.py` 检查参数，然后：
- 要么修改 Blender 设置重新渲染
- 要么修改 `configs/default.yaml` 中的参数匹配你的 Blender 设置

### Q3: 需要多少张图像？
A: 根据当前配置，理论上需要：
- 4 个俯仰角 × 360 个方位角 = **1440 张图像**
- 但可以先用较少的角度测试（如每 10° 一张，共 144 张）

### Q4: 如何检查数据是否正确加载？
A: 运行：
```bash
python check_data.py --config configs/default.yaml
```
这会检查数据加载、统计信息，并保存一张样本图像。

## 🎯 下一步行动

1. **创建数据目录**：`mkdir -p data/RS307_n/`
2. **从 Blender 导出图像**（使用上述脚本或手动）
3. **验证数据格式**：`python check_blender_data.py`
4. **测试数据加载**：`python check_data.py`
5. **开始训练**

## 参考

- 数据集代码：`graf/datasets.py` (line 81-83)
- 配置文件：`configs/default.yaml`
- 球坐标转换：`graf/utils.py` 中的 `to_sphere` 函数
