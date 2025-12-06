"""
检查 Blender 数据的相机参数
"""
import numpy as np
import json
import os

def check_camera_params(data_dir='data/RS307_n'):
    print("="*60)
    print("相机参数检查")
    print("="*60)
    
    # 检查数据目录
    if not os.path.exists(data_dir):
        print(f"✗ 数据目录不存在: {data_dir}")
        return
    
    print(f"\n数据目录: {data_dir}")
    
    # 列出文件
    files = os.listdir(data_dir)
    print(f"文件数量: {len(files)}")
    print(f"文件类型: {set([f.split('.')[-1] for f in files if '.' in f])}")
    
    # 查找可能的相机参数文件
    camera_files = [f for f in files if 'camera' in f.lower() or 'pose' in f.lower() or 'transform' in f.lower()]
    print(f"\n相机相关文件: {camera_files}")
    
    # 查找JSON文件
    json_files = [f for f in files if f.endswith('.json')]
    print(f"JSON文件: {json_files}")
    
    # 读取并检查JSON文件
    for json_file in json_files[:3]:  # 只检查前3个
        json_path = os.path.join(data_dir, json_file)
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            print(f"\n{json_file} 内容:")
            print(f"  Keys: {list(data.keys())}")
            
            # 检查常见的相机参数
            if 'camera_angle_x' in data:
                print(f"  FOV (x): {data['camera_angle_x']} rad = {np.degrees(data['camera_angle_x']):.2f} deg")
            if 'camera_angle_y' in data:
                print(f"  FOV (y): {data['camera_angle_y']} rad = {np.degrees(data['camera_angle_y']):.2f} deg")
            if 'fl_x' in data:
                print(f"  Focal length x: {data['fl_x']}")
            if 'fl_y' in data:
                print(f"  Focal length y: {data['fl_y']}")
                
        except Exception as e:
            print(f"  ✗ 无法读取: {e}")
    
    # 检查图像文件
    img_files = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"\n图像文件数量: {len(img_files)}")
    
    if img_files:
        from PIL import Image
        sample_img = Image.open(os.path.join(data_dir, img_files[0]))
        print(f"图像尺寸: {sample_img.size}")
        print(f"图像模式: {sample_img.mode}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    check_camera_params()
