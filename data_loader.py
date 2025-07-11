import os
import pandas as pd
from pathlib import Path

class MRIDataLoader:
    def __init__(self, data_dir, labels_csv):
        """
        初始化数据加载器。
        
        Args:
            data_dir (str): datas_mri 目录路径
            labels_csv (str): labels.csv 文件路径
        """
        self.data_dir = Path(data_dir)
        self.labels_df = pd.read_csv(labels_csv, encoding='latin1')
        
    def load_data(self):
        """
        加载图像路径和标签，跳过图像不全的样本，并统计样本数量。
        
        Returns:
            tuple: (data, stats)
                   data: 样本列表，每个元素为 (sample_id, image_paths, label)
                   stats: 字典，包含总样本数、正常样本数和患病样本数
        """
        data = []
        total_samples = 0
        normal_samples = 0
        disease_samples = 0
        
        # 遍历 data_dir 中的所有样本文件夹
        for sample_folder in self.data_dir.iterdir():
            if not sample_folder.is_dir():
                continue
                
            sample_id = sample_folder.name  # e.g., BNI_29006_1
            # 从 sample_id 中提取 SUB_ID
            sub_id = sample_id.split('_')[1]  # e.g., 29006
            site_id = f"ABIDEII-{sample_id.split('_')[0]}_1"  # e.g., ABIDEII-BNI_1
            
            # 检查三张图像是否都存在
            anat_path = sample_folder / 'anat_1' / 'SNAPSHOTS' / 'qc_t.gif'
            dti_path = sample_folder / 'dti_1' / 'SNAPSHOTS' / 'qc_t.gif'
            rest_path = sample_folder / 'rest_1' / 'SNAPSHOTS' / 'qc_t.gif'
            
            # 如果有任意图像缺失，跳过该样本
            if not (anat_path.exists() and dti_path.exists() and rest_path.exists()):
                # print(f"跳过 {sample_id}：图像不完整")
                continue
                
            # 在 CSV 中查找对应的标签
            label_row = self.labels_df[
                (self.labels_df['SITE_ID'] == site_id) & 
                (self.labels_df['SUB_ID'] == int(sub_id))
            ]
            
            if label_row.empty:
                # print(f"跳过 {sample_id}：CSV 中未找到标签")
                continue
                
            label = label_row['DX_GROUP'].iloc[0]
            
            # 存储图像路径和标签
            image_paths = {
                'anat': str(anat_path),
                'dti': str(dti_path),
                'rest': str(rest_path)
            }
            
            data.append((sample_id, image_paths, label))
            
            # 更新统计信息
            total_samples += 1
            if label == 1:
                disease_samples += 1
            elif label == 2:
                normal_samples += 1
                
        stats = {
            'total_samples': total_samples,
            'normal_samples': normal_samples,
            'disease_samples': disease_samples
        }
        
        return data, stats

# 示例用法：
if __name__ == "__main__":
    data_loader = MRIDataLoader(
        data_dir="datas_mri",
        labels_csv="lables.csv"
    )
    dataset, stats = data_loader.load_data()
    
    # 输出统计信息
    print(f"可用样本总数量: {stats['total_samples']}")
    print(f"正常样本数量 (DX_GROUP=2): {stats['normal_samples']}")
    print(f"患病样本数量 (DX_GROUP=1): {stats['disease_samples']}")