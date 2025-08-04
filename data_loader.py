import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class MRIDataset(Dataset):
    def __init__(self, data_dir, csv_file, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        self.labels_df = pd.read_csv(csv_file, encoding='latin1')
        
        self.sample_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
        
        self.valid_samples = []
        self.labels = []
        
        for folder in self.sample_folders:
            sub_id = folder.split('_')[1]
            
            anat_path = os.path.join(data_dir, folder, 'rest_1', 'SNAPSHOTS', 'qc_t.gif')
            
            if os.path.exists(anat_path):
                site_id = f"ABIDEII-{folder.split('_')[0]}_1"
                row = self.labels_df[(self.labels_df['SITE_ID'] == site_id) & 
                                   (self.labels_df['SUB_ID'] == int(sub_id))]
                
                if not row.empty:
                    dx_group = row['DX_GROUP'].iloc[0]
                    # Convert DX_GROUP to binary (1: diseased -> 1, 2: normal -> 0)
                    label = 1 if dx_group == 1 else 0
                    self.valid_samples.append((anat_path, folder))
                    self.labels.append(label)
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        img_path, sample_id = self.valid_samples[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, sample_id

def main():
    data_dir = "datas_mri"
    csv_file = "labels.csv"
    
    dataset = MRIDataset(data_dir, csv_file)
    
    print(f"Number of valid samples: {len(dataset)}")

if __name__ == "__main__":
    main()
