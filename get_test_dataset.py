import pandas as pd
import numpy as np
import pickle

path_to_excel = r"OC171D/X-ray scoring NZKS OT high fish oil trial Score Data OC171D.xlsx"
dfb = pd.read_excel(path_to_excel, skiprows=(1,2,3),engine='openpyxl')

columns = ['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 2','LKS Region 1 X-ray', 'LKS Region 2 X-ray', 'LKS Region 3 X-ray', 'LKS Region 4 X-ray',
            'Fusion Region 1 X-ray', 'Fusion Region 2 X-ray', 'Fusion Region 3 X-ray', 'Fusion Region 4 X-ray',
            'Compression and/or reduced intervertebral space Region 1 X-ray',
            'Compression and/or reduced intervertebral space Region 2 X-ray',
            'Compression and/or reduced intervertebral space Region 3 X-ray',
            'Compression and/or reduced intervertebral space Region 4 X-ray',
            'Vertical shift Region 1 X-ray', 
            'Vertical shift Region 2 X-ray', 
            'Vertical shift Region 3 X-ray', 
            'Vertical shift Region 4 X-ray'
            ]
data = dfb[columns]
gt =[0,0,0,0]
for i in range(0,300):
  file_name = data["Unnamed: 0"][i].split("/")[1]+".jpg"
  label =[]
  for key in columns[3:7]:
    gt[data[key][i]] += 1
print(gt)
print(sum(gt))


