import pandas as pd
import numpy as np
import cv2    
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML
from PIL import Image
from io import BytesIO
import shutil

celeb_path = 'beard_data\\list_attr_celeba.csv'
celeb_data = pd.read_csv(celeb_path) 
# celeb_data.set_index('image_id', inplace=True)
# celeb_data.replace(to_replace=-1, value=0, inplace=True)
print(celeb_data.columns)

# Drop all females
indexNames = celeb_data[celeb_data['Male'] == 0 ].index 
celeb_data.drop(indexNames , inplace=True)
celeb_data.describe()

# celeb_features = ['Goatee', 'Mustache', 'No_Beard']
# celeb_data = celeb_data[celeb_features]
# celeb_data.head() 5_o_Clock_Shadow

a = celeb_data.loc[(celeb_data['No_Beard'] != 1) & (celeb_data['Male'] == 1)& (celeb_data['Mustache'] == 1),"image_id"].tolist()
celeb_data.head()
hair_p = "beard_data\\img_align_celeba\\img_align_celeba"
idex = 0
for hair in a:
    if idex == 5000:
        break
    path = hair_p + "\\" + hair
    new_path = "data_set\\hair"
    shutil.copy(path, new_path)
    idex = idex + 1

a = celeb_data.loc[(celeb_data['No_Beard'] == 1) & (celeb_data['Male'] == 1),"image_id"].tolist()
celeb_data.head()
hair_p = "beard_data\\img_align_celeba\\img_align_celeba"
idex = 0
for no_hair in a:
    if idex == 5000:
        break
    path = hair_p + "\\" + no_hair
    new_path = "data_set\\no_hair"
    shutil.copy(path, new_path)
    idex = idex + 1

print(a[0])