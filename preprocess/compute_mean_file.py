from PIL import Image
import os
import numpy as np

def compute_mean_file(image_dir):
    means = []


    image_names = os.listdir(image_dir)
    for image_name in image_names:
        if os.path.splitext(image_name)[1] not in ['.jpg','.png']:
            continue
        
        image_path = os.path.join(image_dir,image_name)
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = np.array(image).astype(float)
        image = image / 255.0
        mean = image.mean(dim=2)

        means.append(mean)

    means = np.array(means)

    mean = np.mean(means)
    std = np.std(manes)

    print(mean,std)




if __name__ == '__main__':
    image_path = '../../../../Desktop/CXR10_IM-0002-1001.png'
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = np.array(image).astype(float)
    mean = image.mean(axis=2)
    print(mean)

    

    
    







