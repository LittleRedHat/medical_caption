import os
import requests
from bs4 import BeautifulSoup
import json
import traceback
from multiprocessing import Manager, Pool
import multiprocessing

import sys
sys.path.append('..')
from utils import unicode_to_ascii, normalize_string




def parse(http_url,results,img_names):
    for img_name in img_names:
        try:
            print('parse {}'.format(img_name))
            name,ext = os.path.splitext(img_name)
            if ext in ['.png','.PNG','jpg']:
                response = requests.get(http_url.format(name))
                text = json.loads(response.text)

                if 'error' in text:
                    continue

                text = text.get('list') or []
                if not len(text):
                    continue
                else:
                    text = text[0]
                obj = {
                    'problems':text.get('Problems'),
                    'mesh':text.get('MeSH')
                    
                }

                abstract = text.get('abstract')
                abstract = abstract.replace('\/','/')
                soup = BeautifulSoup(abstract,'html.parser')
                
                paragraphs = soup.find_all('p')

                for p in paragraphs:
                    inner_text = p.text.strip()

                    if inner_text.startswith('Comparison'):
                        comparison = p.contents[1]
                        obj['comparison'] = comparison
                    elif inner_text.startswith('Indication'):
                        indication = p.contents[1]
                        obj['indication'] = indication

                    elif inner_text.startswith('Findings'):
                        findings = p.contents[1]
                        obj['findings'] = findings
                    elif inner_text.startswith('Impression'):
                        impression = p.contents[1]
                        obj['impression'] = impression

                results[img_name] = obj
        except:
            traceback.print_exc()


class ChestXRayParser():
    def __init__(self,image_dir):
        super(ChestXRayParser,self).__init__()
        self.image_dir = image_dir
        self.http_url = 'http://openi.nlm.nih.gov/retrieve.php?img={}&query=&coll=cxr&req=4'
    
    def multiproecess_parse(self):
        
        
        img_names = os.listdir(self.image_dir)

        results = Manager().dict()
        pool = Pool(os.cpu_count())

        total_imags = len(img_names)
        bucket_size = total_imags // os.cpu_count() + 1

        for i in range(os.cpu_count()):
            pool.apply_async(parse,args = (self.http_url,results,img_names[i*bucket_size:(i+1)*bucket_size]))

        pool.close()
        pool.join()

        print(results)

        with open('findings.json','w') as f:
            json.dump(results.copy(),f)   

        


def download_chest_xray_meta():
    parser = ChestXRayParser('../dataset/IU_Chest_XRay/NLMCXR_png')
    parser.multiproecess_parse()

def get_mesh_frequency():
    with open('../data/IU_Chest_XRay/findings.json') as f:
        records = json.load(f)

    tag2count = {}
    for record in records.values():
        mesh = record.get('mesh',{})
        tags = mesh.get('major',[])
        for tag in tags:
            tag = tag.strip().lower().replace('/','_').replace(', ','_').replace(' ','_')
            tag = unicode_to_ascii(tag)
            tag2count[tag] = tag2count.get(tag,0) + 1
    total = sum(tag2count.values())
    tag_frequency = [(tag, count, count * 1.0 / total) for tag,count in tag2count.items()]
    tag_frequency.sort(key=lambda x:x[1],reverse=True)

    frequency_sum = 0.0
    for idx,item in enumerate(tag_frequency):
        frequency_sum = frequency_sum + item[2]
        if frequency_sum > 0.99:
            print('top {} tags covers 99% occurrences'.format(idx + 1))
            break
    
    with open('../output/preprocess/IU_Chest_XRay/tags.txt','w') as f:
        for tag,count,frequency in tag_frequency:
            f.write('{} {} {:.8f}\n'.format(tag,count,frequency))


def train_val_test_split():
    pass


def get_word_frequency():
    with open('../data/IU_Chest_XRay/findings.json') as f:
        records = json.load(f)

    word2count = {}
    for record in records.values():
        impression = record.get('impression','')
        findings = record.get('findings','')
        final_report = impression + ' ' + findings
        final_report = normalize_string(final_report)
        captions = final_report.split(' .')
        
        for caption in captions:
            caption = caption.strip()
            for word in caption.split():
                word2count[word] = word2count.get(word,0) + 1
    total = sum(word2count.values())
    word_frequency = [(word,count,count * 1.0 / total) for word,count in word2count.items()]
    word_frequency.sort(key=lambda x: x[1], reverse=True)

    frequency_sum = 0.0
    for idx,item in enumerate(word_frequency):
        frequency_sum = frequency_sum + item[2]
        if frequency_sum > 0.99:
            print('top {} words covers 99% occurrences'.format(idx + 1))
            break
            
            
        

    with open('../output/preprocess/IU_Chest_XRay/words.txt','w') as f:
        for word,count,frequency in word_frequency:
            f.write('{} {} {:.8f}\n'.format(word,count,frequency))

        




    





if __name__ == '__main__':
    # download_chest_xray_meta()
    get_word_frequency()
    get_mesh_frequency()
    
                

                
            
            
        
