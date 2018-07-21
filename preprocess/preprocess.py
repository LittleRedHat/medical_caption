import os
import requests
from bs4 import BeautifulSoup
import json
import traceback
from multiprocessing import Manager, Pool
import multiprocessing

import xml.etree.ElementTree as ET
import os
import re
import pandas as pd

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


def extract_meta(filepath):
    tree = ET.ElementTree(file=filepath)

    findings = ''
    impression = ''
    for elem in tree.iter('AbstractText'):
        
        if elem.attrib.get('Label') == 'FINDINGS':
            findings = elem.text or ''

        if elem.attrib.get('Label') == 'IMPRESSION':
            impression = elem.text or ''

    report = impression + ' ' + findings

    ids = []
    for elem in tree.iter('parentImage'):
        ids.append(elem.attrib.get('id'))
    
    tags = []
    for elem in tree.iter('automatic'):
        tags.append(elem.text)

    if not len(tags):
        tags.append('normal')
    
    if report.strip() == '' or ids == []:
        return None

    else:
        return {'ids':ids,'tags':tags,'report':report}
        



def extract(root,output):
    results = []
    none_cnt = 0

    for name in os.listdir(root):
        path = os.path.join(root, name)
        meta = extract_meta(path)

        if meta == None:
            none_cnt += 1
            continue
        results.append(meta)

    with open(os.path.join(output,'findings.json'),'w') as f:
        json.dump(results,f)

    print(none_cnt)

def word_frequency(findings_file):
    with open(findings_file,'r') as f:
        findings = json.load(f)
    
    word2count = {}
    
    for item in findings:
        caption = item['report'].strip().lower()
        caption = normalize_string(caption)
        caption = [sent.strip() for sent in caption.split(' .') if len(sent.strip()) > 0]
        for sent in caption:
            for word in sent.split():
                
                if not re.match(r'^[a-zA-Z]+$',word):
                    continue

                word2count[word] = word2count.get(word,0) + 1

    total = sum(word2count.values())


    word_frequency = [{'word':word,'count':count, 'frequency':count * 1.0 / total} for word,count in word2count.items()]
    word_frequency.sort(key=lambda x:x['count'],reverse=True)

    frequency_sum = 0.0
    for idx,item in enumerate(word_frequency):
        frequency_sum = frequency_sum + item['frequency']
        if frequency_sum > 0.99:
            print('top {} words covers 99% occurrences'.format(idx + 1))
            break
        
    word_frequency = pd.DataFrame(word_frequency)
    word_frequency.to_csv('../output/preprocess/IU_Chest_XRay/words.csv',index=False)

    # with open('../output/preprocess/IU_Chest_XRay/words.txt','w') as f:
    #     for word,count,frequency in word_frequency:
    #         f.write('{} {} {:.8f}\n'.format(word,count,frequency))



def tag_frequency(findings_file):
    with open(findings_file,'r') as f:
        findings = json.load(f)
    
    tag2count = {}
    
    for item in findings:
        tags = item['tags']
        for tag in tags:
            tag = tag.strip().lower()
            tag2count[tag] = tag2count.get(tag,0) + 1

    total = sum(tag2count.values())


    tag_frequency = [{'tag':tag, 'count':count, 'frequency':count * 1.0 / total} for tag,count in tag2count.items()]
    tag_frequency.sort(key=lambda x:x['count'],reverse=True)

    frequency_sum = 0.0
    for idx,item in enumerate(tag_frequency):
        frequency_sum = frequency_sum + item['frequency']
        if frequency_sum > 0.99:
            print('top {} tags covers 99% occurrences'.format(idx + 1))
            break
    
    tag_frequency = pd.DataFrame(tag_frequency)
    tag_frequency.to_csv('../output/preprocess/IU_Chest_XRay/tags.csv',index=False)


    # with open('../output/preprocess/IU_Chest_XRay/tags.txt','w') as f:
    #     for tag,count,frequency in tag_frequency:
    #         f.write('{} {} {:.8f}\n'.format(tag,count,frequency))

def train_val_test_split(findings_file,output,test_samples = 500, val_samples = 500):
    with open(findings_file,'r') as f:
        findings = json.load(f)

    total_findings = []
    for item in findings:
        ids = item['ids']
        for id in ids:
            total_findings.append({'id':id,'report':item['report'],'tags':item['tags']})

    import random
    random.shuffle(total_findings)

    test_findings = total_findings[:test_samples]
    val_findings = total_findings[test_samples:(test_samples+val_samples)]
    train_findings = total_findings[(test_samples+val_samples):]

    with open(os.path.join(output,'test_findings.json'),'w') as f:
        json.dump(test_findings,f)
    with open(os.path.join(output,'val_findings.json'),'w') as f:
        json.dump(val_findings,f)
    with open(os.path.join(output,'train_findings.json'),'w') as f:
        json.dump(train_findings,f)



def stat_lang(findings_file):
    with open(findings_file,'r') as f:
        findings = json.load(f)
    
    sent_max_num = 0
    word_max_num = 0
    sent_max_report = None
    word_max_report = None

    for item in findings:
        report = item['report']
        caption = normalize_string(report)
        caption = [sent.strip() for sent in caption.split(' .') if len(sent.strip()) > 0]
        if sent_max_num < len(caption):
            sent_max_num = len(caption)
            sent_max_report = caption

        for sent in caption:
            words = sent.split()
            if word_max_num < len(words):
                word_max_num = len(words)
                word_max_report = sent
    print('max sentence number is {} and max words num is {}'.format(sent_max_num,word_max_num))
    print(sent_max_report)
    print(word_max_report)

                
                
            
        

    
    

             

    
    
    
    


if __name__ == '__main__':
    # extract('../data/IU_Chest_XRay/ecgen-radiology','../output/preprocess/IU_Chest_XRay/')
    word_frequency('../output/preprocess/IU_Chest_XRay/findings.json')
    tag_frequency('../output/preprocess/IU_Chest_XRay/findings.json')
    train_val_test_split('../output/preprocess/IU_Chest_XRay/findings.json','../output/preprocess/IU_Chest_XRay/')
    stat_lang('../output/preprocess/IU_Chest_XRay/findings.json')

                

                
            
            
        
