"""Retrieve official Canadian settlement names; preserve administrative reserves separately."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import requests,json,hashlib,time

ROOT=Path(__file__).resolve().parent;RAW=ROOT/'原始数据';RAW.mkdir(exist_ok=True)
URL='https://geogratis.gc.ca/services/geoname/en/geonames.json'
CODES=['CITY','TOWN','VILG','HAM','UNP','IR']
def fetch_code(code):
    records=[];sources=[];skip=0
    while True:
        params={'concise':code,'bbox':'-142,50,-50,90','num':1000,'category':'O','sort-field':'name','skip':skip}
        filename=RAW/f'canada_{code}_O_{skip:04}.json'
        if filename.exists():d=json.loads(filename.read_text(encoding='utf-8'))
        else:
            for attempt in range(3):
                try:
                    r=requests.get(URL,params=params,timeout=(15,45));r.raise_for_status();d=r.json()
                    if 'items' not in d:raise ValueError('missing items')
                    filename.write_text(json.dumps(d,ensure_ascii=False),encoding='utf-8');break
                except Exception:
                    if attempt==2:raise
                    time.sleep(2)
        items=d.get('items',[])
        if any(r.get('category')!='O' or r.get('concise',{}).get('code')!=code for r in items):raise ValueError('provider returned out-of-scope category')
        sources.append({'file':filename.name,'url':URL,'parameters':params,'records':len(items),'sha256':hashlib.sha256(filename.read_bytes()).hexdigest()})
        records.extend(items)
        if len(items)<1000:break
        skip+=1000
    return code,records,sources
def main():
    all_records=[];sources=[];counts={};failures={}
    with ThreadPoolExecutor(3) as pool:
        futures={pool.submit(fetch_code,c):c for c in CODES}
        for f in as_completed(futures):
            try:
                c,r,s=f.result();all_records.extend(r);sources.extend(s);counts[c]=len(r);print(c,len(r),flush=True)
            except Exception as e:failures[futures[f]]=str(e);print('FAILED',futures[f],str(e),flush=True)
    (RAW/'canada_official_settlement_records.json').write_text(json.dumps(all_records,ensure_ascii=False),encoding='utf-8')
    (RAW/'canada_official_download_record.json').write_text(json.dumps({'source':'CGNDB_Natural_Resources_Canada','scope':'official names category O; 50N bounding box followed by local AMAP polygon clip','codes':CODES,'counts':counts,'failures':failures,'pages':sources,'note':'IR are reserve administrative areas, kept separately from settlement-place records; a name being official does not prove current residence or historical continuity.'},ensure_ascii=False,indent=2),encoding='utf-8')
    if failures:raise RuntimeError('Some Canadian source classes were not retrieved')
if __name__=='__main__':main()
