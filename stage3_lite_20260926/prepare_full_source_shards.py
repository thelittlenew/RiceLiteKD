"""Immutable full-source shards, grouped split and exact RGB overlap audit."""
import collections,hashlib,io,json,os,re,tarfile,time,traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from prepare_source_pilot import index,group

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'data/Pseudo_label_training_data'
OUT=ROOT/'source_full_v1'
REMOTE='/root/stage3_full_source_20260926'
def sha(x):return hashlib.sha256(x).hexdigest()
def save(name,obj):
    p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');t.replace(p)
def process(item):
    stem,ip,mp=item
    image=ip.read_bytes();mask=mp.read_bytes()
    with Image.open(io.BytesIO(image)) as im:
        rgb=im.convert('RGB');assert rgb.size==(512,512),(stem,rgb.size)
        pixelsha=sha(rgb.tobytes())
    with Image.open(io.BytesIO(mask)) as im:
        assert im.size==(512,512)
        values=np.array(im)
    if values.ndim==3: values=np.any(values!=0,axis=2).astype(np.uint8)
    else:
        assert np.isin(values,[0,1,255]).all(),stem
        values=(values!=0).astype(np.uint8)
    buf=io.BytesIO();Image.fromarray(values).save(buf,format='PNG',compress_level=6)
    return stem,image,buf.getvalue(),pixelsha,sha(image),sha(mask),int(values.sum())

def process_safe(item):
    try:return process(item)
    except (OSError,ValueError,AssertionError) as error:
        stem,ip,mp=item
        return {'excluded_stem':stem,'image_path':str(ip),'mask_path':str(mp),
                'image_sha256':sha(ip.read_bytes()),'mask_sha256':sha(mp.read_bytes()),
                'reason':type(error).__name__+': '+str(error)}

def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'PREPARED.json').exists():print('Already prepared');return
    with (OUT/'prepare.lock').open('x') as lock:lock.write(str(os.getpid()))
    started=time.time()
    images=index(DATA/'Panicle_data_2025_early',{'.jpg','.jpeg','.png','.bmp','.tif','.tiff'})
    masks=index(DATA/'mask',{'.png','.bmp','.tif','.tiff'})
    paired=sorted(images.keys()&masks.keys());assert len(paired)==103817
    pilot=json.loads((ROOT/'source_pilot/selection.json').read_text())
    original_val=set(pilot['validation_groups'])
    groups={group(s) for s in paired};assert len(groups)==1014
    parent={g:g for g in groups}
    def find(g):
        while parent[g]!=g:
            parent[g]=parent[parent[g]];g=parent[g]
        return g
    def join(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    blocked=set()
    for folder in [ROOT.parent/'data/RiceSEG',ROOT.parent/'_records/server_work/public_backup_20260916/cea_public_20260915/data/RiceSEG/images']:
        for ip in folder.rglob('*'):
            if ip.suffix.lower() in ('.jpg','.jpeg'):
                with Image.open(ip) as im:
                    rgb=im.convert('RGB');blocked.add((rgb.size,sha(rgb.tobytes())))
    save('status.json',{'state':'preparing','pid':os.getpid(),'total':len(paired),'processed':0,'started':started})
    first_pixels={};records=[];shards=[];duplicates=0;excluded=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for base in range(0,len(paired),512):
            stems=paired[base:base+512];number=base//512
            name=f'shard_{number:04d}.tar';temp=OUT/(name+'.tmp');path=OUT/name
            reuse=path.exists() and (OUT/(name+'.ready.json')).exists()
            if reuse:
                prior=json.loads((OUT/(name+'.ready.json')).read_text())
                assert sha(path.read_bytes())==prior['sha256']
            with tarfile.open(path if reuse else temp,'r' if reuse else 'w',format=tarfile.USTAR_FORMAT) as archive:
                for returned in pool.map(process_safe,[(s,images[s],masks[s]) for s in stems]):
                    if isinstance(returned,dict):
                        assert not reuse,'Previously packaged sample became invalid'
                        excluded.append(returned);save('excluded.json',excluded);continue
                    stem,image,mask,pixelsha,filesha,masksha,foreground=returned
                    assert ((512,512),pixelsha) not in blocked, 'Source/RiceSEG exact RGB overlap: '+stem
                    g=group(stem)
                    if pixelsha in first_pixels:
                        join(g,first_pixels[pixelsha]);duplicates+=1
                    else:first_pixels[pixelsha]=g
                    item={'stem':stem,'group':g,'archive':f'{REMOTE}/{name}',
                          'image_sha256':filesha,'normalized_mask_sha256':sha(mask),
                          'original_mask_sha256':masksha,'rgb_sha256':pixelsha,'foreground_pixels':foreground}
                    for kind,blob in [('image',image),('mask',mask)]:
                        member=f'{base+len(records)-base:06d}/{kind}'
                        info=tarfile.TarInfo(member);info.size=len(blob);info.mtime=0
                        item[kind+'_bytes']=len(blob)
                        if reuse:
                            old=archive.getmember(member);assert old.size==len(blob)
                            assert archive.extractfile(old).read()==blob
                            item[kind+'_offset']=old.offset_data
                        else:
                            item[kind+'_offset']=archive.offset+512
                            archive.addfile(info,io.BytesIO(blob))
                    records.append(item)
            if not reuse:temp.replace(path)
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            entry={'name':name,'bytes':path.stat().st_size,'sha256':digest}
            shards.append(entry);save(name+'.ready.json',entry)
            save('status.json',{'state':'preparing','pid':os.getpid(),'total':len(paired),'processed':len(records),
                                'ready_shards':len(shards),'excluded':len(excluded),'bytes':sum(s['bytes'] for s in shards),'started':started,'updated':time.time()})
            print(json.dumps({'prepared':len(records),'shards':len(shards),'bytes':entry['bytes']}),flush=True)
    # Preserve pilot validation groups; expand to connected groups if exact duplicate RGB bridges groups.
    val_components={find(g) for g in original_val}
    splits={'train':[],'val':[]}
    for item in records:
        split='val' if find(item['group']) in val_components else 'train'
        splits[split].append(item)
    assert not ({r['rgb_sha256'] for r in splits['train']}&{r['rgb_sha256'] for r in splits['val']})
    assert not ({r['group'] for r in splits['train']}&{r['group'] for r in splits['val']})
    assert 0<len(splits['val'])<.2*len(records)
    manifests={}
    for split,rows in splits.items():
        save(split+'.json',rows)
        manifests[split]={'name':split+'.json','count':len(rows),'sha256':sha((OUT/(split+'.json')).read_bytes())}
    assert len(records)+len(excluded)==len(paired)
    save('excluded.json',excluded)
    save('inventory.json',{'paired_total':len(paired),'paired_usable':len(records),'excluded':excluded,'groups':len(groups),'images_total':len(images),'masks_total':len(masks),
          'unmatched_images':len(images)-len(paired),'unmatched_masks':len(masks)-len(paired),
          'validation_group_rule':'pilot fixed original-photo validation groups, expanded only by exact RGB connected components',
          'validation_groups':sorted({r['group'] for r in splits['val']}),'exact_rgb_duplicates_within_source':duplicates,
          'rice_rgb_hashes_checked':len(blocked),'split_counts':{s:len(r) for s,r in splits.items()},
          'pixel_counts_train':{'foreground':sum(r['foreground_pixels'] for r in splits['train']),
              'background':len(splits['train'])*512*512-sum(r['foreground_pixels'] for r in splits['train'])},
          'limitations':'Exact RGB duplicate checks do not prove absence of all near-duplicates.',
          'mask_mapping':'Nonzero foreground mapped to1; background0. Original mask hash retained.'})
    info={'state':'prepared_full_source','paired_total':len(paired),'paired':len(records),'excluded_count':len(excluded),'shards':shards,'manifests':manifests,
          'remote_root':REMOTE,'bytes':sum(s['bytes'] for s in shards),'inventory_sha256':sha((OUT/'inventory.json').read_bytes()),
          'code_sha256':sha(Path(__file__).read_bytes()),'started':started,'finished':time.time()}
    save('PREPARED.json',info);save('status.json',{'state':'prepared_full_source','processed':len(records),'updated':time.time()})
    print(json.dumps({'state':'prepared_full_source','counts':{s:len(r) for s,r in splits.items()},'bytes':info['bytes']}),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        save('FAILED.json',{'time':time.time(),'traceback':traceback.format_exc()});raise
