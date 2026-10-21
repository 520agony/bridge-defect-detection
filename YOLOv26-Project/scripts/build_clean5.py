"""wash -> clean：只修错、不删小，test原样保留可比性。
规则(train/val)：
 1. 越界坐标 clip到[0,1]（conc_导出舍入误差，ultralytics整图丢弃，修好=白捡312图）
 2. crack顶点<8 -> 删polygon（粗糙残留421个）
 3. 4点轴对齐矩形 -> 删polygon（假mask残留4个）
 4. 退化(去重后<3点/面积<0.0002) -> 删polygon
 5. tiny(<0.1%)全部保留（rebar本体，test也有，删了必掉分）
 6. 变空的图整图删（避免无标注正样本变负样本毒害）
test：逐字节原样复制。
"""
import shutil
from pathlib import Path
from collections import Counter

SRC = Path('datasets/final_5class_wash')
DST = Path('datasets/final_5class_clean')
NAMES = {0:'crack',1:'spalling',2:'hole',3:'rebar_exposed',4:'efflorescence'}

def is_fake_rect(v):
    if len(v)!=8: return False
    xs,ys=v[0::2],v[1::2]
    return xs[0]==xs[3] and xs[1]==xs[2] and ys[0]==ys[1] and ys[2]==ys[3]

def clean_polygon(cid, vals, stat):
    xs=[min(1.0,max(0.0,x)) for x in vals[0::2]]
    ys=[min(1.0,max(0.0,y)) for y in vals[1::2]]
    if vals[0::2]!=xs or vals[1::2]!=ys: stat['clipped']+=1
    if len(set(zip(xs,ys)))<3:
        stat['degenerate']+=1; return None
    area=(max(xs)-min(xs))*(max(ys)-min(ys))
    if area<=1e-9:
        stat['degenerate']+=1; return None
    if cid==0 and len(xs)<8:
        stat['crude_crack']+=1; return None
    if is_fake_rect(vals):
        stat['fake_rect']+=1; return None
    out=[str(cid)]+[f'{x:.6f}' for p in zip(xs,ys) for x in p]
    return ' '.join(out)

def main():
    assert SRC.exists(), 'wash源不在'
    if DST.exists(): raise SystemExit(f'{DST} 已存在，先删再跑')
    rep=Counter(); kept_img=Counter(); drop_img=Counter()
    for split in ['train','val','test']:
        (DST/'images'/split).mkdir(parents=True)
        (DST/'labels'/split).mkdir(parents=True)
        for img in (SRC/'images'/split).iterdir():
            if img.suffix.lower() not in ('.jpg','.jpeg','.png','.bmp'): continue
            lab=SRC/'labels'/split/(img.stem+'.txt')
            if split=='test':  # 原样
                shutil.copy(img, DST/'images'/split/img.name)
                if lab.exists(): shutil.copy(lab, DST/'labels'/split/lab.name)
                kept_img[split]+=1; continue
            stat=Counter()
            raw=lab.read_text(encoding='utf-8',errors='ignore').strip() if lab.exists() else ''
            lines=raw.splitlines() if raw else []
            new=[]
            for line in lines:
                p=line.split()
                if len(p)<7: stat['short']+=1; continue
                try: cid=int(float(p[0])); vals=[float(x) for x in p[1:]]
                except: stat['badnum']+=1; continue
                if cid not in NAMES: stat['badcid']+=1; continue
                r=clean_polygon(cid, vals, stat)
                if r: new.append(r)
            for k,v in stat.items(): rep[(split,k)]+=v
            if not new:
                if not raw:
                    # 原生空标签=背景负样本，保留
                    shutil.copy(img, DST/'images'/split/img.name)
                    (DST/'labels'/split/(img.stem+'.txt')).write_text('',encoding='utf-8')
                    kept_img[split]+=1; rep[(split,'keep_negative')]+=1; continue
                drop_img[split]+=1; continue  # 被洗空的整图删
            shutil.copy(img, DST/'images'/split/img.name)
            (DST/'labels'/split/(img.stem+'.txt')).write_text('\n'.join(new)+'\n',encoding='utf-8')
            kept_img[split]+=1
            for line in new: rep[(split,'keep_'+NAMES[int(float(line.split()[0]))])]+=1
    print('保留图:',dict(kept_img),'删空图:',dict(drop_img))
    print('---动作---')
    for k in sorted(rep): print(k,rep[k])

if __name__=='__main__': main()
