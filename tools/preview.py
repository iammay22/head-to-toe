#!/usr/bin/env python3
"""用 assets 裡的正式圖疊出組合圖，檢查對齊。輸出 preview.png"""
import json, os, numpy as np
from PIL import Image
R=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); A=os.path.join(R,'assets')
M=json.load(open(os.path.join(A,'manifest.json'))); meta=M['meta']; items=M['items']
SK=np.array(meta['skin'])
def layer(i,size):
    it=items[i]; c=Image.new('RGBA',size,(0,0,0,0)); c.paste(Image.open(os.path.join(A,i+'.webp')).convert('RGBA'),(it['x'],it['y'])); return c
def recolor(im,target):
    a=np.asarray(im).astype(np.float32); rgb=a[...,:3]; l=rgb.mean(-1)
    warm=(rgb[...,0]>rgb[...,1])&(rgb[...,1]>rgb[...,2])&(rgb[...,0]-rgb[...,2]>25)
    w=np.clip((l-95)/55,0,1)*warm
    out=rgb*(1-w[...,None])+np.clip(rgb*(np.array(target)/SK),0,255)*w[...,None]
    return Image.fromarray(np.dstack([out,a[...,3]]).astype(np.uint8),'RGBA')
def head(face,parts,tone):
    h=recolor(layer(face,(1024,1024)),tone)
    for p in parts: h.alpha_composite(layer(p,(1024,1024)))
    return h
def figure(face,parts,tone):
    b=meta['body']; an=b['anchor']; hd=meta['head']
    c=Image.new('RGBA',(b['w'],b['h']),(0,0,0,0)); c.alpha_composite(recolor(layer('body_base',(b['w'],b['h'])),tone))
    h=head(face,parts,tone); k=an['k']; s=round(1024*k); hs=h.resize((s,s),Image.LANCZOS)
    c.alpha_composite(hs,(round(an['cx']-hd['cx']*k),round(an['top']-hd['top']*k))); return c,h
combos=[('face_oval_base',['brows_flat','eyes_peach'],meta['skin']),('face_square_base',['brows_sword','eyes_phoenix'],(236,188,150)),
        ('face_oval_round',['brows_flat','eyes_phoenix'],(198,140,100)),('face_oval_pointed',['brows_sword','eyes_peach'],(140,92,62))]
W=420; out=Image.new('RGB',(W*4,1150),(253,244,240))
for i,(f,p,t) in enumerate(combos):
    fig,h=figure(f,p,t)
    fig=fig.crop((240,120,840,1880)).resize((W-140,int((W-140)*1760/600)),Image.LANCZOS)
    out.paste(fig,(i*W+70,330),fig)
    hh=h.crop((200,20,824,840)).resize((236,310),Image.LANCZOS); out.paste(hh,(i*W+92,10),hh)
out.save(os.path.join(R,'preview.png')); print('ok')
