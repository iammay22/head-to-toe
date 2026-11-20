import asyncio, sys, json
from playwright.async_api import async_playwright
import os, pathlib
HERE = pathlib.Path(__file__).resolve().parent
INDEX = (HERE.parent / 'index.html').as_uri()
os.makedirs(HERE / 'shots', exist_ok=True); os.chdir(HERE)
from PIL import Image
out, combos = sys.argv[1], json.loads(sys.argv[2]); cols = int(sys.argv[3]) if len(sys.argv) > 3 else 4
crop = json.loads(sys.argv[4]) if len(sys.argv) > 4 else None
async def main():
    async with async_playwright() as p:
        b=await p.chromium.launch(executable_path='/usr/bin/google-chrome',args=['--no-sandbox'])
        pg=await b.new_page(viewport={'width':375,'height':740},device_scale_factor=2); logs=[]
        pg.on('pageerror',lambda e: logs.append('PAGEERROR '+str(e)))
        pg.on('console',lambda m: logs.append(m.text) if m.type=='error' else None)
        await pg.goto(''+INDEX+''); await pg.wait_for_function("!document.getElementById('startBtn').disabled")
        await pg.click('#startBtn'); await pg.wait_for_timeout(300)
        ims=[]
        for i,c in enumerate(combos):
            v=c.pop('_view','head')
            await pg.evaluate("([c,v])=>{ch=Object.assign(defaultChar(),{hair:0},c);view=v;camNow=null;redraw()}",[c,v]); await pg.wait_for_timeout(150)
            await (await pg.query_selector('#view')).screenshot(path=f'shots/_t.png')
            im=Image.open('shots/_t.png').convert('RGB')
            if crop: w,h=im.size; im=im.crop((int(w*crop[0]),int(h*crop[1]),int(w*crop[2]),int(h*crop[3])))
            ims.append(im)
        w,h=ims[0].size; rows=(len(ims)+cols-1)//cols
        sheet=Image.new('RGB',(w*cols,h*rows),'white')
        for i,im in enumerate(ims): sheet.paste(im,((i%cols)*w,(i//cols)*h))
        sheet.save(f'shots/{out}.png'); print(logs or 'no errors', sheet.size); await b.close()
asyncio.run(main())
