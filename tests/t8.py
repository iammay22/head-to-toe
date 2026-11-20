import asyncio, json
from playwright.async_api import async_playwright
import os, pathlib
HERE = pathlib.Path(__file__).resolve().parent
INDEX = (HERE.parent / 'index.html').as_uri()
os.makedirs(HERE / 'shots', exist_ok=True); os.chdir(HERE)
async def run(url,tag,b):
    pg=await b.new_page(viewport={'width':375,'height':740},device_scale_factor=2); logs=[]
    pg.on('pageerror',lambda e: logs.append('PAGEERROR '+str(e))); pg.on('console',lambda m: logs.append(m.text) if m.type=='error' else None)
    await pg.goto(url); await pg.wait_for_function("!document.getElementById('startBtn').disabled")
    await pg.click('#startBtn'); await pg.wait_for_timeout(500)
    t=await pg.evaluate("""()=>{const cv=document.getElementById('view');const r={};
      let t0=performance.now();for(let i=0;i<40;i++){paint(cv,ch,camFor(i%2?'head':'full',ch));cv.getContext('2d').getImageData(0,0,1,1)} r.camFrame=((performance.now()-t0)/40).toFixed(1);
      t0=performance.now();for(let i=0;i<20;i++){ch.upper=20+i*3;(typeof drawCached==='function'?0:prepare(ch));paint(cv,ch,camFor('head',ch));cv.getContext('2d').getImageData(0,0,1,1)} r.slider=((performance.now()-t0)/20).toFixed(1);
      ch.shoulder=1;t0=performance.now();for(let i=0;i<20;i++){ch.weight=20+i*3;(typeof drawCached==='function'?0:prepare(ch));paint(cv,ch,camFor('full',ch));cv.getContext('2d').getImageData(0,0,1,1)} r.bodySlider=((performance.now()-t0)/20).toFixed(1);
      return r}""")
    print(tag,t,logs or '')
    return pg
async def main():
    async with async_playwright() as p:
        b=await p.chromium.launch(executable_path='/usr/bin/google-chrome',args=['--no-sandbox'])
        await run('file:///home/claude/head-to-toe/index.html','舊版 ms/畫格',b)
        pg=await run(''+INDEX+'','新版 ms/畫格',b)
        logs=[]; pg.on('pageerror',lambda e: logs.append('PAGEERROR '+str(e)))
        await pg.evaluate("()=>{ch=defaultChar();tab='gender';view='full';renderTabs();renderPanel();camNow=null;redraw()}"); await pg.wait_for_timeout(300); await pg.screenshot(path='shots/g1.png')
        await pg.click('#panel [data-k=gender][data-i="1"]'); await pg.wait_for_timeout(400); await pg.screenshot(path='shots/g2.png')
        await pg.click('#tabs button[data-t=clothes]'); await pg.click('#panel [data-ct=bottom]'); await pg.click('#panel [data-k=bottom][data-i="3"]'); await pg.wait_for_timeout(400); await pg.screenshot(path='shots/c0.png'); await pg.click('#panel [data-ct=shoes]'); await pg.wait_for_timeout(200); await pg.screenshot(path='shots/c5.png')
        print('性別編碼', await pg.evaluate("decodeChar(encodeChar(ch)).gender"), '長度', await pg.evaluate("encodeChar(ch).length"), '舊連結', await pg.evaluate("decodeChar('AQEAAQEDAgABAAAKCgoKCgoKCgoKCgIBAwACAQ').gender"))
        await pg.click('#readBtn'); await pg.click('#skipBtn'); await pg.wait_for_timeout(300)
        await pg.click('#homeLink'); await pg.click('#boothBtn'); await pg.wait_for_timeout(500); await pg.screenshot(path='shots/b1.png')
        shots=2
        for day in range(9):
            for i in range(5):
                if await pg.evaluate("bs.phase")=='guest':
                    await pg.wait_for_timeout(300)
                    if shots<7: await pg.screenshot(path=f'shots/b{shots}.png'); shots+=1
                    g=await pg.evaluate("bs.list[bs.i].guest.ans"); await pg.click(f'#bOpts [data-g="{g}"]')
                else:
                    best=await pg.evaluate("bestParts(bs.list[bs.i].c,bs.list[bs.i].fs)[0]")
                    await pg.click('#bSpots [data-p=nose]'); await pg.click(f'#bSpots [data-p={best}]')
                    if day==0 and i==0: await pg.wait_for_timeout(250); await pg.screenshot(path='shots/c1.png')
                    await pg.click('#bOpts [data-confirm]')
                    if day==0 and i==0:
                        await pg.wait_for_timeout(250); await pg.screenshot(path='shots/c2.png'); await pg.click('#bOpts [data-back]'); await pg.click('#bOpts [data-confirm]')
                    if day==0 and i==0: await pg.wait_for_timeout(200); await pg.screenshot(path='shots/b2.png'); shots=3
                    a=await pg.evaluate("(()=>{const c=bs.list[bs.i],j=judge(c.c),st=c.fs.length===1?j.stars[c.fs[0]]:Math.floor((j.stars[c.fs[0]]+j.stars[c.fs[1]])/2);return st>=4?0:st===3?1:2})()")
                    if day==0 and i==1: a=(a+1)%3
                    await pg.click(f'#bOpts [data-a="{a}"]')
                    if day==0 and i<2: await pg.wait_for_timeout(200); await pg.screenshot(path=f'shots/b{shots}.png'); shots+=1
                await pg.click('#bNext')
            if day==0:
                await pg.wait_for_timeout(200); await pg.screenshot(path='shots/c3.png'); await pg.click('#bOpts [data-rank]'); await pg.fill('#rankNick','小美師'); await pg.click('#rankSave'); await pg.wait_for_timeout(200); await pg.screenshot(path='shots/c4.png'); await pg.click('#rankClose')
            await pg.click('#bOpts [data-day]')
        print('攤位', await pg.evaluate("JSON.stringify(store.booth)"), '禁用詞', await pg.evaluate("checkCopy()"))
        await pg.click('#homeLink'); await pg.click('#dexBtn'); await pg.click('#dexTabs [data-d=guests]'); await pg.wait_for_timeout(200); await pg.screenshot(path='shots/b8.png')
        print(logs or 'no errors'); await b.close()
asyncio.run(main())
