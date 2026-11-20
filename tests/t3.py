import asyncio
from playwright.async_api import async_playwright
import os, pathlib
HERE = pathlib.Path(__file__).resolve().parent
INDEX = (HERE.parent / 'index.html').as_uri()
os.makedirs(HERE / 'shots', exist_ok=True); os.chdir(HERE)
async def main():
    async with async_playwright() as p:
        b=await p.chromium.launch(executable_path='/usr/bin/google-chrome',args=['--no-sandbox'])
        pg=await b.new_page(viewport={'width':1100,'height':720})
        logs=[]; pg.on('console',lambda m: logs.append(m.type+': '+m.text) if m.type!='warning' else None); pg.on('pageerror',lambda e: logs.append('PAGEERROR '+str(e)))
        await pg.goto(''+INDEX+''); await pg.wait_for_function("!document.getElementById('startBtn').disabled")
        await pg.click('#startBtn'); await pg.wait_for_timeout(600); await pg.screenshot(path='shots/desk.png')
        r=await pg.evaluate("""()=>{const out={same:true,titles:new Set(),stars:[0,0,0,0,0],main:[0,0,0,0]};
          for(let i=0;i<300;i++){const c=randomChar();const a=JSON.stringify(judge(c)),b=JSON.stringify(judge(JSON.parse(JSON.stringify(c))));if(a!==b)out.same=false;
            const j=judge(c);out.titles.add(j.prefix+j.name);j.stars.forEach(s=>out.stars[s-1]++);out.main[j.main]++; if(i<40){ch=c;prepare(c);makeCard(j);} }
          return {same:out.same,titles:out.titles.size,stars:out.stars,main:out.main,bad:checkCopy()}}""")
        print(r)
        async with pg.expect_download() as d:
            await pg.click('#readBtn'); await pg.click('#skipBtn'); await pg.click('#saveBtn')
        dl=await d.value; path=await dl.path(); import os; print('下載',dl.suggested_filename,os.path.getsize(path)//1024,'KB')
        await pg.click('#backBtn'); await pg.click('#randBtn'); await pg.click('#undoBtn'); await pg.reload(); await pg.wait_for_function("!document.getElementById('startBtn').disabled")
        print('存檔還在:',await pg.evaluate("!!JSON.parse(localStorage.getItem('mianxiang-save-v1')).character"))
        print('\n'.join(logs) or 'no errors'); await b.close()
asyncio.run(main())
