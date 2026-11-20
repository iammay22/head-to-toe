import asyncio
from playwright.async_api import async_playwright
import os, pathlib
HERE = pathlib.Path(__file__).resolve().parent
INDEX = (HERE.parent / 'index.html').as_uri()
os.makedirs(HERE / 'shots', exist_ok=True); os.chdir(HERE)
async def main():
    async with async_playwright() as p:
        b=await p.chromium.launch(executable_path='/usr/bin/google-chrome',args=['--no-sandbox'])
        pg=await b.new_page(viewport={'width':375,'height':740},device_scale_factor=2); logs=[]
        pg.on('pageerror',lambda e: logs.append('PAGEERROR '+str(e)))
        await pg.goto(''+INDEX+''); await pg.wait_for_function("!document.getElementById('startBtn').disabled")
        await pg.click('#startBtn'); await pg.wait_for_timeout(300)
        await pg.click('#tabs button[data-t=hand]'); await pg.wait_for_timeout(300)
        await pg.screenshot(path='shots/hand1.png')
        await pg.evaluate("change(()=>{ch.skin=4})"); await pg.click('#panel button[data-i="1"]'); await pg.wait_for_timeout(400)
        await pg.screenshot(path='shots/hand2.png')
        await pg.click('#zoomBtn'); await pg.wait_for_timeout(500); await pg.screenshot(path='shots/hand3.png')
        print(logs or 'ok'); await b.close()
asyncio.run(main())
