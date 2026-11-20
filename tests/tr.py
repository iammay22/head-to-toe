import asyncio
from playwright.async_api import async_playwright
import os, pathlib
HERE = pathlib.Path(__file__).resolve().parent
INDEX = (HERE.parent / 'index.html').as_uri()
os.makedirs(HERE / 'shots', exist_ok=True); os.chdir(HERE)
async def main():
    async with async_playwright() as p:
        b=await p.chromium.launch(executable_path='/usr/bin/google-chrome',args=['--no-sandbox'])
        pg=await b.new_page(viewport={'width':375,'height':740},device_scale_factor=2)
        await pg.goto(''+INDEX+''); await pg.wait_for_function("!document.getElementById('startBtn').disabled")
        await pg.click('#startBtn'); await pg.wait_for_timeout(300)
        await pg.evaluate("ch.moles=['brow','lobe'];showResult()"); await pg.wait_for_timeout(500)
        await pg.evaluate("document.querySelector('#rRead').closest('details').open=true; document.querySelector('#rRead').scrollIntoView()")
        await pg.wait_for_timeout(300); await pg.screenshot(path='shots/read.png')
        await b.close()
asyncio.run(main())
