import asyncio, base64, json
from playwright.async_api import async_playwright
import os, pathlib
HERE = pathlib.Path(__file__).resolve().parent
INDEX = (HERE.parent / 'index.html').as_uri()
os.makedirs(HERE / 'shots', exist_ok=True); os.chdir(HERE)
URL=''+INDEX+''
A=dict(face=1,brows=1,eyes=1,shoulder=1,nose=3,mouth=2,hair=3,hairColor=0,skin=1,top=2,bottom=1,moles=['tear'])
B=dict(face=4,eyes=0,brows=2,mouth=1,hair=5,hairColor=5,skin=3,moles=['tear'],top=3,bottom=4)
async def newpage(b,logs,tag):
    pg=await b.new_page(viewport={'width':375,'height':740},device_scale_factor=2)
    pg.on('console',lambda m: logs.append(tag+' '+m.type+': '+m.text) if m.type in('error',) else None)
    pg.on('pageerror',lambda e: logs.append(tag+' PAGEERROR '+str(e)))
    return pg
async def ready(pg): await pg.wait_for_function("!document.getElementById('startBtn').disabled")
async def main():
    async with async_playwright() as p:
        b=await p.chromium.launch(executable_path='/usr/bin/google-chrome',args=['--no-sandbox'])
        logs=[]
        # A 看相、產生邀請
        pa=await newpage(b,logs,'A'); await pa.goto(URL); await ready(pa)
        await pa.click('#startBtn'); await pa.evaluate("c=>{ch=Object.assign(defaultChar(),c);persist();redraw()}",A)
        await pa.click('#readBtn'); await pa.click('#skipBtn'); await pa.wait_for_timeout(500)
        print('A 結果', await pa.evaluate("JSON.stringify([judge(ch).prefix,judge(ch).name])"))
        await pa.screenshot(path='shots/s2_resultA.png')
        await pa.evaluate("document.getElementById('lnView').scrollIntoView()"); await pa.wait_for_timeout(200); await pa.screenshot(path='shots/s2_liunian.png')
        await pa.evaluate("document.getElementById('lnAge').value=62;drawLiunian()"); await pa.wait_for_timeout(100); await pa.screenshot(path='shots/s2_liunian2.png')
        await pa.click('#pairBtn'); await pa.click('#relSeg button[data-r=friend]'); await pa.fill('#nickIn','小美'); await pa.click('#makeLink'); await pa.wait_for_timeout(400)
        await pa.screenshot(path='shots/s2_sheet.png')
        link=await pa.input_value('#linkBox'); print('邀請連結', link.split('#')[1], '角色編碼長度', len(link.split('p=')[1].split('&')[0]))
        # B 打開
        pb=await newpage(b,logs,'B'); await pb.goto(link); await ready(pb); await pb.wait_for_timeout(400)
        await pb.screenshot(path='shots/s2_invite.png')
        await pb.fill('#invNick','阿哲'); await pb.click('#invStart'); await pb.evaluate("c=>{ch=Object.assign(defaultChar(),c);persist();redraw()}",B)
        await pb.click('#readBtn'); await pb.click('#skipBtn'); await pb.wait_for_timeout(400); await pb.screenshot(path='shots/s2_resultB.png')
        await pb.click('#pairGo'); await pb.wait_for_timeout(1800); await pb.screenshot(path='shots/s2_pair.png')
        sb=await pb.text_content('#pScore'); print('B 看到', sb, await pb.text_content('#pTitle'))
        d=await pb.evaluate("makePairCard().toDataURL('image/png')"); open('shots/s2_paircard.png','wb').write(base64.b64decode(d.split(',')[1]))
        await pb.click('#pReturn'); await pb.wait_for_timeout(300); back=await pb.input_value('#pLink'); print('傳回連結長度', len(back))
        pc=await newpage(b,logs,'C'); await pc.goto(back); await ready(pc); await pc.wait_for_timeout(1800)
        sa=await pc.text_content('#pScore'); print('A 看到', sa, await pc.text_content('#pTitle'), '一致' if sa==sb else '不一致!!')
        await pc.screenshot(path='shots/s2_pairA.png')
        # 對稱性、分佈、隱藏稱號
        r=await pc.evaluate("""()=>{let asym=0;const dist={};let maxLen=0;
          for(let i=0;i<2000;i++){const a=randomChar(),b=randomChar();const x=pairScore(a,b,'甲','乙','love').score,y=pairScore(b,a,'乙','甲','love').score;if(x!==y)asym++;
            const band=PAIR_TITLES.find(t=>x>=t[0])[0];dist[band]=(dist[band]||0)+1;maxLen=Math.max(maxLen,encodeChar(a).length);
            const d=decodeChar(encodeChar(a)); if(JSON.stringify(judge(d))!==JSON.stringify(judge(a))) asym+=1000;}
          const H={};for(let i=0;i<60000;i++){const c=randomChar();const j=judge(c);if(j.hidden)H[j.hidden.id]=(H[j.hidden.id]||0)+1;}
          return {asym,dist,maxLen,hidden:H,bad:checkCopy(),bogus:decodeChar('!!!')===null}}""")
        print(json.dumps(r,ensure_ascii=False))
        # 圖鑑
        
        await pa.evaluate("go('home');refreshHome()"); await pa.click('#dexBtn'); await pa.wait_for_timeout(200); await pa.screenshot(path='shots/s2_dex1.png')
        await pa.click('#dexTabs button[data-d=titles]'); await pa.wait_for_timeout(200); await pa.screenshot(path='shots/s2_dex2.png')
        print('\n'.join(logs) or 'no errors'); await b.close()
asyncio.run(main())
