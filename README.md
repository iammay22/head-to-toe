# 從頭相到腳

> 接手開發請先看 [HANDOFF.md](HANDOFF.md)。

捏一個從頭到腳的角色，照傳統相學看相的網頁小遊戲。作者：may22

## 玩法
開 `index.html`，捏人、按「看相」，拿到相格稱號和結果卡。

## 檔案
- `index.html`：遊戲本體（CSS、JS 都在裡面）
- `assets/pack.js`：所有圖片和座標，由腳本產生，遊戲只讀這個檔
- `assets/*.webp`、`assets/*.svg`：切好的正式圖和 SVG 佔位圖（給人看的，遊戲不直接讀）
- `raw/head/`、`raw/body/`、`raw/ui/`：Gemini 生的原圖（頭部特寫、全身、手部特寫）；`raw_old/` 是第一版畫風的舊圖，已停用
- `tools/build_assets.py`：素材處理腳本
- `tools/preview.py`：把正式圖疊成 `preview.png` 檢查對齊
- `tests/`：Playwright 測試腳本
- `HANDOFF.md`：交接文件（架構、補圖流程、待辦）

## 補圖的方法
1. 把 Gemini 的圖照檔名規則放進 `raw/head/` 或 `raw/body/`
   - 頭部特寫：`eyes_apricot.jpg`、`brows_willow.jpg`、`nose_gall.jpg`、`mouth_smile.jpg`、`ears_fortune.jpg`、`face_round.jpg`、`face_square_chin_pointed.jpg`
   - 全身：`top_xxx.jpg`、`bottom_xxx.jpg`、`hair_xxx.jpg`
   - 選項的英文代號看 `index.html` 裡 `PARTS` 的 `id`
2. 執行 `python3 tools/build_assets.py`
3. 同名的佔位圖會被正式圖蓋掉，`index.html` 不用改

## 更新紀錄
- 2026-11-17：第 1 階段（捏人、看相、結果頁、結果卡、24 種基本稱號）
- 2026-11-17：第 2 階段（合盤連結、雙人卡、8 種隱藏稱號、流年滑桿、圖鑑）
- 2026-11-17：第 3 階段（相士攤位、5 級攤位、5 位特別客人）、可選性別、繪圖加速
- 2026-11-17：攤位可以先看多個部位再決定、拿掉站姿、服裝分類、相士排行榜（apps-script/Code.gs）
- 2026-11-17：換成新畫風素材（男女標準人、3 種臉型、7 種眼睛、4 種手），男女各有自己的身體和脖子
- 2026-11-17：補上鼻、嘴、耳、下巴正式圖，高解析脖子，手部特寫
- 2026-11-17：髮型拆成前髮和後髮，前髮 5 種、後髮 8 種正式圖
- 2026-11-18：耳朵 5 種正式圖、圖層順序調整（耳朵在後髮前、前髮後；眉毛壓在眼睛上）、眼睛實心、嘴唇去陰影、頭髮自動補滿被擋住的地方、痣縮小
- 2026-11-18：臉型男女共用，切換性別會重畫頭
- 2026-11-20：面相說明依《圖解麻衣神相》重寫，各部位附古書出處；測試腳本放進 `tests/`；新增交接文件
