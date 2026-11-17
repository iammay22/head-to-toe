# 從頭相到腳

捏一個從頭到腳的角色，照傳統相學看相的網頁小遊戲。作者：may22

## 玩法
開 `index.html`，捏人、按「看相」，拿到相格稱號和結果卡。

## 檔案
- `index.html`：遊戲本體（CSS、JS 都在裡面）
- `assets/pack.js`：所有圖片和座標，由腳本產生，遊戲只讀這個檔
- `assets/*.webp`、`assets/*.svg`：切好的正式圖和 SVG 佔位圖（給人看的，遊戲不直接讀）
- `raw/head/`、`raw/body/`：Gemini 生的原圖
- `tools/build_assets.py`：素材處理腳本
- `tools/preview.py`：把正式圖疊成 `preview.png` 檢查對齊

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
