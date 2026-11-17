// 《從頭相到腳》試算表後端
// 分頁 events：時間、session、事件、數值、裝置、來源
// 分頁 scores：id、暱稱、口碑、銅板、開攤天數、更新時間
function sheet_(name, header) {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  let sh = ss.getSheetByName(name);
  if (!sh) { sh = ss.insertSheet(name); sh.appendRow(header); }
  return sh;
}
function clean_(s) { return String(s || '').replace(/[<>&"=]/g, '').slice(0, 8); }

function doPost(e) {
  const d = JSON.parse(e.postData.contents);
  if (d.event === 'booth_score') {
    const sh = sheet_('scores', ['id', '暱稱', '口碑', '銅板', '開攤天數', '更新時間']);
    const id = String(d.id || '').slice(0, 24); if (!id) return ContentService.createTextOutput('no id');
    const row = [id, clean_(d.name) || '無名相士', Math.min(99999, Number(d.rep) || 0), Number(d.coins) || 0, Number(d.days) || 0, new Date()];
    const ids = sh.getRange(1, 1, Math.max(1, sh.getLastRow()), 1).getValues().map(function (r) { return r[0]; });
    const at = ids.indexOf(id);
    if (at > 0) sh.getRange(at + 1, 1, 1, row.length).setValues([row]); else sh.appendRow(row);
    return ContentService.createTextOutput('ok');
  }
  sheet_('events', ['時間', 'session', '事件', '數值', '裝置', '來源'])
    .appendRow([new Date(), d.session || '', d.event || '', d.value || '', d.device || '', d.source || '']);
  return ContentService.createTextOutput('ok');
}

function doGet(e) {
  if (e && e.parameter && e.parameter.action === 'top') {
    const sh = sheet_('scores', ['id', '暱稱', '口碑', '銅板', '開攤天數', '更新時間']);
    const rows = sh.getLastRow() > 1 ? sh.getRange(2, 1, sh.getLastRow() - 1, 5).getValues() : [];
    const top = rows.map(function (r) { return {id: r[0], name: r[1], rep: Number(r[2]) || 0}; })
      .sort(function (a, b) { return b.rep - a.rep; }).slice(0, 30);
    return ContentService.createTextOutput(JSON.stringify({rows: top, total: rows.length})).setMimeType(ContentService.MimeType.JSON);
  }
  return ContentService.createTextOutput('alive');
}
