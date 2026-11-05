"""Prepare a local two-stage human packet; never export reader predictions.

Only creates a new review directory. Does not edit/rerun the original experiment.
"""
import hashlib
import json
import math
from pathlib import Path
import shutil

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'results/real_gauge_feasibility_20261023'
OUT = ROOT/'results/human_reference_review_20261023'
IMAGES = ROOT/'data/task_verification/paddle_audit/inventory_20261023'
IDS = [0, 6, 18, 23, 24, 30, 36, 42, 48]

STYLE = '''
body{font:17px/1.5 system-ui,sans-serif;max-width:1050px;margin:30px auto;padding:0 18px;color:#202020}
h1{font-size:27px}h2{font-size:23px}label{display:block;margin:12px 0}input,textarea,select{font:inherit;padding:5px;max-width:96%;border:1px solid #777}
textarea{display:block;width:95%;height:65px}input[type=number]{width:130px}button{font:inherit;padding:10px;margin:10px 8px 10px 0;cursor:pointer}
img{max-width:100%;height:auto;border:1px solid #ddd}.crop{max-height:620px;width:auto}.case{border-top:3px solid #777;padding:20px 0;margin-top:35px}
.notice{background:#f2f4f7;padding:16px}.prior{background:#fff6de;padding:15px;white-space:pre-wrap}.error{color:#a00000}
@media print{button,.controls{display:none}.case{break-before:page}textarea{height:65px}img{max-height:450px;width:auto}}
'''

JS = r'''
const fieldDefs=[
 ['units','Unit exactly as printed (or unknown)','text'],
 ['minimum','Lowest numbered value','number'],['maximum','Highest numbered value','number'],
 ['numbered_ticks','Transcribe numbered ticks, in increasing scale order','area'],
 ['minor_divisions','Count minor intervals between two named numbered ticks; give values or unknown','area'],
 ['pointer','Identify the pointer tip and which marks bracket it; describe any ambiguity','area'],
 ['counterweight','Identify the counterweight/tail and explain how you distinguished it from the tip','area'],
 ['reference_status','Your reference decision','select'],
 ['low','Plausible lower reading bound (same printed units)','number'],
 ['high','Plausible upper reading bound','number'],
 ['scale_uncertainty','Scale uncertainty: units, tick interpretation, linearity, endpoints','area'],
 ['pointer_uncertainty','Pointer uncertainty: blur/width/visibility and plausible bracket','area'],
 ['interval_basis','Why this interval is plausible, or why not determinable','area']
];
let blind=null;
function el(tag,text){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;return e;}
function formField(parent,id,key,title,type,value){
 const label=el('label',title+' ');let input;
 if(type==='area') input=el('textarea');
 else if(type==='select'){
  input=el('select');[['','Choose'],['interval','Plausible interval'],['not_determinable','Not determinable']].forEach(([v,t])=>{let o=el('option',t);o.value=v;input.append(o);});
 }else{input=el('input');input.type=type;if(type==='number')input.step='any';}
 input.id=id+'_'+key;input.value=value===undefined?'':value;label.append(input);parent.append(label);
}
function photo(parent,path,title,cls){
 const link=el('a',title+' — open full resolution');link.href=path;link.target='_blank';parent.append(link);
 const img=el('img');img.src=path;img.alt=title;img.className=cls||'';parent.append(img);
}
function render(){
 const host=document.getElementById('cases');host.textContent='';
 DATA.forEach(c=>{
  const card=el('section');card.className='case';card.append(el('h2',c.id+' — '+c.file));
  photo(card,'assets/'+c.id+'_original.jpg','Original JPEG');
  photo(card,'assets/'+c.id+'_crop.png','Native-pixel dial crop; no enhancement or geometric warp','crop');
  if(STAGE===2){
   card.append(el('h3','Your recorded first interpretation'));
   const first=blind.cases.find(x=>x.id===c.id);card.append(el('pre',JSON.stringify(first,null,2)));
   card.append(el('h3','Previous model annotation and frozen geometry — assumptions to audit'));
   const previous=el('div',JSON.stringify(c.prior,null,2));previous.className='prior';card.append(previous);
   photo(card,'assets/'+c.id+'_assumptions.png','Assumptions only: red center/circle, blue zero ray, green derived maximum ray','crop');
  }
  const initial=STAGE===2?blind.cases.find(x=>x.id===c.id):{};
  fieldDefs.forEach(([k,t,type])=>formField(card,c.id,k,t,type,initial[k]));
  if(STAGE===2){
   formField(card,c.id,'geometry_uncertainty','Supplied geometry: center, zero/max rays, circularity, 270-degree sweep and identity homography. What is doubtful or contradicted?','area','');
   formField(card,c.id,'reconciliation','What changed from your first interpretation, and why? Unchanged is a valid answer. Do not average incompatible readings.','area','');
  }
  host.append(card);
 });
 document.getElementById('actions').hidden=false;
}
function collect(complete){
 const name=document.getElementById('reviewer').value.trim();
 if(complete&&!name)throw Error('Enter reviewer name or initials.');
 const answers=DATA.map(c=>{
  let row={id:c.id,file:c.file};
  const keys=fieldDefs.map(x=>x[0]).concat(STAGE===2?['geometry_uncertainty','reconciliation']:[]);
  keys.forEach(k=>row[k]=document.getElementById(c.id+'_'+k).value.trim());
  if(complete){
   if(!['interval','not_determinable'].includes(row.reference_status))throw Error(c.id+': choose interval or not determinable.');
   if(!row.interval_basis||!row.scale_uncertainty||!row.pointer_uncertainty||!row.units||!row.numbered_ticks||!row.minor_divisions||!row.pointer||!row.counterweight)throw Error(c.id+': complete interpretation/uncertainty fields; unknown is a valid explicit answer.');
   if(row.reference_status==='interval'){
    if(row.low===''||row.high===''||!Number.isFinite(Number(row.low))||!Number.isFinite(Number(row.high))||Number(row.low)>Number(row.high)||row.minimum===''||row.maximum===''||!Number.isFinite(Number(row.minimum))||!Number.isFinite(Number(row.maximum))||Number(row.maximum)<=Number(row.minimum))throw Error(c.id+': supply units, scale range and ordered finite interval bounds.');
   }
   if(STAGE===2&&(!row.geometry_uncertainty||!row.reconciliation))throw Error(c.id+': record geometry audit and reconciliation.');
  }
  return row;
 });
 if(complete&&!document.getElementById('exposure').value.trim())throw Error('Record prior exposure, including none if appropriate.');
 if(complete&&STAGE===2&&!document.getElementById('recorded').checked)throw Error('Confirm that reference decisions are recorded before opening reader predictions.');
 return {protocol:'site-gauge-human-review-v1',stage:STAGE,complete,reviewer:name,
   exposure_note:document.getElementById('exposure').value,recorded_at:new Date().toISOString(),
   cases:answers,...(STAGE===2?{blind_record:blind,reference_decisions_recorded:true}:{} )};
}
function download(complete){
 try{
  const record=collect(complete),a=el('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(record,null,2)],{type:'application/json'}));
  a.download=(STAGE===1?'blind_review':'reconciled_review')+(complete?'':'_draft')+'.json';a.click();
  document.getElementById('message').textContent=complete?'Record downloaded. Keep this file before proceeding.':'Draft downloaded; final stage1 records are required for stage2.';
 }catch(e){document.getElementById('message').textContent=e.message;}
}
async function loadBlind(event){
 try{
  let obj=JSON.parse(await event.target.files[0].text());
  const draft=obj.stage===2?obj:null;if(draft)obj=draft.blind_record;
  if(obj.protocol!=='site-gauge-human-review-v1'||obj.stage!==1||!obj.complete||obj.cases.length!==DATA.length||new Set(obj.cases.map(x=>x.id)).size!==DATA.length||!DATA.every(c=>obj.cases.some(x=>x.id===c.id&&['interval','not_determinable'].includes(x.reference_status))))throw Error('Choose a completed blind_review.json for all nine cases.');
  blind=obj;document.getElementById('reviewer').value=obj.reviewer;document.getElementById('exposure').value=obj.exposure_note||'';
  document.getElementById('gate').hidden=true;render();
  if(draft)draft.cases.forEach(r=>Object.keys(r).forEach(k=>{const field=document.getElementById(r.id+'_'+k);if(field)field.value=r[k];}));
 }catch(e){document.getElementById('message').textContent=e.message;}
}
async function loadDraft(event){
 try{
  const obj=JSON.parse(await event.target.files[0].text());
  if(obj.protocol!=='site-gauge-human-review-v1'||obj.stage!==1||obj.cases.length!==DATA.length||!DATA.every(c=>obj.cases.some(x=>x.id===c.id)))throw Error('Choose your stage1 draft or completed record.');
  document.getElementById('reviewer').value=obj.reviewer;document.getElementById('exposure').value=obj.exposure_note||'';
  obj.cases.forEach(r=>Object.keys(r).forEach(k=>{const field=document.getElementById(r.id+'_'+k);if(field)field.value=r[k];}));
 }catch(e){document.getElementById('message').textContent=e.message;}
}
if(STAGE===1)render();
'''


def page(stage, data):
    intro = ('First interpret each image from its pixels. Prior model readings and geometry are absent from this page. '
             'Record units, numbered ticks, minor intervals, the pointer and counterweight before giving an interval. '
             'Unknown / not determinable is valid. Do not interpolate merely to produce a verdict.' if stage == 1 else
             'Import your completed stage1 record to expose previous intervals and frozen geometry. '
             'Preserve your first interpretation; reconcile only from visible evidence. Reader predictions are absent from this page.')
    gate = ('<div class="controls">Optional: resume your stage1 draft <input type="file" accept=".json" onchange="loadDraft(event)"></div>' if stage == 1 else '<div id="gate" class="notice">Load completed blind_review.json (or a saved stage2 draft): <input type="file" accept=".json" onchange="loadBlind(event)"></div>')
    confirmation = '' if stage == 1 else '<label><input id="recorded" type="checkbox"> I have recorded these reference decisions before opening reader predictions.</label>'
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Gauge human review stage '+str(stage)+'</title><style>'+STYLE+'</style><body>'
            '<h1>Human reference review — stage '+str(stage)+'</h1><p class="notice">'+intro+'</p>'
            '<p>This packet contains nine previously accepted photographs; acceptance does not establish their numerical correctness. '
            'Open images at native resolution. A PNG crop preserves decoded pixels; it adds no detail. '
            'Identify printed zero marks separately from mechanical stops. Count intervals between tick marks, not just marks.</p>'
            '<label>Reviewer name or initials <input id="reviewer"></label>'
            '<label>Prior exposure to old annotations or reader predictions (state none, reports already seen, or details)<textarea id="exposure"></textarea></label>'
            +gate+'<div id="cases"></div><div id="actions" hidden>'+confirmation+
            '<button onclick="download(false)">Save draft JSON</button><button onclick="download(true)">Record completed stage '+str(stage)+'</button>'
            '<p>No answers are uploaded or saved automatically. Keep the downloaded JSON before closing. You can reload drafts with the file control above.</p></div>'
            '<p id="message" class="error" role="status"></p><script>const STAGE='+str(stage)+';const DATA='+json.dumps(data)+';'+JS+'</script></body></html>')


def main():
    OUT.mkdir(exist_ok=False)
    (OUT/'assets').mkdir()
    originals = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in SOURCE.rglob('*') if p.is_file()}
    report = ROOT/'docs/task_verification/real_image_result.md'
    originals[str(report.relative_to(ROOT))] = hashlib.sha256(report.read_bytes()).hexdigest()
    (OUT/'preservation_before.json').write_text(json.dumps(originals, indent=2)+'\n')
    labels = {r['index']: r for r in json.loads((SOURCE/'reference_labels.json').read_text())}
    configs = {r['index']: r for r in json.loads((SOURCE/'method_inputs.json').read_text())}
    blind, reconciliation, inventory = [], [], []
    for n, i in enumerate(IDS, 1):
        label, g = labels[i], configs[i]
        sid = 'R'+str(n).zfill(2)
        original = IMAGES/label['file']
        shutil.copyfile(original, OUT/'assets'/f'{sid}_original.jpg')
        im = Image.open(original).convert('RGB')
        cx, cy = g['center']; r = g['radius']; padding = math.ceil(r*.25)+10
        box = (max(0,cx-r-padding), max(0,cy-r-padding), min(im.width,cx+r+padding), min(im.height,cy+r+padding))
        crop = im.crop(box); crop.save(OUT/'assets'/f'{sid}_crop.png')
        # Rendering is separate from the unaltered crop. No pointer annotation.
        overlay = im.copy(); draw = ImageDraw.Draw(overlay)
        draw.ellipse((cx-r,cy-r,cx+r,cy+r),outline='red',width=2)
        draw.line((cx-7,cy,cx+7,cy),fill='red',width=2); draw.line((cx,cy-7,cx,cy+7),fill='red',width=2)
        endpoints = []
        for angle, color in [(g['scale_start_deg'],'blue'), (g['scale_start_deg']+g['scale_sweep_deg'],'green')]:
            endpoint = [cx+r*math.cos(math.radians(angle)),cy+r*math.sin(math.radians(angle))]
            endpoints.append(endpoint); draw.line((cx,cy,*endpoint),fill=color,width=2)
        overlay.crop(box).save(OUT/'assets'/f'{sid}_assumptions.png')
        item = {'id': sid, 'file':label['file']}
        blind.append(item)
        prior = {'previous_interval': label['reference_interval'], 'previous_status':label['reference_status'],
                 'previous_units':label['unit'], 'previous_basis':label['label_basis'],
                 'previous_annotator':'One language-model visual annotator, no independent human/reference instrument',
                 'frozen_assumptions':g, 'fixed_absolute_tolerance':.02*(g['max_value']-g['min_value']), 'derived_zero_endpoint_px':endpoints[0],
                 'derived_maximum_endpoint_px':endpoints[1],
                 'endpoint_caution':'Maximum is derived from assumed270-degree sweep, not independently annotated from a printed maximum tick.'}
        reconciliation.append({**item,'prior':prior})
        inventory.append({**item,'source_index':i,'crop_box_xyxy':box,'native_crop_size':crop.size,
                          'assumed_dial_diameter_px':2*r,'fixed_tolerance_units':.02*(g['max_value']-g['min_value']),
                          'nominal_tolerance_at_0.75R_px':.75*r*math.radians(.02*g['scale_sweep_deg']),
                          'original_sha256':hashlib.sha256(original.read_bytes()).hexdigest()})
    (OUT/'stage1_blind.html').write_text(page(1, blind))
    (OUT/'stage2_reconcile.html').write_text(page(2, reconciliation))
    (OUT/'packet_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    (OUT/'status.json').write_text(json.dumps({'status':'awaiting_human_review','reference_review_completed':False,
         'predictions_exposed_in_packet':False,'new_classifications':None,'abstentions_in_original_experiment':2,
         'source_failures_in_original_experiment':1,'reader_runs_this_task':0},indent=2)+'\n')
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest for p,digest in originals.items())
    print(OUT)


if __name__ == '__main__':
    main()
