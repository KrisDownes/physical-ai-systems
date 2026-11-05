"""One-case geometric analysis of an existing ray. Does not import/run the reader.

Image-mark observations are manually recorded independently of the270-degree
model. Local interpolations are post-hoc explanatory diagnostics, not predictions.
"""
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/r02_geometry_diagnostic_20261023'


def angle(point, center):
    d = np.array(point)-center
    return math.degrees(math.atan2(d[1], d[0])) % 360


def main():
    assert not (OUT/'analysis.json').exists(), 'Preserve the first diagnostic.'
    observations = json.loads((OUT/'image_observations.json').read_text())
    old = ROOT/'results/real_gauge_feasibility_20261023'
    config = next(x for x in json.loads((old/'method_inputs.json').read_text()) if x['index'] == 6)
    pred = next(x for x in json.loads((old/'predictions.json').read_text()) if x['index'] == 6)['prediction']
    human = next(x for x in json.loads((ROOT/'results/human_reference_review_20261023/final_reference_record.json').read_text())['cases'] if x['id']=='R02')
    assert [human['low'], human['high'], human['units']] == ['6', '6.5', 'MPa']
    offset = np.array(observations['crop_box_xyxy'][:2])
    center = np.array(config['center'])-offset
    start, sweep, span = config['scale_start_deg'], config['scale_sweep_deg'], config['max_value']-config['min_value']
    theta = start + pred['reading_fraction']*sweep
    marks = []
    for m in observations['marks']:
        midpoint = (np.array(m['outer'])+m['inner'])/2
        phi = angle(midpoint, center)
        unwrapped = phi if phi >= start-10 else phi+360
        # The numbered25 graduation is beyond360 degrees in this orientation.
        if m['value'] >= 20 and unwrapped < start:
            unwrapped += 360
        marks.append({**m,'midpoint':midpoint.tolist(),'angle_deg':unwrapped,
                      'value_assigned_by_frozen_mapping':span*(unwrapped-start)/sweep})
    by_value = {m['value']:m for m in marks}
    t6, t65 = by_value[6]['angle_deg'], by_value[6.5]['angle_deg']
    local_value = 6+(theta-t6)/(t65-t6)*.5
    t5, t10 = by_value[5]['angle_deg'], by_value[10]['angle_deg']
    broad_local = 5+(theta-t5)/(t10-t5)*5
    effective_zero = t5-sweep*5/span
    fixed_slope_counterfactual = span*(theta-effective_zero)/sweep
    visible_tip_angle = angle(observations['visible_pointer_tip'], center)
    # Sensitivity of the post-hoc interpolation to hand-placed tick locations.
    candidates = []
    for dx6,dy6 in [(-2,-2),(-2,2),(2,-2),(2,2)]:
        for dx65,dy65 in [(-2,-2),(-2,2),(2,-2),(2,2)]:
            a = angle(np.array(by_value[6]['midpoint'])+[dx6,dy6],center)
            b = angle(np.array(by_value[6.5]['midpoint'])+[dx65,dy65],center)
            candidates.append(6+(theta-a)/(b-a)*.5)
    analysis = {'case':'R02','reader_runs':0,'new_evaluation_images':0,
        'superseded_model_reference_used':False,'human_reference':{'interval':[6,6.5],'units':'MPa','unchanged':True},
        'frozen':{'center_crop_px':center.tolist(),'zero_deg':start,'sweep_deg':sweep,'span_MPa':span,
                  'saved_fraction':pred['reading_fraction'],'saved_prediction_MPa':pred['reading_value'],
                  'reconstructed_needle_angle_deg':theta,'angle_delta_from_zero_deg':theta-start},
        'observed_marks':marks,'visible_tip_angle_deg':visible_tip_angle,
        'tip_vs_saved_ray_difference_deg':visible_tip_angle-theta,
        'local6_to6_5_interpolation_MPa':local_value,'local5_to10_interpolation_MPa':broad_local,
        'local5_to10_implied_full_scale_sweep_deg':(t10-t5)*span/5,
        'zero_implied_by_observed5_and_frozen_slope_deg':effective_zero,
        'implied_zero_minus_supplied_zero_deg':effective_zero-start,
        'counterfactual_change_only_zero_MPa':fixed_slope_counterfactual,
        'local_interpolation_tick_midpoint_plus_minus2px_range_MPa':[min(candidates),max(candidates)],
        'sensitivity_caution':'Geometric endpoint sensitivity only, fixed assumed center; not a confidence interval or full perspective/annotation uncertainty.',
        'causal_assessment':'Saved needle ray agrees with visible pointer; frozen angle/value calibration disagrees with nearby printed graduations. Effective zero mismatch is supported. Exact split among endpoint selection, center/scale-plane geometry and perspective remains unidentified.',
        'decision':'One small conventional known-asset calibration correction is worth developing; no generic reader expansion or validation claim.'}
    (OUT/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
    with (OUT/'printed_tick_comparison.csv').open('w') as f:
        f.write('printed_value_MPa,image_angle_deg,frozen_mapping_MPa\n')
        for m in marks:f.write(f"{m['value']},{m['angle_deg']:.6f},{m['value_assigned_by_frozen_mapping']:.6f}\n")
    image_path = ROOT/'data/task_verification/paddle_audit/inventory_20261023'/observations['source_image']
    crop = Image.open(image_path).convert('RGB').crop(tuple(observations['crop_box_xyxy']))
    crop.resize((1700,1700),Image.Resampling.NEAREST).save(OUT/'r02_unannotated_4x.png')
    obs_color, assumed_color, output_color = '#007a80','#d77a00','#cb2ca1'
    fig, axes = plt.subplots(1,2,figsize=(14,8),gridspec_kw={'width_ratios':[1.3,1]})
    a,b = axes
    for ax in axes:
        ax.imshow(crop);ax.set_aspect('equal');ax.set_xlabel('Native crop x [px]');ax.set_ylabel('Native crop y [px]')
    a.set_title('Whole dial: image observations vs supplied geometry',fontsize=12)
    a.plot(*center,marker='+',color=assumed_color,markersize=14,markeredgewidth=2)
    circle=plt.Circle(center,config['radius'],fill=False,color=assumed_color,ls=':',lw=1);a.add_patch(circle)
    for phi,label,ls in [(start,'supplied zero','--'),(start+sweep,'derived 25 endpoint',':')]:
        q=center+config['radius']*np.array([math.cos(math.radians(phi)),math.sin(math.radians(phi))])
        a.plot([center[0],q[0]],[center[1],q[1]],color=assumed_color,ls=ls,lw=1.8)
        a.annotate(label,xy=q,xytext=((5,418) if phi==start else (290,395)),fontsize=8,color=assumed_color,arrowprops={'arrowstyle':'-','color':assumed_color},bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'})
    q=center+190*np.array([math.cos(math.radians(theta)),math.sin(math.radians(theta))])
    for ax in axes:
        ax.plot([center[0],q[0]],[center[1],q[1]],color=output_color,ls='--',lw=1.6)
        for m in marks:
            ax.plot([m['outer'][0],m['inner'][0]],[m['outer'][1],m['inner'][1]],color=obs_color,lw=1.5,alpha=.85)
    label_locations={0:(18,370),5:(4,235),10:(66,38),15:(275,25),20:(390,142),25:(375,315)}
    for value,xytext in label_locations.items():
        m=by_value[value]
        label='0-associated feature (?)' if value==0 else str(value)+' printed'
        a.annotate(label,xy=m['midpoint'],xytext=xytext,color=obs_color,fontsize=8,arrowprops={'arrowstyle':'-','color':obs_color},bbox={'facecolor':'white','alpha':.8,'edgecolor':'none'})
    a.annotate('counterweight',xy=observations['counterweight_point'],xytext=(290,245),fontsize=8,color=obs_color,arrowprops={'arrowstyle':'->','color':obs_color},bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'})
    a.annotate('supplied center',xy=center,xytext=(245,190),fontsize=8,color=assumed_color,arrowprops={'arrowstyle':'-','color':assumed_color},bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'})
    a.set_xlim(-5,455);a.set_ylim(425,-5)
    b.set_title('Printed 5–10 scale; intermediate marks counted from pixels',fontsize=10)
    for m in marks:
        if 5<=m['value']<=10:
            b.annotate(str(m['value']),xy=m['midpoint'],xytext=(m['outer'][0]-26,m['outer'][1]+1),fontsize=9,color=obs_color,arrowprops={'arrowstyle':'-','color':obs_color},bbox={'facecolor':'white','alpha':.8,'edgecolor':'none'})
    b.plot(*observations['visible_pointer_tip'],marker='o',markersize=7,fillstyle='none',color=obs_color)
    b.set_xlim(0,150);b.set_ylim(237,58)
    fig.legend(handles=[Line2D([0],[0],color=obs_color,lw=2,label='Teal: manually observed image marks'),
                        Line2D([0],[0],color=assumed_color,lw=2,ls='--',label='Orange: frozen supplied assumptions'),
                        Line2D([0],[0],color=output_color,lw=2,ls='--',label='Magenta: saved reader direction')],
               loc='lower center',bbox_to_anchor=(.5,.105),ncol=1,frameon=False,fontsize=10)
    fig.suptitle('R02 diagnostic — frozen prediction preserved; human reference [6, 6.5] MPa',fontsize=14)
    fig.text(.06,.065,f'Saved mapping: 25 × ({theta:.3f}° − {start:.3f}°) / 270° = 5.625 MPa',fontsize=11)
    fig.text(.06,.027,f'Observed local ticks place the SAME ray near {local_value:.2f} MPa. Post-hoc diagnosis only; no reader rerun or corrected evaluation.',fontsize=10)
    fig.subplots_adjust(left=.055,right=.97,top=.88,bottom=.27,wspace=.25)
    fig.savefig(OUT/'r02_overlay.png',dpi=160,facecolor='white');plt.close(fig)
    preserved=json.loads((OUT/'preservation_before.json').read_text())
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest for p,digest in preserved.items())
    print(json.dumps({k:v for k,v in analysis.items() if k not in ['observed_marks']},indent=2))


if __name__=='__main__':main()
