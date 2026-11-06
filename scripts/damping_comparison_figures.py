"""Presentation only: plots/HTML built from evaluated damping records.

No physical formulas or fits belong here. Static figures are suitable for
scientific export; the complete numerical table is retained in the report.
"""
from __future__ import annotations
import html
from pathlib import Path
import re

import numpy as np

from physical_process.damping_models import REPO_MODEL, XUE_MODEL


COLORS = {REPO_MODEL:'#2a78d6', XUE_MODEL:'#eb6834'}
NAMES = {REPO_MODEL:'Repository', XUE_MODEL:'Xue fixed reference'}
STYLES = {REPO_MODEL:'-', XUE_MODEL:'--'}


def make_figures(output, population, flux, fixed, tracks):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
        'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#b7c1ca',
        'axes.labelcolor':'#263746','text.color':'#263746','xtick.color':'#526371',
        'ytick.color':'#526371','axes.facecolor':'#fcfcfb','figure.facecolor':'#fcfcfb',
        'grid.color':'#e1e7ec','grid.linewidth':.6,'lines.linewidth':2.,
        'savefig.facecolor':'#fcfcfb'})
    fig,axes=plt.subplots(2,2,figsize=(12,8.5),layout='constrained')
    t=np.array([r['time_right_myr'] for r in flux])
    mid=np.array([np.sqrt(r['time_left_myr']*r['time_right_myr']) if r['time_left_myr']>0
                  else r['time_right_myr']/2 for r in flux])
    for model,prefix in ((REPO_MODEL,'repo'),(XUE_MODEL,'xue')):
        color=COLORS[model];name=NAMES[model];style=STYLES[model]
        values=np.array([r[prefix+'_flux'] if r[prefix+'_flux'] is not None else np.nan for r in flux])
        low=np.array([r[prefix+'_flux_ci95'][0] if r[prefix+'_flux_ci95'] else np.nan for r in flux])
        high=np.array([r[prefix+'_flux_ci95'][1] if r[prefix+'_flux_ci95'] else np.nan for r in flux])
        axes[0,0].plot(mid,values,style,color=color,label=name)
        axes[0,0].fill_between(mid,low,high,color=color,alpha=.1)
        fraction=np.array([r[prefix+'_cumulative_fraction'] if r[prefix+'_cumulative_fraction'] is not None else np.nan for r in flux])
        ci=[r[prefix+'_cumulative_fraction_ci95'] for r in flux]
        axes[0,1].step(t,fraction,where='post',color=color,linestyle=style,label=name)
        axes[0,1].fill_between(t,[v[0] if v else np.nan for v in ci],[v[1] if v else np.nan for v in ci],
                              color=color,alpha=.1,step='post')
        selected=[r for r in tracks if r['model_id']==model and r['source_id']=='trajectory_8deg']
        time=[r['time_myr'] for r in selected]
        axes[1,0].plot(time,[np.rad2deg(r['inclination_rad']) for r in selected],style,color=color,label=name)
        axes[1,1].plot(time,[r['radius_cm']/selected[0]['radius_cm'] for r in selected],style,color=color,label=name)
        if selected:
            axes[1,0].plot(time[-1],np.rad2deg(selected[-1]['inclination_rad']),'o',color=color,
                           markersize=8,markeredgecolor='#fcfcfb',markeredgewidth=1.5)
    axes[0,0].set(xscale='log',xlabel='First-embedding time (Myr)',ylabel='Weighted objects / Myr',
                   title='First-embedding flux · realization mean ± 95% interval')
    axes[0,1].set(xscale='log',xlabel='Time (Myr)',ylabel='Fraction of total input weight',ylim=(-.02,1.05),
                   title='Cumulative embedding · all sources in denominator')
    axes[1,0].set(xscale='log',xlabel='Time (Myr)',ylabel='Inclination (degrees)',
                   title='Matched 8° source · inclination history')
    axes[1,1].set(xscale='log',xlabel='Time (Myr)',ylabel='Semimajor axis / initial semimajor axis',
                   title='Matched 8° source · radius evolution')
    axes[1,1].text(.04,.08,'Xue: fixed reference radius is an assumption',transform=axes[1,1].transAxes,
                    fontsize=9,color='#526371')
    for ax in axes.flat:
        ax.grid(True,axis='y')
        ax.legend(loc='best',frameon=False,fontsize=9)
        ax.margins(x=.03)
    fig.suptitle('Damping benchmark · conditional combined-fit comparison',fontsize=15,fontweight='medium')
    fig.savefig(output/'damping_comparison.png',dpi=170)
    fig.savefig(output/'damping_comparison.pdf')
    plt.close(fig)
    # A second figure resolves source-state dependence rather than adding axes
    # with incompatible units to the time-domain panels.
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for model in (REPO_MODEL,XUE_MODEL):
        selected=[r for r in fixed if r['model_id']==model and r['t_embed_myr'] is not None]
        axes[0].plot([np.rad2deg(r['inclination_rad']) for r in selected],
                     [r['t_embed_myr'] for r in selected],STYLES[model],color=COLORS[model],
                     marker='o',markersize=8,label=NAMES[model])
        events=sorted([r for r in population if r['model_id']==model and r['t_embed_myr'] is not None],
                      key=lambda r:r['t_embed_myr'])
        if events:
            denominator=sum(r['source_weight'] for r in population if r['model_id']==model)
            cumulative=np.cumsum([r['source_weight'] for r in events])/denominator
            axes[1].step([r['t_embed_myr'] for r in events],cumulative,where='post',color=COLORS[model],
                         linestyle=STYLES[model],label=NAMES[model])
    axes[0].set(yscale='log',xlabel='Initial inclination (degrees)',ylabel='First-embedding time (Myr)',
                title='Fixed sources · common local environment')
    axes[1].set(xscale='log',xlabel='First-embedding time (Myr)',ylabel='Fraction of total input weight',
                ylim=(-.02,1.05),title='Observed weighted events · lower bound if incomplete')
    axes[1].text(.04,.1,'Censored and unavailable weight stays in denominator',transform=axes[1].transAxes,
                 fontsize=9,color='#526371')
    for ax in axes:
        ax.grid(True,axis='y');ax.legend(frameon=False)
    fig.savefig(output/'damping_distributions.png',dpi=170)
    fig.savefig(output/'damping_distributions.pdf')
    plt.close(fig)


def markdown_html(text):
    """Small escaped renderer for our generated prose, code and pipe tables."""
    result=[];paragraph=[];code=[];table=[];in_code=False
    def inline(s):
        s=html.escape(s)
        s=re.sub(r'`([^`]+)`',r'<code>\1</code>',s)
        s=re.sub(r'(https://[^\s<]+)',r'<a href="\1">\1</a>',s)
        return s
    def flush():
        if paragraph:
            result.append('<p>'+inline(' '.join(paragraph))+'</p>');paragraph.clear()
        if table:
            header=table[0]
            body=table[2:]
            result.append('<div class="table-scroll"><table><thead><tr>'+''.join('<th>'+inline(c)+'</th>' for c in header)+'</tr></thead><tbody>')
            for row in body:
                result.append('<tr>'+''.join('<td>'+inline(c)+'</td>' for c in row)+'</tr>')
            result.append('</tbody></table></div>');table.clear()
    for line in text.splitlines():
        if line.startswith('```'):
            if in_code:
                result.append('<pre><code>'+html.escape('\n'.join(code))+'</code></pre>');code=[]
            else:
                flush()
            in_code=not in_code
        elif in_code: code.append(line)
        elif line.startswith('|'):
            if paragraph: flush()
            table.append([part.strip() for part in line.strip('|').split('|')])
        elif line.startswith('#'):
            flush();n=len(line)-len(line.lstrip('#'))
            result.append(f'<h{n}>'+inline(line[n:].strip())+f'</h{n}>')
        elif line.startswith('!['):
            flush()  # figures are inserted above the report instead
        elif line.strip():
            if table: flush()
            paragraph.append(line.lstrip('- '))
        else: flush()
    flush()
    return '\n'.join(result)


def make_html_report(output):
    images=[]
    for name in ('damping_comparison.png','damping_distributions.png'):
        images.append(f'<figure><img src="{name}" alt="Damping comparison scientific figure; numerical values follow in the report tables."></figure>')
    content=markdown_html((output/'damping_model_comparison_report.md').read_text())
    style='''<title>IMRI Damping Benchmark</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Semi+Condensed:wght@500;600&family=Source+Sans+3:wght@400;600&display=swap">
<style>
:root{--ground:#eef2f5;--paper:#fcfcfb;--ink:#263746;--muted:#526371;--rule:#d7e0e7;--link:#225c9b}
@media(prefers-color-scheme:dark){:root:not([data-theme="light"]){--ground:#121a21;--paper:#1c2832;--ink:#e2eaf0;--muted:#b3c4d0;--rule:#3b4e5e;--link:#9bc5ef}}
:root[data-theme="dark"]{--ground:#121a21;--paper:#1c2832;--ink:#e2eaf0;--muted:#b3c4d0;--rule:#3b4e5e;--link:#9bc5ef}
*{box-sizing:border-box}body{background:var(--ground);color:var(--ink);font-family:"Source Sans 3",Arial,sans-serif;font-size:17px;line-height:1.65}
main{max-width:1150px;margin:auto;padding:36px clamp(18px,4vw,54px);background:var(--paper)}h1,h2,h3{font-family:"Barlow Semi Condensed","Arial Narrow",sans-serif;line-height:1.15;text-wrap:balance;color:var(--ink)}h1{font-size:40px;margin:0 0 24px}h2{font-size:28px;margin:40px 0 18px}p{max-width:76ch;margin:16px 0}a{color:var(--link);overflow-wrap:anywhere}a:focus-visible{outline:3px solid var(--link)}figure{margin:24px 0}img{display:block;width:100%;height:auto;border:1px solid var(--rule)}.table-scroll{overflow-x:auto;margin:22px 0;max-width:100%}table{border-collapse:collapse;min-width:100%;font-size:14px;font-variant-numeric:tabular-nums;line-height:1.45}th,td{padding:12px 14px;border-bottom:1px solid var(--rule);text-align:left;vertical-align:top}th{font-weight:600;color:var(--ink)}td{max-width:320px;overflow-wrap:anywhere}pre{overflow-x:auto;background:var(--ground);padding:18px;font-size:13px}code{font-family:"DejaVu Sans Mono",monospace;font-size:.85em}.eyebrow{color:var(--muted);font-size:13px;letter-spacing:.08em;text-transform:uppercase}.scope{padding:16px 0;border-bottom:2px solid var(--rule);max-width:80ch}
</style>'''
    page=style+'<main><div class="eyebrow">Disk crossing physics · reproducible numerical experiment</div><p class="scope">Conditional combined-fit comparison. Xue’s radius is held fixed; missing orbital losses and unsupported states remain explicit. Full equations, uncertainty definitions and reproduction commands follow the figures.</p>'+''.join(images)+content+'</main>'
    (output/'damping_benchmark.html').write_text(page)
