"""Editable conceptual diagrams for presentation stages 3 and 4."""

import argparse
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch,FancyBboxPatch

BLUE='#086395'
ORANGE='#ff6b29'


def canvas():
    fig,ax=plt.subplots(figsize=(3,2.1))
    fig.subplots_adjust(left=.01,right=.99,bottom=.01,top=.99)
    ax.set(xlim=(0,300),ylim=(0,210));ax.set_axis_off()
    return fig,ax


def box(ax,x,y,width,height,label):
    ax.add_patch(FancyBboxPatch((x,y),width,height,boxstyle='round,pad=0,rounding_size=10',
        linewidth=1.3,edgecolor=ORANGE,facecolor='white'))
    ax.text(x+width/2,y+height/2,label,ha='center',va='center',fontsize=10.5,color='#111111')


def arrow(ax,start,end,curve=0):
    ax.add_patch(FancyArrowPatch(start,end,connectionstyle=f'arc3,rad={curve}',
        arrowstyle='-|>',mutation_scale=12,linewidth=1.4,color=BLUE))


def draw_returns():
    fig,ax=canvas()
    ax.text(150,193,'Repeat route',ha='center',va='center',fontsize=11)
    arrow(ax,(107,135),(193,135),-.85)
    arrow(ax,(193,135),(107,135),-.85)
    ax.plot([150,150],[96,74],color=BLUE,lw=1.3)
    ax.plot([70,230],[74,74],color=BLUE,lw=1.3)
    arrow(ax,(70,74),(70,58));arrow(ax,(230,74),(230,58))
    box(ax,8,8,124,48,'Standard')
    box(ax,168,8,124,48,'Sugar\nresponse')
    return fig


def draw_histories():
    fig,ax=canvas()
    ax.text(150,193,'Different histories',ha='center',va='center',fontsize=11)
    box(ax,8,125,113,47,'A: wait')
    box(ax,8,49,113,47,'B: move')
    box(ax,175,125,117,47,'Same input')
    box(ax,175,49,117,47,'Same input')
    arrow(ax,(123,148.5),(173,148.5))
    arrow(ax,(123,72.5),(173,72.5))
    ax.text(150,16,'Compare A and B',ha='center',va='center',fontsize=10.5)
    return fig


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    args=p.parse_args(argv);output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    with plt.rc_context({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42}):
        for name,draw in [('cx_home_vector_stage03_repeated_returns',draw_returns),
                          ('cx_home_vector_stage04_history_test',draw_histories)]:
            fig=draw()
            for extension in ('png','svg','pdf'):
                fig.savefig(output/(name+'.'+extension),dpi=300,transparent=True)
            plt.close(fig)
    shutil.copyfile(__file__,output/'presentation_stage_diagrams.py')
    (output/'README.md').write_text('Conceptual schematics, not anatomical diagrams or measured results. Stage 3 shows repeated-return testing in standard and sugar-response models. Stage 4 shows different histories followed by identical future inputs in two separate simulations; the two rows must not be read as neural states or physical trajectories merging. Orange boxes and blue arrows match the presentation draft. SVG text remains editable; PNG/PDF/SVG backgrounds are transparent.\n')
    print(output.resolve())


if __name__=='__main__':main()
