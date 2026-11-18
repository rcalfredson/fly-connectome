"""Curated cell-type evidence matrix for presentation slide 9."""

import argparse
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROWS=[
    ('PFNd/PFNv','Heading + body-centered\ntranslation [1]',
     'Simplified tuning prescribed;\nnot a learned result.'),
    ('hDeltaB','World-centered velocity /\ntravel direction [1, 2]',
     'Coordinate-transform foundation;\nlatest recurrent tuning untested.'),
    ('vDeltaE','Perturbations affect hDeltaG\nactivity [3]',
     'Recurrent interactions learned;\nperturbation response untested.'),
    ('hDeltaG','Sugar suppression;\nrecent-motion signals [3]',
     'Sugar effect approximated;\nfull activity profile unvalidated.'),
    ('hDeltaC','Odor-gated wind-direction\ntuning [4]',
     'No odor/wind inputs;\nmeasured task not tested.')]
SOURCES={
    '1':dict(citation='Lu et al., Nature 2022',url='https://www.nature.com/articles/s41586-021-04191-x'),
    '2':dict(citation='Lyu et al., Nature 2022',url='https://www.nature.com/articles/s41586-021-04067-0'),
    '3':dict(citation='Janke, thesis, repository 2025; PDF title page September 2024',url='https://digitalcommons.rockefeller.edu/student_theses_and_dissertations/808/'),
    '4':dict(citation='Matheson et al., Nature Communications 2022',url='https://www.nature.com/articles/s41467-022-32247-7')}
STEM='cx_home_vector_cell_type_biological_evidence_slide09'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    output=Path(p.parse_args(argv).output);output.mkdir(parents=True,exist_ok=False)
    with plt.rc_context({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42}):
        fig,ax=plt.subplots(figsize=(11.6,4.8))
        fig.subplots_adjust(left=.01,right=.99,bottom=.01,top=.99)
        ax.set(xlim=(0,1130),ylim=(0,470));ax.set_axis_off()
        for x,label in [(15,'Cell type'),(195,'Evidence from flies'),(610,'Status in this model')]:
            ax.text(x,444,label,ha='left',va='center',fontsize=16,weight='bold',color='#086395')
        ax.plot([5,1120],[417,417],color='#086395',lw=1.3)
        for index,(cell,evidence,status) in enumerate(ROWS):
            y=374-index*81
            ax.add_patch(FancyBboxPatch((6,y-24),162,48,boxstyle='round,pad=0,rounding_size=8',
                edgecolor='#ff6b29',facecolor='white',linewidth=1.2))
            ax.text(87,y,cell,ha='center',va='center',fontsize=14,weight='bold')
            ax.text(195,y,evidence,ha='left',va='center',fontsize=14,linespacing=1.35)
            ax.text(610,y,status,ha='left',va='center',fontsize=14,linespacing=1.35)
            if index<4:ax.plot([185,1120],[y-40,y-40],color='#e1e5eb',lw=.8)
        for extension in ('png','svg','pdf'):fig.savefig(output/(STEM+'.'+extension),dpi=300,transparent=True)
        plt.close(fig)
    with (output/'table_content.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['cell_type','evidence_from_flies','model_status'])
        writer.writerows((cell,e.replace('\n',' '),s.replace('\n',' ')) for cell,e,s in ROWS)
    (output/'sources.json').write_text(json.dumps(SOURCES,indent=2)+'\n')
    shutil.copyfile(__file__,output/'slide_biology.py')
    (output/'notes.md').write_text('Conceptual evidence comparison, not a physiological fit or an exhaustive cell-type review. Use a slide title and takeaway outside this graphic. Source [3] is a thesis; the experimental observations and the thesis formal integration model are distinct from our sugar-gate approximation. The hDeltaB coordinate-transform foundation does not validate the latest learned recurrent activity. The vDeltaE wording refers to experimental perturbation effects; exact integration dynamics are an interpretation requiring additional tests. hDeltaC evidence concerns odor/wind tasks absent from this model, not a claim that hDeltaC cannot contribute to other functions. No comparison in this table demonstrates that arbitrary model activity units match calcium or spike-rate measurements.\n')
    print(output.resolve())


if __name__=='__main__':main()
