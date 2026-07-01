import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
FIG = Path("../reports/figures"); FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi":120,"savefig.dpi":300,"font.size":11})
INK="#6f6f6f"; BLUE="#1a4f8a"

models=["TrOCR\n(no augmentation)","Augmented\nTrOCR"]
em=[0.569,0.655]; cer=[0.216,0.176]
fig,(a1,a2)=plt.subplots(1,2,figsize=(10,4.6))

b1=a1.bar(models,em,color=[INK,BLUE],edgecolor="black",width=0.6)
for b,v in zip(b1,em): a1.text(b.get_x()+b.get_width()/2,v+0.012,f"{v:.3f}",ha="center",fontsize=11,weight="bold")
a1.set_ylabel("Exact-Match Accuracy"); a1.set_ylim(0,0.78); a1.set_title("(a) Exact match (+8.6 points)")
a1.spines[["top","right"]].set_visible(False)

b2=a2.bar(models,cer,color=[INK,BLUE],edgecolor="black",width=0.6)
for b,v in zip(b2,cer): a2.text(b.get_x()+b.get_width()/2,v+0.006,f"{v:.3f}",ha="center",fontsize=11,weight="bold")
a2.set_ylabel("Character Error Rate (lower = better)"); a2.set_ylim(0,0.26); a2.set_title("(b) Character error rate")
a2.spines[["top","right"]].set_visible(False)

fig.suptitle("Effect of data augmentation on TrOCR recognition",fontsize=12,weight="bold")
plt.tight_layout(); plt.savefig(FIG/"final_fig2_augmentation.png"); plt.show()
print("saved final_fig2_augmentation.png")
