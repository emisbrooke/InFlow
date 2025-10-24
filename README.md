<p align="center">
  <img src="assets/inFlow_logo.png" alt="InFlow Logo" width="350"/>
</p>

<h1 align="center">InFlow</h1>
<p align="center"><b>An information-theoretic framework for quantifying transcriptional information flow in aging</b></p>

---

InFlow models transcriptional regulation inside cells using scRNAseq data.  
It learns transcription-factor (TF)–TF and TF–target-gene (TG) couplings through an energy-based inference scheme,  
computes mutual information (MI) to quantify information decay with age,  
and performs *rejuvenation swaps* of TF distributions and gene-regulatory networks (GRNs)  
to identify the source of information loss in cellular aging.

---

### Key Features
- **Energy-based GRN inference:** learns asymmetric TF–TF and TF–TG couplings and caputures higher order interactions.  
- **Mutual-information quantification:** measures information flow from TF inputs to a TG output.  
- **Rejuvenation swaps:** tests how restoring TF distributions or GRN wiring affects MI recovery.  
- **Tissue-resolved analysis:** compare information decay and restoration across mouse tissues.  

---

### 🚀 Quick Start
```bash
git clone https://github.com/yourname/InFlow.git
