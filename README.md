# TILLING-by-Sequencing⁺ — Automated Pipeline

**Based on:** Lakhssassi et al. (2020), *Journal of Experimental Botany*, 71(22): 6969–6987  
**Lab:** Dept. of Plant, Soil and Agricultural Systems, Southern Illinois University Carbondale

---

## What Was Provided vs. What Is Needed

### ✅ PROVIDED (pipeline blueprint — sufficient to write the scripts)

| File | Contents |
|------|----------|
| `TILLING_Pipeline_Guide.docx` | Full 8-step pipeline architecture, commands, parameters, directory layout, troubleshooting |
| `TILLING_by_Sequencing_Paper.pdf` | Scientific methodology, filtering thresholds (QUAL>700, AC>4), pooling scheme (96/well), 148 probes, EMS signature (87.7% G>A/C>T) |

### ❌ MISSING (required to actually RUN the pipeline)

| # | Data Item | Description | Where to Get It |
|---|-----------|-------------|-----------------|
| 1 | **Raw FASTQ files** | Paired-end Illumina HiSeqX 2×150bp reads from the 4,000+ EMS-mutagenized M2 pooled DNA libraries. File pairs like `P001_WH07_R1_001.fastq.gz` / `P001_WH07_R2_001.fastq.gz`. These are the actual sequencing data. | From the sequencing facility (Rapid Genomics LLC, Gainesville, FL) or from the lab's data storage. |
| 2 | **Reference genome FASTA** | `Gmax_275_v2.0.fa` — Glycine max Williams 82 assembly v2.0 (Wm82.a2.v1). ~1 GB file. | Download from [Phytozome](https://phytozome.jgi.doe.gov) (requires free JGI account). |
| 3 | **Target regions BED file** | `target_regions.bed` — Genomic coordinates of the 148 capture probes covering GmSACPD-A (41 probes), -B (32), -C (41), -D (34). BED format: `chr\tstart\tend`. Must match the genome assembly version. | From Supplementary Table S2 of the paper, or from Rapid Genomics who designed the probes. |
| 4 | **CRISP binary** | Pool-aware variant caller. Not publicly downloadable — must be requested from the Bansal Lab at UCSD. | Request from: https://bansal-lab.github.io/software/crisp.html (pipeline works without it using FreeBayes only, but both are recommended). |
| 5 | **Sample list / pooling scheme** | The exact sample names and which plate/well each belongs to. The guide mentions names like `P001_WH07`, `P003_WC06`. Needed to verify the pipeline discovers all samples correctly. | From the lab's DNA library records (42 plates × 96 wells). |

### Summary

The two provided files give us **complete instructions** to build the automation scripts. The scripts (`install_tools.sh` + `tilling_pipeline.sh`) are ready to use. But to actually **execute** the pipeline and produce variant calls, the 5 data items above must be placed in the correct directories.

---

## Quick Start

```bash
# 1. Make scripts executable
chmod +x install_tools.sh tilling_pipeline.sh

# 2. Install all bioinformatics tools (run once)
bash install_tools.sh

# 3. Set up your project directory with the MISSING DATA
mkdir -p FASTAQ indexed_genome_mem2

# Place your FASTQ files in FASTAQ/
# Place Gmax_275_v2.0.fa in indexed_genome_mem2/
# Place target_regions.bed in the project root
# Place CRISP binary in the project root

# 4. Edit the CONFIG section in tilling_pipeline.sh
nano tilling_pipeline.sh

# 5. Run the pipeline
bash tilling_pipeline.sh

# For long jobs, run in background:
nohup bash tilling_pipeline.sh > run.log 2>&1 &
tail -f LOGS/pipeline.log
```

---

## Project Directory Structure

```
your_project/
├── tilling_pipeline.sh           ← Main pipeline (8 automated steps)
├── install_tools.sh              ← One-time tool installer
├── README.md                     ← This file
│
├── FASTAQ/                       ← ❌ MISSING: Raw FASTQ files go here
│   ├── P001_WH07_R1_001.fastq.gz
│   ├── P001_WH07_R2_001.fastq.gz
│   ├── P003_WC06_R1_001.fastq.gz
│   └── P003_WC06_R2_001.fastq.gz
│
├── indexed_genome_mem2/          ← ❌ MISSING: Reference genome
│   └── Gmax_275_v2.0.fa
│
├── target_regions.bed            ← ❌ MISSING: Probe coordinates (BED)
├── bwa-mem2-2.0pre2_x64-linux/   ← ✅ Installed by install_tools.sh
└── CRISP                         ← ❌ MISSING: Manual download from Bansal Lab
```

---

## Pipeline Steps

| Step | Tool | What It Does |
|------|------|-------------|
| 1 | fastp | Trim adapters, remove low-quality bases (Q20), discard reads <36bp |
| 2 | BWA-MEM2 | Index reference genome (run once, ~10 GB RAM, ~15 min) |
| 3 | BWA-MEM2 | Align trimmed reads to soybean reference genome |
| 4 | SAMtools | SAM → BAM conversion, coordinate sorting, BAM indexing |
| 5 | samtools faidx | Index reference FASTA for CRISP (run once) |
| 6 | CRISP + FreeBayes | Dual variant calling — pool-aware + haplotype-based |
| 7 | VCFtools | Filter variants (QUAL>700, AC>4), find concordant calls |
| 8 | — | Summary report with EMS mutation signature validation |

---

## Key Pipeline Features

- **Fault-tolerant & resumable**: Each step checks for existing outputs. If the pipeline crashes, just re-run — it skips completed steps.
- **Auto-discovers samples**: Detects paired-end FASTQ naming patterns automatically.
- **Dual variant calling**: CRISP (pool-aware) + FreeBayes (haplotype-based), with concordance analysis to minimize false positives.
- **EMS signature validation**: Automatically calculates G>A / C>T ratio (expected ~87.7%).
- **Heavily commented**: Every section of code explains what it does and why.

---

## Citation

> Lakhssassi N, Zhou Z, Liu S, et al. (2020). Soybean TILLING-by-Sequencing⁺ reveals the role of novel GmSACPD members in unsaturated fatty acid biosynthesis while maintaining healthy nodules. *Journal of Experimental Botany*, 71(22), 6969–6987. https://doi.org/10.1093/jxb/eraa402
