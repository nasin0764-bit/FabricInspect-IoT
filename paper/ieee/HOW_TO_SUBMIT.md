# How to submit to IEEE Internet of Things Journal (free option)

**Cost: $0** if you follow this guide.
- Choose **Traditional** submission, **not** Open Access. Open Access costs US$2,645.
- Keep the paper at **8 pages or less**. Every page beyond 8 costs $175 and is mandatory. The PDF is exactly 8 pages now.
- Also keep it at 8 pages after the reviewers' changes.

Source: IEEE IoT Journal Author Guidelines (April 2025), https://ieee-iotj.org/guidelines-for-authors

## Files to upload (all in this folder)
- `main.pdf`: the manuscript
- `cover_letter.txt`: paste it into the cover-letter box
- If the LaTeX source is requested: zip this whole folder (`main.tex`, `sec*.tex`, `tables/`, `figures/`, `refs.bib`)

## Before you start
1. **All five authors must agree to submit.** The system emails each co-author. Send them `main.pdf` first.
2. **Read the whole PDF yourself.** By submitting, the authors take responsibility for every sentence.
3. **Decide who submits.** It doesn't have to be the corresponding author (Petr Butovskiy); any author can submit and enter him as the corresponding author.

## Step by step
1. Open **https://ieee.atyponrex.com/journal/iot** (IEEE Author Portal).
2. Log in or create a free IEEE account with your e-mail. Add your ORCID if you have one; a free one is available at orcid.org.
3. Click **Submit manuscript** and choose article type **Regular Paper**.
4. **Title:** Few-Shot, Label-Free Fabric Defect Inspection for Networked Seaming Stations via Confidence-Gated Edge Offloading
5. **Abstract:** copy it from `main.tex`, between `\begin{abstract}` and `\end{abstract}`. Replace `\%` with `%`, `\,` with nothing, and `MVTec~AD` with `MVTec AD`.
6. **Keywords:** Industrial Internet of Things (IIoT); edge computing; computation offloading; MQTT; anomaly detection; fabric defect detection; few-shot learning
7. **Authors**, in this order, with affiliations and e-mails exactly as in the PDF:
   1. Nasiba Palvannazirova, Beijing University of Posts and Telecommunications (BUPT), nasiba@bupt.edu.cn
   2. Petr Butovskiy, Tashkent Institute of Textile and Light Industry, gtnz1071980@gmail.com (**corresponding author**)
   3. Jakhongir Mirkhoshimov, Illinois Institute of Technology, School of Computing, jmirkhoshimov@hawk.illinoistech.edu
   4. Ayaz Waseem, BUPT, 2024090002@bupt.cn
   5. Nikita Lobasevich, BUPT, lnikbest@gmail.com
   6. Miralam Allamov, Tashkent Institute of Textile and Light Industry, miralam3843678@gmail.com
   7. Robiya Tuguzbaeva, Tashkent Institute of Textile and Light Industry, trobiya89@mail.ru
   8. Firuza Dustova, **[affiliation missing]**, firuzadustovahamraevna@mail.ru
   9. M. M. Agzamov, Tashkent Institute of Textile and Light Industry, agzamovmirhosil@gmail.com
8. **Open access question:** answer **No / Traditional**.
9. **Suggested reviewers (optional but recommended):** 3–5 researchers in fabric defect detection or edge AI who are not at your institutions and have not worked with you. You can take the corresponding authors of the recent fabric-defect or edge-AI papers in the reference list; their e-mails are printed on their papers (the PDFs are in `Desktop\111`).
10. **Cover letter:** paste `cover_letter.txt`.
11. **Upload** `main.pdf` as the main document.
12. **Declarations:** confirm originality (not submitted elsewhere), no conflicts of interest, and the data and code availability statement. The data is the public MVTec AD set; the code is at https://github.com/nasin0764-bit/FabricInspect-IoT.
13. **Submit.** You'll get a confirmation e-mail with a manuscript ID.

## After submission
- First decision usually takes several weeks to a few months.
- **"Reject without review"** is common at this journal. If it happens, the cheapest next options are:
  - **IEEE Sensors Journal**: hybrid, so free with a traditional submission; check its page limit.
  - **The Journal of The Textile Institute** or **Arabian Journal for Science and Engineering**: hybrid journals, free with the subscription (non-open-access) option. Check the current rules before submitting.
- **Avoid** journals that always charge authors (IEEE Access, MDPI *Sensors*/*Applied Sciences*/*Electronics*, about $2,000 or more).
- **Free preprint (optional):** post the PDF on arXiv (cs.CV or eess.SY) so the work is public right away. IEEE allows this. You need an arXiv account; first-time posters may need an endorsement.
