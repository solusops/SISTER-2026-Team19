# NewInML 2026 submission bundle

This bundle formats the supplied paper with the official NeurIPS 2026 workshop style using:

```latex
\usepackage[dblblindworkshop]{neurips_2026}
\workshoptitle{New in Machine Learning (NewInML) at NeurIPS 2026}
```

Build with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The compiled paper is anonymous. Research content occupies pages 1--6, with references beginning on page 6. Supplementary material is on pages 8--9 and the mandatory NeurIPS checklist follows on pages 10--16. The main paper is therefore within the workshop's 2--8 page limit excluding references.

## Items the authors must resolve before submission

1. Add an anonymized code/data repository or supplemental archive. Public links and self-citations from the original manuscript were omitted because they reveal author identity.
2. Confirm NeurIPS Code of Ethics compliance.
3. Add the human-study participant instructions, compensation, and IRB/exemption/equivalent determination as applicable.
4. Add compute hardware, memory, run time, and total-compute information.
5. Add licenses/terms for the six evaluated model assets.
6. Decide whether to add a dedicated broader-impact statement.
7. Recheck every checklist answer after supplying the information above.

Do not change `neurips_2026.sty`; it is copied unchanged from the provided formatting package.
