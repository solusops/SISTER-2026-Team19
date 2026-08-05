# SISTER Paper Repository

This repository is set up for collaborative, local LaTeX drafting of the **SISTER** research paper.

---

## 📁 Repository Structure

```
SISTER/
├── .github/workflows/
│   └── build-pdf.yml       # Automatic PDF compilation via GitHub Actions
├── .vscode/
│   └── settings.json       # Configured for VS Code + LaTeX Workshop
├── figures/                # Figures, diagrams, and TikZ sources
│   ├── overview.pdf        # System diagram vector PDF
│   └── overview.tex        # TikZ source for system diagram
├── tables/                 # Standalone table files (optional inclusion)
├── sections/               # Modular LaTeX paper sections
│   ├── 00_abstract.tex
│   ├── 01_introduction.tex
│   ├── 02_related_work.tex
│   ├── 03_methodology.tex
│   ├── 04_experiments.tex
│   ├── 05_discussion.tex
│   ├── 06_conclusion.tex
│   └── 99_appendix.tex
├── macros/                 # Custom macros and commands
│   ├── comments.tex        # Draft mode toggle & team author comments
│   └── commands.tex        # Custom math operators & shortcuts
├── build/                  # Out-of-tree build directory (git-ignored)
├── main.tex                # Root LaTeX file
├── references.bib          # BibTeX bibliography database
├── .latexmkrc              # Latexmk build pipeline configuration
├── Makefile                # Build automation (Linux / macOS / WSL)
├── build.ps1               # Build automation (Windows PowerShell)
└── README.md
```

---

## 🚀 Quick Start & Compilation

### Option 1: VS Code (Recommended)
1. Install the **LaTeX Workshop** extension in VS Code.
2. Open `main.tex`.
3. Press `Ctrl + Alt + B` (Windows/Linux) or `Cmd + Option + B` (macOS) to build.
4. Click the **View LaTeX PDF** button in the top-right toolbar.

### Option 2: Windows PowerShell
```powershell
# Standard compilation
.\build.ps1

# Continuous watch mode (recompiles on save)
.\build.ps1 -Watch

# Clean build artifacts
.\build.ps1 -Clean
```

### Option 3: Command Line / Makefile (macOS / Linux / WSL)
```bash
# Compile paper
make

# Watch mode
make watch

# Clean auxiliary build files
make clean
```

---

## 👥 Team Collaboration Guidelines

1. **One Sentence Per Line**:
   Write `.tex` source files with **one sentence per line**. This dramatically reduces Git merge conflicts when co-authors edit the same paragraph.
2. **Author Comments**:
   Use team comment macros defined in `macros/comments.tex`:
   - `\todo{Refine proof}`
   - `\authorA{Check equation 3}`
   - `\authorB{Updated numbers}`
   Toggle off all comments before submission by setting `\draftfalse` in `macros/comments.tex`.
3. **Custom Math Shortcuts**:
   Add shared notation to `macros/commands.tex` (e.g., `\R`, `\E`, `\argmax`).
4. **Vector Figures**:
   Place PDF/EPS vector graphics in `figures/` for crisp output.

---

## 🎯 Target Venue Template Integration

This repository defaults to a clean `article` class setup. If submitting to a specific venue (e.g. IEEE, ACM, NeurIPS, ICML, Springer LNCS):
1. Copy the target `.cls` file (or `IEEEtran.cls`, `acmart.cls`, etc.) into the root directory.
2. Update the `\documentclass{...}` at the top of `main.tex`.
