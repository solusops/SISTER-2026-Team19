# ==============================================================================
# Latexmk Configuration File
# ==============================================================================

# Output directory for auxiliary files and target PDF
$out_dir = 'build';

# Build PDF using pdflatex (1 = pdflatex, 2 = postscript, 3 = dvi, 4 = lualatex, 5 = xelatex)
$pdf_mode = 1;

# Enable SyncTeX for forward/backward search with PDF viewers (VS Code, Skim, Sumatrapdf)
$pdflatex = 'pdflatex -interaction=nonstopmode -synctex=1 %O %S';

# Automatically run bibtex when bibliography changes
$bibtex_use = 2;

# Clean up extensions when running `latexmk -c` or `latexmk -C`
$clean_ext = 'aux bbl blg fdb_latexmk fls log out synctex.gz toc lof lot acn glo ist xml bcf nav snm vrb';
