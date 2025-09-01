# Makefile for IEEE Conference Paper
# Targets: CVPR, ICCV, ECCV, NeurIPS, ICML

.PHONY: all clean cvpr iccv eccv neurips icml

# Default target
all: main.pdf

# Main compilation
main.pdf: main.tex *.png *.csv *.json
	pdflatex main.tex
	bibtex main
	pdflatex main.tex
	pdflatex main.tex

# Conference-specific targets
cvpr: main.pdf
	@echo "Paper compiled for CVPR submission"
	@echo "File: main.pdf"

iccv: main.pdf
	@echo "Paper compiled for ICCV submission"
	@echo "File: main.pdf"

eccv: main.pdf
	@echo "Paper compiled for ECCV submission"
	@echo "File: main.pdf"

neurips: main.pdf
	@echo "Paper compiled for NeurIPS submission"
	@echo "File: main.pdf"

icml: main.pdf
	@echo "Paper compiled for ICML submission"
	@echo "File: main.pdf"

# Clean auxiliary files
clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.fdb_latexmk *.fls *.synctex.gz

# Clean everything including PDF
cleanall: clean
	rm -f *.pdf

# View the PDF (platform-specific)
view: main.pdf
	@if command -v xdg-open > /dev/null; then \
		xdg-open main.pdf; \
	elif command -v open > /dev/null; then \
		open main.pdf; \
	else \
		echo "Please open main.pdf manually"; \
	fi

# Help
help:
	@echo "Available targets:"
	@echo "  all      - Compile the paper (default)"
	@echo "  cvpr      - Compile for CVPR submission"
	@echo "  iccv      - Compile for ICCV submission"
	@echo "  eccv      - Compile for ECCV submission"
	@echo "  neurips   - Compile for NeurIPS submission"
	@echo "  icml      - Compile for ICML submission"
	@echo "  clean     - Remove auxiliary files"
	@echo "  cleanall  - Remove all generated files"
	@echo "  view      - Open the PDF"
	@echo "  help      - Show this help"
