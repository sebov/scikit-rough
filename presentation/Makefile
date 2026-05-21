QUARTO ?= quarto
HTML_DIR ?= _output
PDF_DIR ?= _output_pdf
BACKUP_DIR ?= backup
NOTEBOOK ?= presentation.ipynb

build-all: build build-pdf

build:
	uv run quarto render $(NOTEBOOK) --to revealjs --output-dir $(HTML_DIR) --execute

build-pdf:
	uv run quarto render $(NOTEBOOK) --to beamer --output-dir $(PDF_DIR) --execute

backup:
	@mkdir -p $(BACKUP_DIR)
	@DIR=$(BACKUP_DIR)/$$(date +%Y%m%d%H%M); \
	mkdir -p $$DIR; \
	cp $(NOTEBOOK) $$DIR/$(NOTEBOOK); \
	if [ -d $(HTML_DIR) ]; then \
		cp -r $(HTML_DIR) $$DIR/$(HTML_DIR); \
	fi; \
	if [ -d $(PDF_DIR) ]; then \
		cp -r $(PDF_DIR) $$DIR/$(PDF_DIR); \
	fi; \
	echo "Backup saved to $$DIR"

clean:
	rm -rf $(HTML_DIR) $(PDF_DIR)

.PHONY: build build-pdf build-all backup clean
