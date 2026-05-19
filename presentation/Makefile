QUARTO ?= quarto
OUTPUT_DIR ?= _output
BACKUP_DIR ?= backup
NOTEBOOK ?= presentation.ipynb

build:
	$(QUARTO) render $(NOTEBOOK) --to revealjs --output-dir $(OUTPUT_DIR)

backup:
	@mkdir -p $(BACKUP_DIR)
	@cp $(NOTEBOOK) $(BACKUP_DIR)/$$(date +%Y%m%d%H%M)_$(NOTEBOOK)

clean:
	rm -rf $(OUTPUT_DIR)

.PHONY: build backup clean
