QUARTO ?= quarto
OUTPUT_DIR ?= _output
BACKUP_DIR ?= backup
NOTEBOOK ?= presentation.ipynb

build:
	uv run quarto render $(NOTEBOOK) --to revealjs --output-dir $(OUTPUT_DIR) --execute

backup:
	@mkdir -p $(BACKUP_DIR)
	@DIR=$(BACKUP_DIR)/$$(date +%Y%m%d%H%M); \
	mkdir -p $$DIR; \
	cp $(NOTEBOOK) $$DIR/$(NOTEBOOK); \
	if [ -d $(OUTPUT_DIR) ]; then \
		cp -r $(OUTPUT_DIR) $$DIR/$(OUTPUT_DIR); \
	fi; \
	echo "Backup saved to $$DIR"

clean:
	rm -rf $(OUTPUT_DIR)

.PHONY: build backup clean
