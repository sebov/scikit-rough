# Create PDF files from notebooks

## install nb-pdf-template

For better PDF look (no filename title, no date) use jinja templates from
https://pypi.org/project/nb-pdf-template/ package.


- copy template files from
  `.venv/lib/python3.9/site-packages/nb_pdf_template/templates/*.tplx`
  to `.venv/share/jupyter/nbconvert/templates/latex/`
- change extensions from `.tplx` to `.tex.j2`
- in `classic.tex.j2` fix extension of the file referenced in `set cell_style = ...` to
  `.tex.j2`
- in `classicm.tex.j2` fix extension of the file referenced in the `extends` section to
  `.tex.j2`

To generate PDF file for a given jupyter notebook:

```bash
jupyter nbconvert feature_importance.ipynb --to pdf --execute --template-file classic
```

To generate PDF (webpdf) file for a given jupyter notebook:

```bash
jupyter nbconvert feature_importance.ipynb --to webpdf --execute --allow-chromium-download
```
