from pathlib import Path

import fitz

from mozilla_sec_eia.utils.pdf import combine_doc_pages, render_page

pdf_filename = "/Users/katielamb/CatalystCoop/mozilla-sec-eia/sec10k_filings/pdfs/922358-2006q4-922358-0000950134-06-018966.pdf"
src_path = Path(pdf_filename)
doc = fitz.Document(str(src_path))
pg = combine_doc_pages(doc)
full_pg_img = render_page(pg)
