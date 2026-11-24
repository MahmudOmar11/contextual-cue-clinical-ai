"""Source data for the figures: one workbook per figure, one sheet per panel, holding the plotted values.

Workbooks are written with fixed internal dates, so that repeated runs give identical files.
"""
import io
import re
import zipfile

import pandas as pd

from common import ROOT

SOURCE_DATA = ROOT / "results" / "source_data"
FIXED_TIME = (2026, 1, 1, 0, 0, 0)
FIXED_STAMP = "2026-01-01T00:00:00Z"


def save_pdf(fig, path, **kw):
    """Vector copy of a figure, without a creation date."""
    fig.savefig(path, format="pdf", metadata={"CreationDate": None, "ModDate": None}, **kw)


def write(name, sheets):
    """Write {sheet name: DataFrame} to results/source_data/<name>."""
    SOURCE_DATA.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for sheet, frame in sheets.items():
            frame.to_excel(xw, sheet_name=sheet, index=False)
    src = zipfile.ZipFile(io.BytesIO(buf.getvalue()))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for info in src.infolist():
            data = src.read(info.filename)
            if info.filename == "docProps/core.xml":
                data = re.sub(rb"(<dcterms:(?:created|modified)[^>]*>)[^<]*", rb"\g<1>" + FIXED_STAMP.encode(), data)
            member = zipfile.ZipInfo(info.filename, date_time=FIXED_TIME)
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = info.external_attr
            z.writestr(member, data)
    (SOURCE_DATA / name).write_bytes(out.getvalue())
