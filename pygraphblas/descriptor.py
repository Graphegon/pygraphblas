from .base import lib

ooco = lib.LAGraph_desc_ooco # compl mask
ooor = lib.LAGraph_desc_ooor # replace
otoo = lib.LAGraph_desc_otoo # transpose A (A')
tocr = lib.LAGraph_desc_tocr # A', compl mask, replace
ttco = lib.LAGraph_desc_ttco # A', B', compl mask
ttor = lib.LAGraph_desc_ttor # A', B', replace
oocr = lib.LAGraph_desc_oocr # compl mask, replace
otco = lib.LAGraph_desc_otco # B', compl mask
otor = lib.LAGraph_desc_otor # B', replace
tooo = lib.LAGraph_desc_tooo # A'
ttcr = lib.LAGraph_desc_ttcr # A', B', compl mask, replace
oooo = lib.LAGraph_desc_oooo # default (NULL)
otcr = lib.LAGraph_desc_otcr # B' compl mask, replace
toco = lib.LAGraph_desc_toco # A', compl mask
toor = lib.LAGraph_desc_toor # A', replace
ttoo = lib.LAGraph_desc_ttoo # A', B'

T_A = (tocr, ttco, ttor, tooo, ttcr, toco, toor, ttoo)
T_B = (otoo, ttco, ttor, otco, otor, ttcr, otcr, ttoo)
C_M = (ooco, tocr, ttco, oocr, otco, ttcr, otcr, toco)
R_V = (ooor, tocr, ttor, oocr, otor, ttcr, otcr, toor)
