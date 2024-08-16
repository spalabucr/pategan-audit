from pate_gans.original import PG_ORIGINAL, PG_ORIGINAL_AUDIT
from pate_gans.updated import PG_UPDATED, PG_UPDATED_AUDIT
from pate_gans.synthcity import PG_SYNTHCITY, PG_SYNTHCITY_AUDIT, PG_SYNTHCITY_FIX
from pate_gans.turing import PG_TURING, PG_TURING_AUDIT
from pate_gans.borealisai import PG_BORAI, PG_BORAI_AUDIT
from pate_gans.smartnoise import PG_SMARTNOISE, PG_SMARTNOISE_AUDIT


PATE_GANS = {pg.__name__: pg for pg in [PG_ORIGINAL, PG_UPDATED, PG_SYNTHCITY, PG_TURING, PG_BORAI, PG_SMARTNOISE]}
PATE_GANS_AUDIT = {pg.__name__: pg for pg in [PG_ORIGINAL_AUDIT, PG_UPDATED_AUDIT, PG_SYNTHCITY_AUDIT, PG_TURING_AUDIT, PG_BORAI_AUDIT, PG_SMARTNOISE_AUDIT]}
PATE_GANS_FIX = {pg.__name__: pg for pg in [PG_SYNTHCITY_FIX]}
