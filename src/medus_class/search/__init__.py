"""The chromosome, its decoder, and the NSGA-II / MicroGA search.

A candidate class-unlearning strategy is an integer vector of length ``L``, the
number of layer groups (``L = 6`` for ResNet-18)::

    x = (b, g, s, d_g, d_s)

``b`` activates a group, ``g``/``s`` choose an editor and a smoother operator for
it, and ``d_g``/``d_s`` set their ordinal intensities (0 = OFF, 1..5). The search
is therefore over *which edit to apply to which block, how hard* -- not over a
single method applied uniformly.
"""

from medus_class.search.decoder import Action, DecodedStrategy, decode
from medus_class.search.genome import GENE_NAMES, Chromosome, ChromosomeBounds

__all__ = [
    "Action", "DecodedStrategy", "decode",
    "GENE_NAMES", "Chromosome", "ChromosomeBounds",
]
