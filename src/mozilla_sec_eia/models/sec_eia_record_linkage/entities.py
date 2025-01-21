"""Define schemas for tables using pandera."""

import pandera as pa
from pandera.typing import Series


class CoreSec10kFilers(pa.DataFrameModel):
    """."""

    _id: Series[str] = pa.Field(alias="id", description="ID of extracted filing.")
    subsidiary: Series[str] = pa.Field(description="Name of subsidiary company.")
    loc: Series[str] = pa.Field(
        description="Location of subsidiary company.", nullable=True
    )
    #: Use str to avoid conversion errors
    own_per: Series[str] = pa.Field(
        description="Percent ownership of subsidiary company.",
        nullable=True,
        coerce=True,
    )
