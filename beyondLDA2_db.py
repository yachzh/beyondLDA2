"""
ASE Database Integration for beyondLDA2
========================================
Stores calculation results in an ase.db database for query and analysis.

Each call to a ``lda_plus_u.get_*`` method inserts one row containing:
- **Atoms object** — enables structure round-trip via ``row.toatoms()``
- **Key-value pairs** (queryable): ``formula``, ``method``, ``xc``,
  ``spin_state``, ``hubbard_u``, ``label``, ``result_value``,
  ``spin_pol``, ``planewave``, ``created_utc``, ``creator``
- **``data`` blob** (non-queryable but restorable): input parameters,
  result breakdowns (e.g. E_ks, D_xc for GLLBSC), walltime, magnetic
  moment, GPAW version

Usage — storage::

    from beyondLDA2_db import LDAPlusUDatabase

    db = LDAPlusUDatabase("results.db")
    dft = lda_plus_u(atoms=mol, database=db, ...)
    energy = dft.get_electronic_energy()   # auto-stored
    gap = dft.get_ksgap()                  # auto-stored

Usage — analysis::

    db = LDAPlusUDatabase("results.db")
    for row in db.select(formula="Fe2O3", method="electronic_energy"):
        atoms = row.toatoms()
        print(row.result_value, row.spin_state, row.hubbard_u)

    # Or use raw ase.db queries for advanced filtering
    for row in db.db.select("formula=Fe2O3", "method=electronic_energy"):
        print(row.id, row.result_value)
"""

from pathlib import Path
import getpass
from datetime import datetime, timezone
import ase.db
from ase.db.core import Database


class LDAPlusUDatabase:
    """Wrapper around ase.db for storing and querying beyondLDA2 results.

    Parameters
    ----------
    filename : str or Path
        Database file path (e.g. ``"results.db"``). If the file exists
        it is opened; otherwise a new database is created.
    """

    def __init__(self, filename):
        self.filename = Path(filename).resolve()
        self._db = ase.db.connect(str(self.filename))

    @property
    def db(self) -> Database:
        """Access the underlying ``ase.db.core.Database`` for advanced queries."""
        return self._db

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------
    def store(self, atoms, method, result_value, *, label=None,
              xc=None, spin_state=None, hubbard_u=None,
              spin_pol=None, planewave=None,
              walltime=None, input_args=None, **extra_kvp):
        """Insert one calculation row.

        Parameters
        ----------
        atoms : ASE Atoms object
            The structure the calculation was performed on.
        method : str
            Calculation type, e.g. ``"electronic_energy"``, ``"ks_gap"``,
            ``"gllbsc_gap"``, ``"g0w0_gap"``, ``"local_opt"``.
        result_value : float
            Primary scalar result (energy in eV, gap in eV, etc.).
        label : str, optional
            User-defined label for grouping related calculations.
        xc, spin_state, hubbard_u, spin_pol, planewave : optional
            Calculation settings stored as queryable key-value pairs.
        walltime : float, optional
            Wall-clock time in seconds (stored in ``data``).
        input_args : dict, optional
            Full input parameter dict (stored in ``data``, not queryable).
        **extra_kvp
            Additional key-value pairs stored as queryable fields.
            Avoid keys reserved by ase.db (see ``_RESERVED_KEYS``).

        Notes
        -----
        Two fields are **automatically** added to every row:

        * ``created_utc`` — ISO 8601 UTC timestamp of when the row was written,
          e.g. ``"2026-06-07T12:34:56Z"``.
        * ``creator`` — system username (via ``getpass.getuser()``) of the
          account that ran the calculation.

        Returns
        -------
        int
            The ``id`` of the inserted row.
        """
        # Build key-value pairs (queryable)
        kvp = dict(extra_kvp)
        kvp['method'] = method
        kvp['result_value'] = result_value
        kvp['created_utc'] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        kvp['creator'] = getpass.getuser()
        if label is not None:
            kvp['label'] = label
        if xc is not None:
            kvp['xc'] = xc
        if spin_state is not None:
            kvp['spin_state'] = spin_state
        if hubbard_u is not None:
            kvp['hubbard_u'] = hubbard_u
        if spin_pol is not None:
            kvp['spin_pol'] = int(spin_pol)
        if planewave is not None:
            kvp['planewave'] = int(planewave)

        # Build data blob (non-queryable)
        data = {}
        if walltime is not None:
            data['walltime'] = walltime
        if input_args is not None:
            # Filter out non-serializable objects from input_args
            data['input_args'] = _filter_args(input_args)

        # Insert into database
        row_id = self._db.write(atoms, data=data, **kvp)
        return row_id

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def select(self, **filters) -> list:
        """Return all rows matching the given filters.

        Each row has ``.key_value_pairs`` for the queryable fields,
        ``.data`` for the blob, and ``.toatoms()`` for the structure.

        Parameters
        ----------
        **filters
            Query filters, e.g. ``formula="Fe2O3"``,
            ``method="electronic_energy"``, ``xc="LDA"``.

        Returns
        -------
        list of ``ase.db.row.Row``
        """
        return list(self._db.select(**filters))

    def get(self, row_id: int):
        """Fetch a single row by its database id.

        Returns
        -------
        ``ase.db.row.Row`` or ``None``
        """
        try:
            return self._db.get(id=row_id)
        except (KeyError, IndexError):
            return None

    def get_ids(self, **filters) -> list:
        """Return database ids matching the given filters (useful for
        aggregated analysis)."""
        return [row.id for row in self._db.select(**filters)]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def count(self, **filters) -> int:
        """Number of rows matching the given filters."""
        return self._db.count(**filters)

    @property
    def nrows(self) -> int:
        """Total number of rows in the database."""
        return self._db.count()

    def __repr__(self):
        return f"<LDAPlusUDatabase {self.filename} ({self.nrows} rows)>"


def _filter_args(args):
    """Remove non-serializable entries from a parameters dict.

    Strips out ASE Atoms objects, numpy arrays of magnetic moments,
    and other objects that cannot be JSON-serialized."""
    if args is None:
        return None
    filtered = {}
    for k, v in args.items():
        if isinstance(v, (str, int, float, bool)):
            filtered[k] = v
        elif v is None:
            filtered[k] = None
        elif isinstance(v, (list, tuple)):
            # Convert small numeric sequences; skip large/atomic arrays
            filtered[k] = str(v)
        elif hasattr(v, '__module__') and 'ase' in str(getattr(v, '__module__', '')):
            # Skip ASE objects (atoms, calculators, etc.)
            filtered[k] = f"<{type(v).__name__}>"
        else:
            try:
                # Test JSON-serializability
                import json
                json.dumps(v)
                filtered[k] = v
            except (TypeError, OverflowError):
                filtered[k] = str(v)
    return filtered



