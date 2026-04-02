# GeoINR documentation

## Build

```bash
python -m pip install -r docs/requirements.txt
python -m sphinx -b html docs/source docs/build/html
```

Open `docs/build/html/index.html` in your browser.

## Notes

- API pages are generated automatically from `geoinr_faults/`.
- `test.ipynb` and `horizontal_stratigraphy.ipynb` are copied from repo root
  to `docs/source/examples/` at build time.
