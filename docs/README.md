# GeoINR documentation

## Build

```bash
python -m pip install -r docs/requirements.txt
python -m sphinx -b html docs/source docs/build/html
```

Open `docs/build/html/index.html` in your browser.

## Notes

- API pages are generated automatically from `geoinr_faults/`.
- Notebook examples are maintained in the repository `examples/` directory and
  linked from the docs examples gallery.
