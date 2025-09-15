# Submission Bundle (Fake-News XAI)

This folder contains the code and light artifacts to reproduce results and run the Streamlit demo.

## Structure
~~~
submission_Bundle/
  README.md
  requirements.txt
  streamlit_app.py
  notebooks/
  scripts/
  artifacts_final/
    plots/
    metrics/
    meta.json
  data/README.md
  figures/
  LICENSE
~~~

## Quick Start
~~~bash
pip install -r requirements.txt
streamlit run streamlit_app.py
~~~

In the app sidebar, set **Artifacts folder** to the parent directory that contains `artifacts_final/`
and either `best_checkpoint/` or one `checkpoint-*/`. (This package only includes *lightweight* artifacts.)
