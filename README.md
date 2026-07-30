# Multimodal signal processing pipeline for glucose estimation
This work belongs to the Master's dissertation project conducted by Sofia P. Klautau at the Biomedical Engineering Institute at the Federal University of Santa Catarina (IEB-UFSC), Florianópolis, Brazil, under the guidance of Prof. Cesar R. Rodrigues.

This work was approved by the Ethics in Research with Humans Committee at UFSC, under the CAAE number: 84846024.9.0000.0121.

## How to run the code
1) On the Windows CMD, clone this repository. Run:
   ```python
   git clone https://github.com/sklautau/multi-sig-glc.git
   ```
2) Create a Python environment with the packages in src/requirements.txt
3) Set the Python path to this repository's src folder with:
   ```python
   set PYTHONPATH=yourpath\multi-sig-glc\src
   ```
   or
   ```python
   chdir yourpath\multi-sig-glc\src
   set PYTHONPATH=.
   ```
   Replace "yourpath" with your own path to the multi-sig-glc folder.
4) From the src folder, run the script "ieb3_run_all_pipeline.py": 
    ```python
    python run_scripts\ieb3_run_all_pipeline.py
    ```
    All the codes in the pipeline will be executed in turn.
