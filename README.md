---
noteId: "048efbb013c111ef9a5397c9c7cb3d4c"
tags: []

---

# FHIRspeak 

## Overview
FHIRspeak is a demonstration of a pipeline that implements a knowledge graph and vector semantic search on FHIR (Fast Healthcare Interoperability Resources) patient documents. The patient data is generated using Synthea, an open-source software that simulates patient histories. This rich, detailed data is ideal for testing healthcare software.

The ultimate goal of this pipeline is to enable AI-assisted searching through patient records. By transforming the patient data into a knowledge graph and using vector semantic search, we can make the search process more intelligent and context-aware. This can help healthcare professionals find the information they need more quickly and accurately, improving the efficiency and effectiveness of patient care.

## Requirement
Python 3.6 or higher is required to run the scripts in this repository. 

## Usage
To process the patient data, use the following command:

```bash
python flattenbundle.py --input "./Patients/*.json" --output ./Flatten_patients --subfolder

Please see license.txt.
