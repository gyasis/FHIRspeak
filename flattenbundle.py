#  %%import boto3
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from fhir.resources.bundle import Bundle
import glob
import json
import re

class FhirFlattener:
    def __init__(self, in_file_glob, flat_file_path):
        self.in_file_glob = in_file_glob
        self.flat_file_path = flat_file_path
        self.camel_pattern1 = re.compile(r'(.)([A-Z][a-z]+)')
        self.camel_pattern2 = re.compile(r'([a-z0-9])([A-Z])')

        if not os.path.exists(self.flat_file_path):
            os.mkdir(self.flat_file_path)

    def split_camel(self, text):
        new_text = self.camel_pattern1.sub(r'\1 \2', text)
        new_text = self.camel_pattern2.sub(r'\1 \2', new_text)
        return new_text

    def handle_special_attributes(self, attrib_name, value):
        if attrib_name == 'resource Type':
            return self.split_camel(value)
        return value

    def flatten_fhir(self, nested_json):
        out = {}

        def flatten(json_to_flatten, name=''):
            if type(json_to_flatten) is dict:
                for sub_attribute in json_to_flatten:
                    flatten(json_to_flatten[sub_attribute], name + self.split_camel(sub_attribute) + ' ')
            elif type(json_to_flatten) is list:
                for i, sub_json in enumerate(json_to_flatten):
                    flatten(sub_json, name + str(i) + ' ')
            else:
                attrib_name = name[:-1]
                out[attrib_name] = self.handle_special_attributes(attrib_name, json_to_flatten)

        flatten(nested_json)
        return out

    def filter_for_patient(self, entry):
        return entry['resource']['resourceType'] == "Patient"

    def find_patient(self, bundle):
        patients = list(filter(self.filter_for_patient, bundle['entry']))
        if len(patients) < 1:
            raise Exception('No Patient found in bundle!')
        else:
            patient = patients[0]['resource']

            patient_id = patient['id']
            first_name = patient['name'][0]['given'][0]
            last_name = patient['name'][0]['family']

            return {'PatientFirstName': first_name, 'PatientLastName': last_name, 'PatientID': patient_id}

    def flat_to_string(self, flat_entry):
        output = ''

        for attrib in flat_entry:
            output += f'{attrib} is {flat_entry[attrib]}. '

        return output



    def flatten_bundle(self, bundle_file_name, output_dir):
        file_name = bundle_file_name[bundle_file_name.rindex('/') + 1:bundle_file_name.rindex('.')]
        with open(bundle_file_name) as raw:
            bundle = json.load(raw)
            patient = self.find_patient(bundle)
            flat_patient = self.flatten_fhir(patient)
            progress_bar = tqdm(total=len(bundle['entry']), desc=f"Processing {file_name}")
            for i, entry in enumerate(bundle['entry']):
                flat_entry = self.flatten_fhir(entry['resource'])
                with open(f'{output_dir}/{file_name}_{i}.txt', 'w') as out_file:
                    out_file.write(f'{self.flat_to_string(flat_patient)}\n{self.flat_to_string(flat_entry)}')
                progress_bar.update(1)
            progress_bar.close()

    def flatten_all_bundles(self):
        for file in glob.glob(self.in_file_glob):
            # Extract the file name without extension
            file_name = os.path.splitext(os.path.basename(file))[0]
            # Create a new directory for the file
            file_dir = os.path.join(self.flat_file_path, f'Flatten_{file_name}')
            os.makedirs(file_dir, exist_ok=True)
            # Flatten the bundle and store the result in the file's directory
            self.flatten_bundle(file, file_dir)
            
            
            
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Flatten FHIR bundles.')
    parser.add_argument('-i', '--input', required=True, help='Input file glob pattern. This should point to the FHIR bundles you want to flatten.')
    parser.add_argument('-o', '--output', required=True, help='Output directory where the flattened files will be stored.')
    parser.add_argument('-s', '--subfolder', action='store_true', help='Create a subfolder for each file.')

    args = parser.parse_args()

    flattener = FhirFlattener(args.input, args.output)
    if args.subfolder:
        flattener.flatten_all_bundles()
    else:
        for file in glob.glob(flattener.in_file_glob):
            flattener.flatten_bundle(file, flattener.flat_file_path)

    # in_file_glob = filepath
    # flat_file_path = outputpath
    # flattener = FhirFlattener(in_file_glob, flat_file_path)
    # flattener.flatten_all_bundles()
# %%
