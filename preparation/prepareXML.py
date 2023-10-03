import os
import csv
import xml.etree.ElementTree as ET


input_folder = '<INPUT_FOLDER>'  
output_file_path = '<OUTPUT_FOLDER>'  
gs_prefix = 'gs://<CLOUD_STORAGE_LOCATION>'

def process_xml(element, filename, img_width, img_height):
    category = element.find('name').text
    box = element.find('bndbox')
    xmin = int(box.find('xmin').text) / img_width
    ymin = int(box.find('ymin').text) / img_height
    xmax = int(box.find('xmax').text) / img_width
    ymax = int(box.find('ymax').text) / img_height

    return [filename, category, xmin, ymin, '', '', xmax, ymax, '', '']

with open(output_file_path, 'w', newline='') as output_file:
    csv_writer = csv.writer(output_file)

    for entry in os.listdir(input_folder):
        if entry.endswith('.xml'):
            input_file_path = os.path.join(input_folder, entry)
            filename = gs_prefix + os.path.splitext(entry)[0] + '.png'

            tree = ET.parse(input_file_path)
            root = tree.getroot()

            img_width = int(root.find('size/width').text)
            img_height = int(root.find('size/height').text)

            for obj in root.findall('object'):
                row = process_xml(obj, filename, img_width, img_height)
                csv_writer.writerow(row)

print("Consolidation completed successfully.")
