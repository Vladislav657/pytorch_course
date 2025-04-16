import json


with open("instances_default.json") as f:
    instances_default = json.load(f)

categories = {1: 'smoke', 2: 'fire'}
images = {}

for image in instances_default['images']:
    images[image['id']] = image['file_name']

image_annotations = {}

for annotation in instances_default['annotations']:
    image = images[annotation['image_id']]
    if image not in image_annotations:
        image_annotations[image] = []
    if annotation['category_id'] - 3 in categories.keys():
        image_annotations[image].append({'category': annotation['category_id'] - 3,
                                         'area': annotation['area'],
                                         'bbox': annotation['bbox']})

for image in list(image_annotations.keys())[:100]:
    print(f"{image}: {image_annotations[image]}")


with open("format.json", "w") as f:
    json.dump(image_annotations, f)
