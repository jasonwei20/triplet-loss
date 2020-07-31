import json

#cleaning up text
import re
def get_only_chars(line):

    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ") #replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +',' ',clean_line) #delete extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line

def clean_fewrel(input_file, train_output_file, test_output_file):

	d = json.load(open(input_file))

	categories = d.keys()
	category_to_id = {category: i for i, category in enumerate(list(sorted(list(categories))))}

	test_writer = open(test_output_file, 'w')
	train_writer = open(train_output_file, 'w')

	for category, instance_list in d.items():

		_id = category_to_id[category]

		test_instances = instance_list[:100]
		train_instances = instance_list[100:600]

		for instance in test_instances:

			instance_str = ' '.join([' '.join(instance['tokens']), 'head', instance['h'][0], 'tail', instance['t'][0]])
			instance_str = get_only_chars(instance_str)
			test_writer.write('\t'.join([str(_id), instance_str]) + '\n')


		for instance in train_instances:

			instance_str = ' '.join([' '.join(instance['tokens']), 'head', instance['h'][0], 'tail', instance['t'][0]])
			instance_str = get_only_chars(instance_str)
			train_writer.write('\t'.join([str(_id), instance_str]) + '\n')

if __name__ == "__main__":
	clean_fewrel('data/train_wiki.json', 'data/train.txt', 'data/test.txt')